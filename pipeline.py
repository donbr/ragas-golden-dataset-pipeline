"""
RAGAS Evaluation Pipeline with Prefect

This is the main orchestration flow that coordinates all evaluation
components in a modular, configurable way.
"""

import os
import sys
import time
import subprocess
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_markdown_artifact, create_table_artifact, create_link_artifact
from prefect.cache_policies import NO_CACHE
from prefect.events import emit_event

# Import component modules
from libs.evaluation_core.config import setup_global_evaluation_config, create_execution_metadata
from libs.evaluation_data.loaders import load_test_dataset
from libs.evaluation_retrieval.api import (
    prepare_api_server as api_prepare_server,
    shutdown_api_server as api_shutdown_server,
    evaluate_retriever
)
from libs.evaluation_retrieval.metrics import calculate_metrics
from libs.evaluation_results.storage import store_evaluation_results
from libs.evaluation_results.reporting import generate_evaluation_report


@task(
    name="validate-environment",
    description="Validates that the environment is properly configured",
    retries=1,
    retry_delay_seconds=5,
    tags=["setup"]
)
def validate_environment(required_packages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validates that the environment has all required packages installed.
    
    Args:
        required_packages: List of required package names
        
    Returns:
        Dictionary with validation results
    """
    logger = get_run_logger()
    logger.info("Validating execution environment")
    
    # Default required packages if none provided
    if required_packages is None:
        required_packages = ["prefect", "pandas", "matplotlib", "requests"]
    
    # Check for each package
    results = {}
    for package in required_packages:
        try:
            __import__(package)
            results[package] = "installed"
        except ImportError:
            logger.error(f"Required package '{package}' is not installed")
            results[package] = "missing"
    
    # Check for RAGAS separately
    try:
        __import__("ragas")
        results["ragas"] = "installed"
    except ImportError:
        logger.warning("RAGAS not installed. Limited evaluation features will be available.")
        results["ragas"] = "missing"
    
    # Overall status
    all_required = all(results.get(pkg) == "installed" for pkg in required_packages)
    
    if all_required:
        logger.info("All required packages are installed")
    else:
        missing = [pkg for pkg, status in results.items() if status == "missing" and pkg in required_packages]
        logger.warning(f"Missing required packages: {', '.join(missing)}")
    
    # Include system information
    metadata = create_execution_metadata()
    results["metadata"] = metadata
    results["all_required_installed"] = all_required
    
    return results


@flow(
    name="ragas-evaluation-pipeline",
    description="Evaluates and compares RAG retrievers using test datasets",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def ragas_evaluation_pipeline(
    retrievers: List[str],
    test_dataset_path: str = "",
    hf_dataset_repo: str = "",
    dataset_size_limit: int = 0,
    api_port: int = 8000,
    output_dir: str = "evaluation_results/",
    llm_model: str = "gpt-4.1-mini",
    component_enable: Optional[Dict[str, bool]] = None,
    include_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Main flow that coordinates the evaluation of multiple retrievers.
    
    Args:
        retrievers: List of retriever types to evaluate
        test_dataset_path: Path to local test dataset file (JSON)
        hf_dataset_repo: HuggingFace dataset repository path (alternative to local)
        dataset_size_limit: Maximum number of examples to evaluate
        api_port: Port for the API server
        output_dir: Directory to store evaluation results
        llm_model: Model to use for RAGAS evaluation metrics
        component_enable: Dictionary controlling which components to enable
        include_visualizations: Whether to include visualizations in the report
        
    Returns:
        Dictionary with evaluation results and report info
    """
    logger = get_run_logger()
    logger.info(f"Starting RAGAS evaluation pipeline for {len(retrievers)} retrievers")
    
    # Set default component flags if not provided
    if component_enable is None:
        component_enable = {
            "validation": True,
            "api_server": True,
            "evaluation": True,
            "metrics": True,
            "results": True,
            "report": True
        }
    
    # Track execution progress
    results = {}
    all_metrics = {}
    
    # 1. Validate environment if enabled
    if component_enable.get("validation", True):
        env_validation = validate_environment()
        
        # Log validation results as markdown artifact
        markdown_content = f"""
        # Environment Validation Results
        
        ## System Information
        - **Python Version**: {env_validation['metadata']['python_version']}
        - **Prefect Version**: {env_validation['metadata']['prefect_version']}
        - **Execution ID**: {env_validation['metadata']['execution_id']}
        
        ## Required Packages
        | Package | Status |
        | ------- | ------ |
        """
        
        for pkg, status in env_validation.items():
            if pkg not in ["metadata", "all_required_installed"]:
                emoji = "✅" if status == "installed" else "❌"
                markdown_content += f"| {pkg} | {emoji} {status} |\n"
        
        create_markdown_artifact(
            key="environment-validation",
            markdown=markdown_content,
            description="Validation results for the execution environment"
        )
    else:
        logger.info("Environment validation disabled, skipping")
    
    # 2. Setup global evaluation configuration
    evaluation_config = setup_global_evaluation_config(
        llm_model=llm_model,
        max_workers=min(8, len(retrievers))
    )
    
    logger.info(f"Global evaluation config status: {evaluation_config.get('status', 'unknown')}")
    
    if evaluation_config.get("status") == "unavailable":
        logger.warning("RAGAS is not available. Some evaluation features will be limited.")
        create_markdown_artifact(
            key="ragas-status",
            markdown="# ⚠️ RAGAS Not Available\n\nRAGAS is not installed or configured. Evaluation will be limited to basic metrics.\n\nTo enable full evaluation, install RAGAS: `pip install ragas`",
            description="RAGAS availability status"
        )
    
    # 3. Load test dataset
    dataset = load_test_dataset(
        test_dataset_path=test_dataset_path,
        hf_dataset_repo=hf_dataset_repo,
        dataset_size_limit=dataset_size_limit
    )
    
    # 4. Prepare API server if enabled
    server_info = None
    if component_enable.get("api_server", True):
        server_info = api_prepare_server(port=api_port)
    else:
        logger.info("API server management disabled, assuming server is already running")
        server_info = {
            "port": api_port,
            "url": f"http://localhost:{api_port}",
            "status": "external"
        }
    
    try:
        # 5. Evaluate each retriever
        if component_enable.get("evaluation", True):
            for retriever_type in retrievers:
                logger.info(f"Evaluating retriever: {retriever_type}")
                
                # Run evaluation
                evaluation_results = evaluate_retriever(
                    retriever_type=retriever_type,
                    dataset=dataset,
                    api_url=server_info["url"],
                    top_k=5
                )
                
                results[retriever_type] = evaluation_results
                
                # Calculate metrics
                if component_enable.get("metrics", True):
                    metrics = calculate_metrics(
                        retriever_result=evaluation_results,
                        config=evaluation_config
                    )
                    
                    all_metrics[retriever_type] = metrics
                    
                    # Store results
                    if component_enable.get("results", True):
                        storage_info = store_evaluation_results(
                            retriever_type=retriever_type,
                            metrics=metrics,
                            output_dir=output_dir
                        )
        else:
            logger.info("Retriever evaluation disabled, skipping")
        
        # 6. Generate comparative report if multiple retrievers were evaluated
        report_info = None
        if component_enable.get("report", True) and len(all_metrics) > 0:
            if len(all_metrics) > 1:
                logger.info(f"Generating comparative report for {len(all_metrics)} retrievers")
                report_info = generate_evaluation_report(
                    results=all_metrics,
                    output_dir=output_dir,
                    include_visualizations=include_visualizations
                )
            else:
                logger.info("Only one retriever evaluated, skipping comparative report")
                # Still create a simple report for the single retriever
                single_retriever = list(all_metrics.keys())[0]
                report_info = {
                    "single_retriever": single_retriever,
                    "metrics": all_metrics[single_retriever]
                }
        else:
            logger.info("Report generation disabled, skipping")
    
    finally:
        # 7. Shutdown API server if we started it
        if server_info and component_enable.get("api_server", True) and server_info.get("status") != "external":
            api_shutdown_server(server_info)
    
    # 8. Return combined results
    emit_event(
        event="retriever-evaluation/pipeline/completed",
        resource={"prefect.resource.id": "ragas-evaluation-pipeline"},
        payload={
            "retrievers_evaluated": len(all_metrics),
            "dataset_size": dataset["metadata"]["example_count"] if dataset and "metadata" in dataset else 0
        }
    )
    
    logger.info(f"Evaluation pipeline completed for {len(all_metrics)} retrievers")
    
    return {
        "retrievers": retrievers,
        "metrics": all_metrics,
        "dataset_info": dataset["metadata"] if dataset and "metadata" in dataset else {},
        "report_info": report_info
    }


@task(
    name="shutdown-api-server",
    description="Shuts down the API server",
    tags=["api", "cleanup"],
    cache_policy=NO_CACHE,
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def shutdown_api_server(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shuts down the API server.
    
    Args:
        server_info: Server information from prepare_api_server
        
    Returns:
        Dictionary with shutdown status
    """
    logger = get_run_logger()
    port = server_info.get("port", 8000)
    logger.info(f"Shutting down API server on port {port}")
    
    # Try to gracefully terminate server by sending shutdown request
    try:
        url = server_info.get("url", f"http://localhost:{port}")
        shutdown_url = f"{url}/shutdown"
        response = requests.post(shutdown_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"API server on port {port} terminated gracefully")
            return {"status": "success", "port": port}
    except Exception as e:
        logger.warning(f"Could not send shutdown request: {str(e)}")
    
    # Try to forcefully kill the process (platform-specific)
    try:
        # This is a simplified approach - in a production environment
        # you'd want a more robust way to track and kill the specific process
        if sys.platform == "win32":
            subprocess.run(f'taskkill /F /PID $(netstat -ano | findstr ":{port}" | findstr "LISTENING" | awk "{{print $5}}")', shell=True)
            logger.info(f"Forced termination of process on port {port}")
        else:
            subprocess.run(f"kill $(lsof -t -i:{port})", shell=True)
            logger.info(f"Forced termination of process on port {port}")
    except Exception as e:
        logger.warning(f"Could not forcefully terminate process: {str(e)}")
    
    return {"status": "attempted", "port": port}


if __name__ == "__main__":
    """
    Command-line entry point for running the evaluation pipeline.
    
    Example:
        python pipeline.py bm25 dense hybrid --test_dataset_path data/test_questions.json
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG retrievers")
    parser.add_argument("retrievers", nargs="+", help="Retriever types to evaluate")
    parser.add_argument("--test_dataset_path", default="", help="Path to local test dataset file (JSON)")
    parser.add_argument("--hf_dataset_repo", default="", help="HuggingFace dataset repository path")
    parser.add_argument("--dataset_size_limit", type=int, default=0, help="Maximum number of examples to evaluate (0 for all)")
    parser.add_argument("--api_port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--output_dir", default="evaluation_results/", help="Directory to store evaluation results")
    parser.add_argument("--llm_model", default="gpt-4.1-mini", help="Model to use for RAGAS evaluation metrics")
    parser.add_argument("--disable_components", nargs="*", default=[], 
                        choices=["validation", "api_server", "evaluation", "metrics", "results", "report"],
                        help="Components to disable")
    parser.add_argument("--no_visualizations", action="store_true", help="Disable visualizations in the report")
    
    args = parser.parse_args()
    
    # Create component_enable dictionary based on disabled components
    component_enable = {component: component not in args.disable_components 
                       for component in ["validation", "api_server", "evaluation", "metrics", "results", "report"]}
    
    # Run the pipeline
    ragas_evaluation_pipeline(
        retrievers=args.retrievers,
        test_dataset_path=args.test_dataset_path,
        hf_dataset_repo=args.hf_dataset_repo,
        dataset_size_limit=args.dataset_size_limit,
        api_port=args.api_port,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        component_enable=component_enable,
        include_visualizations=not args.no_visualizations
    ) 