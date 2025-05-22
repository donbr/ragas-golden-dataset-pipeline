import os
import sys
import time
import json
import pickle
import argparse
import subprocess
from uuid import uuid4
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Prefect imports
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_markdown_artifact, create_table_artifact, create_link_artifact
from prefect.cache_policies import NO_CACHE

# Requests for API interaction
import requests

# Add RAGAS imports for evaluation metrics
try:
    from ragas.metrics import (
        faithfulness, 
        answer_relevancy, 
        context_relevancy, 
        context_recall,
        context_precision
    )
    from ragas.metrics.critique import harmfulness
    from ragas import evaluate, RunConfig
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not available, some evaluation metrics will be disabled")

# Set Prefect to use ephemeral mode before importing Prefect
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "True"
# Configure Prefect to persist results by default for better caching
os.environ["PREFECT_RESULTS_PERSIST_BY_DEFAULT"] = "true"

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from datetime import datetime, timedelta
import argparse
import concurrent.futures
import requests
from uuid import uuid4
import concurrent.futures
import time

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.cache_policies import NO_CACHE
from prefect.artifacts import create_markdown_artifact, create_link_artifact, create_table_artifact
from prefect.events import emit_event
from prefect.task_runners import ConcurrentTaskRunner
from prefect.exceptions import PrefectException
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.documents import Document
from huggingface_hub import HfApi, login

# Import functions from existing pipeline modules
from prefect_pipeline_v2 import validate_environment, download_pdfs, load_documents

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

SENSITIVE_VARS = {"OPENAI_API_KEY", "HF_TOKEN"}

def mask_value(value: str) -> str:
    """
    Mask sensitive values to protect privacy.
    
    Args:
        value: The string value to mask
        
    Returns:
        String with first 4 and last 4 characters preserved, middle replaced with asterisks
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]

def validate_path(path: str, must_exist: bool = False, create_if_missing: bool = False) -> Path:
    """
    Validate a path string and convert to Path object.
    
    Args:
        path: String path to validate
        must_exist: If True, path must exist
        create_if_missing: If True, create directory if it doesn't exist
        
    Returns:
        Path object
        
    Raises:
        ValueError: If path doesn't exist and must_exist=True
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
        
    if create_if_missing and not path_obj.exists():
        try:
            if path_obj.suffix:  # Has extension, so create parent directory
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            else:  # No extension, assume directory
                path_obj.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create directory for {path}: {str(e)}")
            
    return path_obj

def validate_int_range(value: int, name: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """
    Validate an integer value within a range.
    
    Args:
        value: Integer value to validate
        name: Name of the parameter (for error messages)
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        The validated integer value
        
    Raises:
        ValueError: If value is outside specified range
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be at least {min_val}, got {value}")
        
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be at most {max_val}, got {value}")
        
    return value

def create_execution_metadata() -> Dict[str, Any]:
    """
    Create metadata about the execution environment.
    
    Returns:
        Dictionary with execution metadata
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "execution_id": str(uuid4()),
        "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
        "prefect_version": os.environ.get("PREFECT_VERSION", "unknown"),
    }

# ------------------------------------------------------------------------------
# Global Evaluation Configuration
# ------------------------------------------------------------------------------

@task(
    name="setup-global-evaluation-config",
    description="Sets up global evaluation configuration with rate limiting",
    retries=2,
    retry_delay_seconds=5,
    tags=["setup", "configuration"],
    task_run_name="Set up global evaluation configuration"
)
def setup_global_evaluation_config(
    api_rate_limit: float = 5.0,  # Requests per second
    llm_rate_limit: float = 3.0,  # Requests per second
    evaluation_batch_size: int = 5,
    parallel_evaluations: int = 2
) -> Dict[str, Any]:
    """
    Sets up global configuration for the evaluation pipeline with rate limiting.
    
    Args:
        api_rate_limit: Maximum API requests per second
        llm_rate_limit: Maximum LLM requests per second
        evaluation_batch_size: Number of questions to evaluate in a batch
        parallel_evaluations: Maximum number of parallel evaluations
        
    Returns:
        Dictionary with global configuration settings
    """
    logger = get_run_logger()
    logger.info(f"Setting up global evaluation configuration")
    
    config = {
        "rate_limits": {
            "api": {
                "requests_per_second": api_rate_limit,
                "min_interval": 1.0 / api_rate_limit if api_rate_limit > 0 else 0,
                "last_request_time": {},  # Will track last request time per endpoint
            },
            "llm": {
                "requests_per_second": llm_rate_limit,
                "min_interval": 1.0 / llm_rate_limit if llm_rate_limit > 0 else 0,
                "last_request_time": datetime.now().timestamp(),
            }
        },
        "batching": {
            "batch_size": evaluation_batch_size,
            "parallel_evaluations": parallel_evaluations,
        },
        "execution": {
            "start_time": datetime.now().isoformat(),
            "id": str(uuid4()),
        }
    }
    
    logger.info(f"API rate limit: {api_rate_limit} requests/second (interval: {config['rate_limits']['api']['min_interval']:.4f}s)")
    logger.info(f"LLM rate limit: {llm_rate_limit} requests/second (interval: {config['rate_limits']['llm']['min_interval']:.4f}s)")
    logger.info(f"Batch size: {evaluation_batch_size}, Parallel evaluations: {parallel_evaluations}")
    
    # Create artifact with configuration settings
    create_table_artifact(
        key="global-evaluation-config",
        table={
            "columns": ["Parameter", "Value"],
            "data": [
                ["API Rate Limit", f"{api_rate_limit} req/s"],
                ["LLM Rate Limit", f"{llm_rate_limit} req/s"],
                ["Evaluation Batch Size", evaluation_batch_size],
                ["Parallel Evaluations", parallel_evaluations],
                ["Execution ID", config["execution"]["id"]],
                ["Start Time", config["execution"]["start_time"]]
            ]
        },
        description="Global evaluation configuration"
    )
    
    return config

def apply_rate_limiting(
    config: Dict[str, Any],
    rate_limit_type: str = "api",
    endpoint: str = "default"
) -> None:
    """
    Applies rate limiting based on configuration.
    
    Args:
        config: Global configuration dictionary
        rate_limit_type: Type of rate limit to apply ("api" or "llm")
        endpoint: API endpoint or operation name (for tracking separate limits)
        
    Returns:
        None (sleeps if needed to maintain rate limit)
    """
    if rate_limit_type not in config["rate_limits"]:
        return
        
    rate_config = config["rate_limits"][rate_limit_type]
    min_interval = rate_config["min_interval"]
    
    if min_interval <= 0:
        return  # No rate limiting needed
        
    # Get current time
    current_time = datetime.now().timestamp()
    
    # For API endpoints, track per-endpoint timing
    if rate_limit_type == "api":
        if endpoint not in rate_config["last_request_time"]:
            rate_config["last_request_time"][endpoint] = 0
            
        last_time = rate_config["last_request_time"][endpoint]
    else:
        last_time = rate_config["last_request_time"]
        
    # Calculate time since last request
    elapsed = current_time - last_time
    
    # If not enough time has elapsed, sleep
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        time.sleep(sleep_time)
        
    # Update last request time
    if rate_limit_type == "api":
        rate_config["last_request_time"][endpoint] = datetime.now().timestamp()
    else:
        rate_config["last_request_time"] = datetime.now().timestamp()

# ------------------------------------------------------------------------------
# API Server Management
# ------------------------------------------------------------------------------

@task(
    name="prepare-api-server",
    description="Prepares the API server for evaluation",
    retries=3,
    retry_delay_seconds=10,
    tags=["api", "setup"],
    task_run_name="Prepare API server",
    cache_policy=NO_CACHE
)
def prepare_api_server(
    server_port: int = 8000,
    start_server: bool = True,
    wait_for_ready: bool = True,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Prepares the API server for evaluation, optionally starting it as a subprocess.
    
    Args:
        server_port: Port for the API server
        start_server: Whether to start the server as a subprocess
        wait_for_ready: Whether to wait for the server to be ready
        timeout: Timeout in seconds to wait for server to start
        
    Returns:
        Dictionary with API server information
    """
    logger = get_run_logger()
    server_process = None
    server_url = f"http://localhost:{server_port}"
    
    try:
        # Check if server is already running
        try:
            logger.info(f"Checking if API server is already running at {server_url}")
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"API server is already running at {server_url}")
                return {
                    "url": server_url,
                    "started": False,
                    "ready": True,
                    "process": None
                }
        except Exception:
            logger.info(f"API server not running on {server_url}")
            
        # Start server if requested
        if start_server:
            logger.info(f"Starting API server on port {server_port}")
            
            # Get path to run.py
            script_dir = Path(__file__).parent
            run_script = script_dir / "run.py"
            
            if not run_script.exists():
                # Try to find run.py in the current directory or src directory
                if Path("run.py").exists():
                    run_script = Path("run.py")
                elif Path("src/run.py").exists():
                    run_script = Path("src/run.py")
                else:
                    raise FileNotFoundError(f"Could not find run.py in expected locations")
            
            # Start server as a subprocess
            server_process = subprocess.Popen(
                ["python", str(run_script)],
                env={**os.environ, "PORT": str(server_port)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"API server process started with PID {server_process.pid}")
            
            # Wait for server to be ready if requested
            if wait_for_ready:
                logger.info(f"Waiting for API server to be ready (timeout: {timeout}s)")
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        response = requests.get(f"{server_url}/health", timeout=5)
                        if response.status_code == 200:
                            logger.info(f"API server is ready at {server_url}")
                            break
                    except Exception:
                        pass
                    
                    time.sleep(1)
                    
                # Check if we timed out
                if time.time() - start_time >= timeout:
                    logger.warning(f"Timed out waiting for API server to be ready")
                    # Don't raise an exception, continue with evaluation
            
            return {
                "url": server_url,
                "started": True,
                "ready": wait_for_ready,
                "process": server_process
            }
        else:
            # Return info without starting server
            return {
                "url": server_url,
                "started": False,
                "ready": False,
                "process": None
            }
            
    except Exception as e:
        logger.error(f"Error preparing API server: {str(e)}")
        if server_process is not None:
            logger.info(f"Terminating API server process")
            server_process.terminate()
            
        create_markdown_artifact(
            key="api-server-error",
            markdown=f"# ❌ API Server Preparation Failed\n\n**Error:**\n```\n{str(e)}\n```",
            description="API server preparation error"
        )
        
        raise RuntimeError(f"Failed to prepare API server: {str(e)}")

@task(
    name="cleanup-api-server",
    description="Cleans up the API server process",
    tags=["api", "cleanup"],
    task_run_name="Clean up API server",
    cache_policy=NO_CACHE
)
def cleanup_api_server(server_info: Dict[str, Any]) -> None:
    """
    Cleans up the API server process if it was started by prepare_api_server.
    
    Args:
        server_info: Server information from prepare_api_server
        
    Returns:
        None
    """
    logger = get_run_logger()
    
    try:
        process = server_info.get("process")
        if process is not None:
            logger.info(f"Terminating API server process (PID: {process.pid})")
            process.terminate()
            try:
                process.wait(timeout=10)
                logger.info(f"API server process terminated successfully")
            except subprocess.TimeoutExpired:
                logger.warning(f"API server process did not terminate, forcing...")
                process.kill()
                logger.info(f"API server process killed")
    except Exception as e:
        logger.warning(f"Error cleaning up API server: {str(e)}")
        # Don't raise exception, just log the warning

# ------------------------------------------------------------------------------
# Retriever Evaluation
# ------------------------------------------------------------------------------

@task(
    name="get-available-retrievers",
    description="Gets list of available retrievers from API",
    retries=3,
    retry_delay_seconds=5,
    tags=["api", "retriever"],
    task_run_name="Get available retrievers",
    cache_policy=NO_CACHE
)
def get_available_retrievers(
    server_info: Dict[str, Any],
    global_config: Dict[str, Any]
) -> List[str]:
    """
    Gets list of available retrievers from the API.
    
    Args:
        server_info: Server information from prepare_api_server
        global_config: Global configuration for rate limiting
        
    Returns:
        List of available retriever types
    """
    logger = get_run_logger()
    
    try:
        server_url = server_info["url"]
        endpoint = "/retrievers"
        
        logger.info(f"Getting available retrievers from {server_url}{endpoint}")
        
        # Apply rate limiting
        apply_rate_limiting(global_config, rate_limit_type="api", endpoint="retrievers")
        
        # Make request to API
        try:
            response = requests.get(f"{server_url}{endpoint}", timeout=10)
            response.raise_for_status()
            
            retrievers = response.json().get("retrievers", [])
            logger.info(f"Found {len(retrievers)} available retrievers: {', '.join(retrievers)}")
            
            return retrievers
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to the API server: {str(e)}")
            
            # Provide fallback retriever types instead of failing
            fallback_retrievers = ["naive", "bm25", "contextual-compression", "ensemble", "self-query"]
            logger.info(f"Using fallback retriever types: {', '.join(fallback_retrievers)}")
            
            create_markdown_artifact(
                key="retrievers-fallback",
                markdown=f"# ⚠️ Using Fallback Retrievers\n\nCould not connect to API server to get available retrievers. Using fallback list:\n- {', '.join(fallback_retrievers)}",
                description="Fallback retrievers"
            )
            
            return fallback_retrievers
            
    except Exception as e:
        logger.error(f"Error getting available retrievers: {str(e)}")
        create_markdown_artifact(
            key="retrievers-error",
            markdown=f"# ❌ Failed to Get Available Retrievers\n\n**Error:**\n```\n{str(e)}\n```",
            description="Error getting available retrievers"
        )
        
        # Return minimal set of retrievers rather than failing
        return ["naive"]

@task(
    name="validate-retriever-types",
    description="Validates requested retriever types against available ones",
    tags=["validation", "retriever"],
    task_run_name="Validate retriever types"
)
def validate_retriever_types(
    requested_types: List[str],
    available_types: List[str]
) -> List[str]:
    """
    Validates requested retriever types against available ones.
    
    Args:
        requested_types: List of requested retriever types
        available_types: List of available retriever types
        
    Returns:
        List of validated retriever types
    """
    logger = get_run_logger()
    
    # If no specific types requested, use all available
    if not requested_types:
        logger.info(f"No specific retriever types requested, using all {len(available_types)} available types")
        return available_types
        
    # Validate each requested type
    valid_types = []
    invalid_types = []
    
    for rtype in requested_types:
        if rtype in available_types:
            valid_types.append(rtype)
        else:
            invalid_types.append(rtype)
            
    # Log results
    if valid_types:
        logger.info(f"Validated {len(valid_types)} retriever types: {', '.join(valid_types)}")
    
    if invalid_types:
        logger.warning(f"Invalid retriever types: {', '.join(invalid_types)}")
        create_markdown_artifact(
            key="invalid-retrievers",
            markdown=f"# ⚠️ Invalid Retriever Types\n\nThe following requested retriever types are not available:\n- {', '.join(invalid_types)}\n\nAvailable types are:\n- {', '.join(available_types)}",
            description="Invalid retriever types"
        )
        
    if not valid_types:
        logger.error(f"No valid retriever types found")
        create_markdown_artifact(
            key="no-valid-retrievers",
            markdown=f"# ❌ No Valid Retriever Types\n\nNone of the requested retriever types are available.\n\nRequested: {', '.join(requested_types)}\n\nAvailable: {', '.join(available_types)}",
            description="No valid retriever types"
        )
        
        raise ValueError(f"No valid retriever types found. Requested: {requested_types}, Available: {available_types}")
        
    return valid_types

@task(
    name="evaluate-retriever",
    description="Evaluates a single retriever type with test questions",
    retries=2,
    retry_delay_seconds=10,
    cache_policy=NO_CACHE,
    tags=["evaluation", "retriever"],
    task_run_name="Evaluate {retriever_type} retriever"
)
def evaluate_retriever(
    retriever_type: str,
    test_dataset: Any,
    server_info: Dict[str, Any],
    global_config: Dict[str, Any],
    max_questions: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates a single retriever type with test questions.
    
    Args:
        retriever_type: Retriever type to evaluate
        test_dataset: Test dataset with questions/answers
        server_info: Server information from prepare_api_server
        global_config: Global configuration for rate limiting
        max_questions: Maximum number of questions to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation results
    """
    logger = get_run_logger()
    logger.info(f"Evaluating retriever: {retriever_type}")
    
    start_time = time.time()
    server_url = server_info["url"]
    endpoint = f"/retrieve/{retriever_type}"
    
    # Get questions from test dataset
    try:
        # Extract questions from test dataset
        questions = []
        contexts = []
        answers = []
        
        if hasattr(test_dataset, "to_pandas"):
            # If test_dataset is a HuggingFace dataset
            df = test_dataset.to_pandas()
            
            # Check for question field (standard format)
            if "question" in df.columns:
                questions = df["question"].tolist()
            # Check for RAGAS format field
            elif "user_input" in df.columns:
                questions = df["user_input"].tolist()
            else:
                questions = []
                
            # Similarly handle contexts with different possible field names
            if "contexts" in df.columns:
                contexts = df["contexts"].tolist()
            elif "context" in df.columns:
                contexts = df["context"].tolist()
            elif "reference_contexts" in df.columns:
                contexts = df["reference_contexts"].tolist()
            else:
                contexts = []
                
            # Similarly handle answers with different possible field names
            if "answer" in df.columns:
                answers = df["answer"].tolist()
            elif "ground_truth" in df.columns:
                answers = df["ground_truth"].tolist()
            elif "reference" in df.columns:
                answers = df["reference"].tolist()
            else:
                answers = []
        elif hasattr(test_dataset, "test_data"):
            # If test_dataset is a RAGAS dataset
            for item in test_dataset.test_data:
                if hasattr(item, "question"):
                    questions.append(item.question)
                    if hasattr(item, "contexts"):
                        contexts.append(item.contexts)
                    if hasattr(item, "ground_truth"):
                        answers.append(item.ground_truth)
        else:
            # Try to access test dataset as dictionary or custom structure
            logger.debug(f"Trying to extract questions from custom dataset structure")
            if isinstance(test_dataset, dict) and "test_data" in test_dataset:
                for item in test_dataset["test_data"]:
                    if hasattr(item, "question"):
                        questions.append(item.question)
                        if hasattr(item, "contexts"):
                            contexts.append(item.contexts)
                        if hasattr(item, "ground_truth"):
                            answers.append(item.ground_truth)
    except Exception as e:
        logger.error(f"Error extracting questions from test dataset: {str(e)}")
        return {
            "retriever_type": retriever_type,
            "status": "error",
            "error": f"Failed to extract questions: {str(e)}",
            "query_count": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "execution_time_s": time.time() - start_time
        }
    
    # Limit questions if needed
    if max_questions and len(questions) > max_questions:
        logger.info(f"Limiting evaluation to {max_questions} questions (from {len(questions)} total)")
        questions = questions[:max_questions]
        if contexts:
            contexts = contexts[:max_questions]
        if answers:
            answers = answers[:max_questions]
    
    if not questions:
        logger.warning(f"No questions found in test dataset")
        return {
            "retriever_type": retriever_type,
            "status": "error",
            "error": "No questions found in test dataset",
            "query_count": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "execution_time_s": time.time() - start_time
        }
    
    logger.info(f"Evaluating {len(questions)} questions on retriever {retriever_type}")
    
    # Process questions in batches
    batch_size = global_config["batching"]["batch_size"]
    results = []
    successful_queries = 0
    failed_queries = 0
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_start = time.time()
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_questions)} questions")
        
        batch_results = []
        for q_idx, question in enumerate(batch_questions):
            question_start = time.time()
            try:
                # Apply rate limiting
                apply_rate_limiting(global_config, rate_limit_type="api", endpoint=f"retrieve/{retriever_type}")
                
                # Make request to API
                logger.debug(f"Sending question to {endpoint}: '{question[:50]}...'")
                response = requests.post(
                    f"{server_url}{endpoint}",
                    json={"query": question},
                    timeout=30
                )
                response.raise_for_status()
                
                # Process response
                result = response.json()
                result["question"] = question
                result["question_idx"] = i + q_idx
                if contexts and i + q_idx < len(contexts):
                    result["context"] = contexts[i + q_idx]
                if answers and i + q_idx < len(answers):
                    result["answer"] = answers[i + q_idx]
                
                result["query_time_s"] = time.time() - question_start
                batch_results.append(result)
                successful_queries += 1
                
                logger.debug(f"Successfully processed question {i + q_idx + 1}/{len(questions)}")
                
            except Exception as e:
                logger.error(f"Error processing question {i + q_idx + 1}/{len(questions)}: {str(e)}")
                batch_results.append({
                    "question": question,
                    "question_idx": i + q_idx,
                    "error": str(e),
                    "status": "error",
                    "query_time_s": time.time() - question_start
                })
                failed_queries += 1
        
        # Add batch results to overall results
        results.extend(batch_results)
        
        # Log batch completion
        batch_time = time.time() - batch_start
        logger.info(f"Completed batch {i//batch_size + 1} in {batch_time:.2f}s ({len(batch_results)} questions)")
    
    # Calculate evaluation time
    execution_time = time.time() - start_time
    
    # Create summary of the evaluation
    result_summary = {
        "retriever_type": retriever_type,
        "status": "completed",
        "query_count": len(questions),
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "execution_time_s": execution_time,
        "results": results
    }
    
    # Log completion
    logger.info(f"Completed evaluation of {retriever_type} in {execution_time:.2f}s")
    logger.info(f"Processed {successful_queries} queries successfully, {failed_queries} failed")
    
    return result_summary

@task(
    name="calculate-retrieval-metrics",
    description="Calculates metrics for retrieval results",
    tags=["evaluation", "metrics"],
    task_run_name="Calculate metrics for {retriever_type}"
)
def calculate_retrieval_metrics(
    evaluation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculates metrics for retrieval results.
    
    Args:
        evaluation_results: Results from evaluate_retriever
        
    Returns:
        Dictionary with calculated metrics
    """
    logger = get_run_logger()
    retriever_type = evaluation_results["retriever_type"]
    
    try:
        results = evaluation_results.get("results", [])
        if not results:
            logger.warning(f"No results found for {retriever_type}")
            return {
                "retriever_type": retriever_type,
                "status": "error",
                "error": "No results found",
                "metrics": {}
            }
        
        # Extract metrics
        successful_results = [r for r in results if "error" not in r]
        
        # Calculate retrieval statistics
        retrieval_stats = {}
        
        # Count responses with non-empty documents
        responses_with_docs = [r for r in successful_results if r.get("documents", [])]
        retrieval_stats["responses_with_documents"] = len(responses_with_docs)
        retrieval_stats["responses_without_documents"] = len(successful_results) - len(responses_with_docs)
        
        # Calculate average number of documents per response
        doc_counts = [len(r.get("documents", [])) for r in successful_results]
        retrieval_stats["avg_documents_per_query"] = sum(doc_counts) / len(doc_counts) if doc_counts else 0
        retrieval_stats["total_documents_retrieved"] = sum(doc_counts)
        
        # Calculate average query time
        query_times = [r.get("query_time_s", 0) for r in successful_results]
        retrieval_stats["avg_query_time_s"] = sum(query_times) / len(query_times) if query_times else 0
        retrieval_stats["max_query_time_s"] = max(query_times) if query_times else 0
        retrieval_stats["min_query_time_s"] = min(query_times) if query_times else 0
        
        # Create a summary of the calculated metrics
        metrics = {
            "retriever_type": retriever_type,
            "status": "completed",
            "retrieval_stats": retrieval_stats,
            "execution_info": {
                "query_count": evaluation_results["query_count"],
                "successful_queries": evaluation_results["successful_queries"],
                "failed_queries": evaluation_results["failed_queries"],
                "execution_time_s": evaluation_results["execution_time_s"]
            }
        }
        
        # Create artifact with metrics summary
        create_table_artifact(
            key=f"metrics-{retriever_type}",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Retriever Type", retriever_type],
                    ["Successful Queries", evaluation_results["successful_queries"]],
                    ["Failed Queries", evaluation_results["failed_queries"]],
                    ["Responses With Documents", retrieval_stats["responses_with_documents"]],
                    ["Responses Without Documents", retrieval_stats["responses_without_documents"]],
                    ["Avg Documents Per Query", f"{retrieval_stats['avg_documents_per_query']:.2f}"],
                    ["Total Documents Retrieved", retrieval_stats["total_documents_retrieved"]],
                    ["Avg Query Time (s)", f"{retrieval_stats['avg_query_time_s']:.4f}"],
                    ["Max Query Time (s)", f"{retrieval_stats['max_query_time_s']:.4f}"],
                    ["Min Query Time (s)", f"{retrieval_stats['min_query_time_s']:.4f}"],
                    ["Total Execution Time (s)", f"{evaluation_results['execution_time_s']:.2f}"]
                ]
            },
            description=f"Metrics for {retriever_type} retriever"
        )
        
        logger.info(f"Calculated metrics for {retriever_type}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {retriever_type}: {str(e)}")
        return {
            "retriever_type": retriever_type,
            "status": "error",
            "error": f"Error calculating metrics: {str(e)}",
            "metrics": {}
        }

# ------------------------------------------------------------------------------
# Reporting and Results
# ------------------------------------------------------------------------------

@task(
    name="generate-evaluation-report",
    description="Generates a comprehensive evaluation report",
    tags=["reporting", "evaluation"],
    task_run_name="Generate evaluation report"
)
def generate_evaluation_report(
    metrics_by_retriever: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates a comprehensive evaluation report.
    
    Args:
        metrics_by_retriever: Dictionary mapping retriever types to metrics
        
    Returns:
        Dictionary with report data
    """
    logger = get_run_logger()
    
    try:
        # Extract successful metrics
        successful_metrics = {
            rtype: metrics for rtype, metrics in metrics_by_retriever.items()
            if metrics.get("status") == "completed"
        }
        
        if not successful_metrics:
            logger.warning("No successful evaluations found for report generation")
            return {
                "status": "error",
                "error": "No successful evaluations found"
            }
            
        # Create comparison table
        retriever_types = list(successful_metrics.keys())
        
        comparison_data = []
        # Query counts
        comparison_data.append(["Successful Queries"] + [
            str(metrics["execution_info"]["successful_queries"]) 
            for metrics in successful_metrics.values()
        ])
        
        # Documents per query
        comparison_data.append(["Avg Documents Per Query"] + [
            f"{metrics['retrieval_stats']['avg_documents_per_query']:.2f}" 
            for metrics in successful_metrics.values()
        ])
        
        # Total documents retrieved
        comparison_data.append(["Total Documents"] + [
            str(metrics["retrieval_stats"]["total_documents_retrieved"]) 
            for metrics in successful_metrics.values()
        ])
        
        # Average query time
        comparison_data.append(["Avg Query Time (s)"] + [
            f"{metrics['retrieval_stats']['avg_query_time_s']:.4f}" 
            for metrics in successful_metrics.values()
        ])
        
        # Execution time
        comparison_data.append(["Execution Time (s)"] + [
            f"{metrics['execution_info']['execution_time_s']:.2f}" 
            for metrics in successful_metrics.values()
        ])
        
        # Create the comparison artifact
        create_table_artifact(
            key="retriever-comparison",
            table={
                "columns": ["Metric"] + retriever_types,
                "data": comparison_data
            },
            description="Comparison of retriever performance"
        )
        
        # Create the markdown report
        report_md = f"""# Retriever Evaluation Report

## Overview

This report compares {len(retriever_types)} different retriever types:
{', '.join(f'`{rtype}`' for rtype in retriever_types)}

## Key Findings

"""
        
        # Add best performer
        if len(retriever_types) > 1:
            # Best by query time
            query_times = [metrics["retrieval_stats"]["avg_query_time_s"] for metrics in successful_metrics.values()]
            fastest_idx = query_times.index(min(query_times))
            fastest_retriever = retriever_types[fastest_idx]
            
            # Best by document count
            doc_counts = [metrics["retrieval_stats"]["avg_documents_per_query"] for metrics in successful_metrics.values()]
            most_docs_idx = doc_counts.index(max(doc_counts))
            most_docs_retriever = retriever_types[most_docs_idx]
            
            report_md += f"""- Fastest retriever: `{fastest_retriever}` ({query_times[fastest_idx]:.4f}s per query)
- Most documents per query: `{most_docs_retriever}` ({doc_counts[most_docs_idx]:.2f} docs per query)
"""
        
        report_md += """
## Performance Comparison

| Retriever | Queries | Documents/Query | Total Docs | Query Time (s) | Execution Time (s) |
|-----------|---------|----------------|------------|----------------|-------------------|
"""
        
        for i, rtype in enumerate(retriever_types):
            metrics = successful_metrics[rtype]
            report_md += f"| {rtype} | {metrics['execution_info']['successful_queries']} | {metrics['retrieval_stats']['avg_documents_per_query']:.2f} | {metrics['retrieval_stats']['total_documents_retrieved']} | {metrics['retrieval_stats']['avg_query_time_s']:.4f} | {metrics['execution_info']['execution_time_s']:.2f} |\n"
            
        report_md += """
## Conclusion

This evaluation provides a comparison of different retriever types based on speed and document retrieval characteristics.
"""
        
        # Create the markdown artifact
        create_markdown_artifact(
            key="evaluation-report",
            markdown=report_md,
            description="Retriever evaluation report"
        )
        
        logger.info(f"Generated evaluation report for {len(retriever_types)} retrievers")
        
        return {
            "status": "completed",
            "retriever_count": len(retriever_types),
            "report": report_md
        }
        
    except Exception as e:
        logger.error(f"Error generating evaluation report: {str(e)}")
        return {
            "status": "error",
            "error": f"Failed to generate report: {str(e)}"
        }

@task(
    name="save-evaluation-results",
    description="Saves evaluation results to disk",
    retries=2,
    retry_delay_seconds=10,
    tags=["reporting", "output"],
    task_run_name="Save evaluation results"
)
def save_evaluation_results(
    results_by_retriever: Dict[str, Dict[str, Any]],
    metrics_by_retriever: Dict[str, Dict[str, Any]],
    output_dir: str
) -> str:
    """
    Saves evaluation results to disk.
    
    Args:
        results_by_retriever: Dictionary mapping retriever types to evaluation results
        metrics_by_retriever: Dictionary mapping retriever types to metrics
        output_dir: Directory to save results to
        
    Returns:
        Path to the saved results file
    """
    logger = get_run_logger()
    
    try:
        # Prepare output directory
        output_path = validate_path(output_dir, create_if_missing=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"evaluation_results_{timestamp}.json"
        
        # Prepare results structure
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "retrievers": {}
        }
        
        # For each retriever, compile data
        for rtype in results_by_retriever.keys():
            # Limit the size of saved results by excluding detailed data
            results = results_by_retriever[rtype]
            metrics = metrics_by_retriever.get(rtype, {})
            
            # Create simplified results with just summary info
            simplified_results = {
                "retriever_type": rtype,
                "status": results.get("status", "unknown"),
                "query_count": results.get("query_count", 0),
                "successful_queries": results.get("successful_queries", 0),
                "failed_queries": results.get("failed_queries", 0),
                "execution_time_s": results.get("execution_time_s", 0),
                # Don't include full results array to keep file size manageable
                "metrics": metrics.get("retrieval_stats", {})
            }
            
            save_data["retrievers"][rtype] = simplified_results
            
        # Write to file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        # Create artifact with link to the file
        file_uri = f"file://{os.path.abspath(results_file)}"
        create_link_artifact(
            key="evaluation-results-file",
            link=file_uri,
            link_text="Evaluation Results JSON File",
            description="Link to the saved evaluation results"
        )
        
        logger.info(f"Saved evaluation results to {results_file}")
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")
        create_markdown_artifact(
            key="save-results-error",
            markdown=f"# ❌ Failed to Save Results\n\n**Error:**\n```\n{str(e)}\n```",
            description="Error saving evaluation results"
        )
        raise RuntimeError(f"Failed to save evaluation results: {str(e)}")

# ------------------------------------------------------------------------------
# RAGAS Generator Functions
# ------------------------------------------------------------------------------

@task(
    name="initialize-ragas-generator",
    description="Initializes a RAGAS test generator with LLM and embedding models",
    retries=2,
    retry_delay_seconds=30,
    tags=["ragas", "initialization"],
    cache_policy=NO_CACHE
)
def initialize_ragas_generator(
    llm_model: str,
    embedding_model: str
) -> TestsetGenerator:
    """
    Initializes a RAGAS test generator with LLM and embedding models.
    
    Args:
        llm_model: OpenAI model to use for generation
        embedding_model: Embedding model to use
        
    Returns:
        Initialized RAGAS TestsetGenerator
    """
    logger = get_run_logger()
    logger.info(f"Initializing RAGAS generator with LLM: {llm_model}, Embeddings: {embedding_model}")
    
    try:
        # Initialize LLM and embedding models
        llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
        
        # Create TestsetGenerator
        generator = TestsetGenerator(llm=llm, embedding_model=emb)
        
        logger.info("RAGAS generator initialized successfully")
        return generator
        
    except Exception as e:
        logger.error(f"Error initializing RAGAS generator: {str(e)}")
        raise RuntimeError(f"Failed to initialize RAGAS generator: {str(e)}")

@task(
    name="generate-testset",
    description="Generates a RAGAS test dataset from documents",
    retries=2,
    retry_delay_seconds=30,
    cache_policy=NO_CACHE,
    tags=["ragas", "generation"]
)
def generate_testset(
    generator: TestsetGenerator,
    docs: List[Document],
    testset_size: int
) -> Tuple[Any, Any]:
    """
    Generates a RAGAS test dataset from documents.
    
    Args:
        generator: Initialized RAGAS TestsetGenerator
        docs: List of documents to use for testset generation
        testset_size: Number of QA pairs to generate
        
    Returns:
        Tuple of (dataset, knowledge_graph)
    """
    logger = get_run_logger()
    logger.info(f"Generating testset with {testset_size} samples from {len(docs)} documents")
    
    try:
        # Generate testset
        dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        
        # Get knowledge graph
        kg = generator.knowledge_graph
        
        logger.info(f"Successfully generated testset with {testset_size} samples")
        return dataset, kg
        
    except Exception as e:
        logger.error(f"Error generating testset: {str(e)}")
        raise RuntimeError(f"Failed to generate testset: {str(e)}")

@task(
    name="save-knowledge-graph",
    description="Saves a knowledge graph to JSON file",
    tags=["ragas", "output"]
)
def save_knowledge_graph(
    kg: Any,
    output_path: str
) -> str:
    """
    Saves a knowledge graph to JSON file.
    
    Args:
        kg: Knowledge graph to save
        output_path: Path to save the knowledge graph to
        
    Returns:
        Path to the saved knowledge graph
    """
    logger = get_run_logger()
    
    try:
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save knowledge graph
        kg.save(output_path)
        logger.info(f"Knowledge graph saved to {output_path}")
        
        # Create artifact
        create_link_artifact(
            key="kg-json",
            link=f"file://{os.path.abspath(output_path)}",
            link_text="Knowledge Graph JSON File",
            description="Link to the generated knowledge graph"
        )
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving knowledge graph: {str(e)}")
        raise RuntimeError(f"Failed to save knowledge graph: {str(e)}")

# ------------------------------------------------------------------------------
# Main Flow - Update to use our new components
# ------------------------------------------------------------------------------

@flow(
    name="RAGAS Evaluation Pipeline",
    description="Evaluates different retrieval strategies using test datasets",
    log_prints=True,
    version=os.environ.get("PIPELINE_VERSION", "1.0.0"),
    task_runner=ConcurrentTaskRunner(),
    validate_parameters=True
)
def evaluation_pipeline(
    test_dataset_path: str = "",
    hf_dataset_repo: str = "",
    output_dir: str = "evaluation_results/",
    retriever_types: List[str] = [],
    max_questions: int = 20,
    api_port: int = 8000,
    api_rate_limit: float = 5.0,
    llm_rate_limit: float = 3.0
) -> Dict[str, Any]:
    """
    Evaluates different retrieval strategies using test datasets.
    
    Args:
        test_dataset_path: Path to local test dataset file
        hf_dataset_repo: HuggingFace dataset repository
        output_dir: Directory for output files
        retriever_types: List of retriever types to evaluate (None for all)
        max_questions: Maximum questions to evaluate per retriever
        api_port: Port for the API server
        api_rate_limit: API rate limit in requests per second
        llm_rate_limit: LLM rate limit in requests per second
        
    Returns:
        Dictionary with evaluation results and statistics
    """
    logger = get_run_logger()
    execution_start = time.time()
    logger.info(f"🚀 Starting RAGAS Evaluation Pipeline")
    
    # --- 1. Setup Phase ---
    
    # Validate environment
    logger.info("Validating environment...")
    env_vars = validate_environment()
    
    # Set up global configuration
    logger.info("Setting up global evaluation configuration...")
    global_config = setup_global_evaluation_config(
        api_rate_limit=api_rate_limit,
        llm_rate_limit=llm_rate_limit,
        evaluation_batch_size=5,
        parallel_evaluations=2
    )
    
    # --- 2. Data Loading Phase ---
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = None
    dataset_source = None
    
    if test_dataset_path:
        # Load local dataset
        logger.info(f"Loading local dataset from {test_dataset_path}")
        test_path = validate_path(test_dataset_path, must_exist=True)
        
        if test_path.suffix == '.json':
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                # Convert to a format compatible with our evaluate_retriever function
                # This is a simplified approach, you may need to adapt based on your data structure
                test_dataset = {
                    'test_data': [
                        type('TestItem', (), {
                            'question': item.get('question', ''),
                            'contexts': item.get('contexts', []),
                            'ground_truth': item.get('answer', '')
                        }) for item in test_data
                    ]
                }
                dataset_source = f"local:{test_path}"
                
        elif test_path.suffix == '.pkl':
            with open(test_path, 'rb') as f:
                test_dataset = pickle.load(f)
                dataset_source = f"local:{test_path}"
        else:
            logger.error(f"Unsupported dataset format: {test_path.suffix}")
            raise ValueError(f"Unsupported dataset format: {test_path.suffix}")
            
    elif hf_dataset_repo:
        # Load HuggingFace dataset
        logger.info(f"Loading dataset from HuggingFace: {hf_dataset_repo}")
        from datasets import load_dataset
        test_dataset = load_dataset(hf_dataset_repo)
        dataset_source = f"huggingface:{hf_dataset_repo}"
    else:
        # Generate a small test dataset using existing pipeline
        logger.info("No dataset specified, downloading PDFs and generating test dataset...")
        docs_path = download_pdfs("data/")
        docs = load_documents(docs_path)
        
        # Initialize RAGAS generator and create dataset
        generator = initialize_ragas_generator(
            llm_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small"
        )
        
        # Generate small test dataset (5 questions)
        dataset, kg = generate_testset(generator, docs, 5)
        test_dataset = dataset
        dataset_source = "generated"
        
        # Save knowledge graph
        save_knowledge_graph(kg, "output/test_kg.json")
    
    if not test_dataset:
        logger.error("Failed to load or generate test dataset")
        create_markdown_artifact(
            key="pipeline-error",
            markdown="# ❌ Pipeline Aborted\n\nFailed to load or generate test dataset.",
            description="Pipeline error"
        )
        return {"status": "failed", "reason": "No test dataset available"}
        
    logger.info(f"Successfully loaded test dataset from {dataset_source}")
    
    # --- 3. API Server Phase ---
    
    # Prepare API server
    logger.info("Preparing API server...")
    server_info = prepare_api_server(server_port=api_port)
    
    # Get available retrievers
    logger.info("Getting available retrievers...")
    available_retrievers = get_available_retrievers(server_info, global_config)
    
    # Validate requested retriever types
    logger.info("Validating retriever types...")
    valid_retrievers = validate_retriever_types(retriever_types, available_retrievers)
    
    # --- 4. Evaluation Phase ---
    
    # Evaluate each retriever type
    logger.info(f"Starting evaluation of {len(valid_retrievers)} retrievers...")
    
    # Create futures for parallel evaluation
    evaluation_futures = {}
    for rtype in valid_retrievers:
        evaluation_futures[rtype] = evaluate_retriever.submit(
            retriever_type=rtype,
            test_dataset=test_dataset,
            server_info=server_info,
            global_config=global_config,
            max_questions=max_questions
        )
    
    # Wait for results and calculate metrics
    logger.info("Collecting evaluation results...")
    evaluation_results = {}
    metrics_results = {}
    
    for rtype, future in evaluation_futures.items():
        try:
            # Get evaluation result
            result = future.result()
            evaluation_results[rtype] = result
            
            # Calculate metrics
            metrics = calculate_retrieval_metrics(result)
            metrics_results[rtype] = metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {rtype}: {str(e)}")
            evaluation_results[rtype] = {
                "retriever_type": rtype,
                "status": "error",
                "error": str(e),
                "query_count": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "execution_time_s": 0
            }
            metrics_results[rtype] = {
                "retriever_type": rtype,
                "status": "error",
                "error": str(e),
                "metrics": {}
            }
    
    # --- 5. Reporting Phase ---
    
    # Generate evaluation report
    logger.info("Generating evaluation report...")
    report = generate_evaluation_report(metrics_results)
    
    # Save evaluation results
    logger.info("Saving evaluation results...")
    results_file = save_evaluation_results(evaluation_results, metrics_results, output_dir)
    
    # Clean up API server
    logger.info("Cleaning up API server...")
    cleanup_api_server(server_info)
    
    # --- 6. Completion ---
    
    # Calculate pipeline execution time
    execution_time_s = time.time() - execution_start
    logger.info(f"✅ RAGAS Evaluation Pipeline completed successfully in {execution_time_s:.2f}s")
    
    # Final success artifact
    create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"""# Pipeline Execution Summary

## Success! ✅

The RAGAS Evaluation Pipeline completed successfully in {execution_time_s:.2f} seconds.

## Data
- Dataset Source: {dataset_source}
- Questions Evaluated: {max_questions} per retriever

## Retrievers Evaluated
{', '.join(valid_retrievers)}

## Output
- Results saved to: `{results_file}`

## Next Steps
View the full evaluation report for detailed comparison of retrievers.
""",
        description="Pipeline execution summary"
    )
    
    # Return results summary
    return {
        "status": "success",
        "retrievers_evaluated": valid_retrievers,
        "dataset_source": dataset_source,
        "max_questions": max_questions,
        "results_file": results_file,
        "execution_time_s": round(execution_time_s, 2)
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Pipeline")
    
    # Dataset parameters
    parser.add_argument("--test-dataset", type=str, default="",
                        help="Path to local test dataset file (JSON or pickle)")
    parser.add_argument("--hf-dataset", type=str, default="",
                        help="HuggingFace dataset repository")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="evaluation_results/",
                        help="Directory for output files")
    
    # Evaluation parameters
    parser.add_argument("--retrievers", type=str, default="",
                        help="Comma-separated list of retriever types to evaluate (empty for all)")
    parser.add_argument("--max-questions", type=int, default=20,
                        help="Maximum questions to evaluate per retriever")
    
    # API parameters
    parser.add_argument("--api-port", type=int, default=8000,
                        help="Port for the API server")
    parser.add_argument("--api-rate-limit", type=float, default=5.0,
                        help="API rate limit in requests per second")
    parser.add_argument("--llm-rate-limit", type=float, default=3.0,
                        help="LLM rate limit in requests per second")
    
    # Other parameters
    parser.add_argument("--version", action="store_true",
                        help="Show pipeline version and exit")
    
    args = parser.parse_args()
    
    # Get pipeline version from environment or default
    pipeline_version = os.environ.get("PIPELINE_VERSION", "1.0.0")
    
    # Handle --version flag
    if args.version:
        print(f"RAGAS Evaluation Pipeline version {pipeline_version}")
        exit(0)
    
    # Parse retriever types
    if args.retrievers:
        retriever_types = [t.strip() for t in args.retrievers.split(",")]
    else:
        retriever_types = []
    
    # Print banner
    print("Starting RAGAS Evaluation Pipeline v" + pipeline_version)
    print("=========================================")
    print(f"- Test Dataset: {args.test_dataset or args.hf_dataset or 'Generating test dataset'}")
    print(f"- Output Directory: {args.output_dir}")
    print(f"- Retrievers: {retriever_types or 'All available'}")
    print(f"- Max Questions: {args.max_questions}")
    print(f"- API Port: {args.api_port}")
    print(f"- API Rate Limit: {args.api_rate_limit} req/s")
    print(f"- LLM Rate Limit: {args.llm_rate_limit} req/s")
    print("=========================================")
    
    # Run the pipeline
    try:
        result = evaluation_pipeline(
            test_dataset_path=args.test_dataset,
            hf_dataset_repo=args.hf_dataset,
            output_dir=args.output_dir,
            retriever_types=retriever_types,
            max_questions=args.max_questions,
            api_port=args.api_port,
            api_rate_limit=args.api_rate_limit,
            llm_rate_limit=args.llm_rate_limit
        )
        
        # Print summary of results
        if result and result.get("status") == "success":
            print("\nPipeline completed successfully!")
            print(f"- Retrievers evaluated: {', '.join(result.get('retrievers_evaluated', []))}")
            print(f"- Results saved to: {result.get('results_file', args.output_dir)}")
            print(f"- Total execution time: {result.get('execution_time_s', 0):.2f}s")
        else:
            print("\nPipeline completed with issues:")
            print(f"- Status: {result.get('status', 'unknown')}")
            print(f"- Reason: {result.get('reason', 'unknown')}")
            exit(1)
            
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        exit(1)
