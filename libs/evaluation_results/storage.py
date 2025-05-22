"""
Evaluation results storage utilities.

This module provides tasks for storing evaluation results
and creating artifacts.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from prefect import task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact, create_link_artifact


@task(
    name="store-evaluation-results",
    description="Saves evaluation results to disk and creates artifacts",
    retries=2,
    retry_delay_seconds=5,
    tags=["results", "storage"],
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def store_evaluation_results(
    retriever_type: str,
    metrics: Dict[str, Any],
    output_dir: str = "evaluation_results/",
    include_full_results: bool = False
) -> Dict[str, Any]:
    """
    Saves evaluation results to disk and creates Prefect artifacts.
    
    Args:
        retriever_type: Type of retriever evaluated
        metrics: Evaluation metrics to store
        output_dir: Directory to store results
        include_full_results: Whether to include full results (may be large)
        
    Returns:
        Dictionary with storage information
    """
    logger = get_run_logger()
    logger.info(f"Storing evaluation results for {retriever_type} retriever")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results filename
    filename = f"{retriever_type}_results_{timestamp}.json"
    result_path = os.path.join(output_dir, filename)
    
    # Create simplified results with just summary info
    simplified_results = {
        "retriever_type": retriever_type,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: v for k, v in metrics.items() if k != "results"}
    }
    
    # Add full results if requested (may be large)
    if include_full_results and "results" in metrics:
        simplified_results["results"] = metrics["results"]
    
    # Save to disk
    with open(result_path, "w") as f:
        json.dump(simplified_results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {result_path}")
    
    # Extract flat metrics for table display
    metric_values = {k: v for k, v in metrics.items() 
                     if k not in ["results", "retriever_type"] and not isinstance(v, (dict, list))}
    
    # Convert to a format suitable for create_table_artifact
    table_data = []
    for metric_name, metric_value in sorted(metric_values.items()):
        if isinstance(metric_value, float):
            formatted_value = f"{metric_value:.4f}"
        else:
            formatted_value = str(metric_value)
        table_data.append([metric_name, formatted_value])
    
    # Add retriever type as a row
    table_data.append(["retriever_type", retriever_type])
    
    # Create artifact with the metrics table in the proper format
    artifact_key = f"metrics-{retriever_type.replace('_', '-')}-{timestamp.replace('_', '-')}"
    create_table_artifact(
        key=artifact_key,
        table={
            "columns": ["Metric", "Value"],
            "data": table_data
        },
        description=f"Evaluation metrics for {retriever_type} retriever"
    )
    
    # Create a link artifact to the full results file
    file_url = f"file://{os.path.abspath(result_path)}"
    create_link_artifact(
        key=f"results-file-{retriever_type.replace('_', '-')}-{timestamp.replace('_', '-')}",
        link=file_url,
        description=f"Full evaluation results for {retriever_type} retriever"
    )
    
    # Create a markdown artifact with key metrics
    markdown_content = f"""
    # Evaluation Results: {retriever_type}
    
    ## Summary
    - **Retriever Type**: {retriever_type}
    - **Evaluated At**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - **Total Questions**: {metrics.get("total_questions", "N/A")}
    - **Success Rate**: {metrics.get("success_rate", 0) * 100:.1f}%
    
    ## Performance Metrics
    | Metric | Value |
    | ------ | ----- |
    """
    
    # Add each metric to the markdown table
    for metric, value in sorted(metric_values.items()):
        if isinstance(value, float):
            markdown_content += f"| {metric} | {value:.4f} |\n"
        else:
            markdown_content += f"| {metric} | {value} |\n"
    
    markdown_content += f"""
    ## Files
    - [Full Results JSON]({file_url})
    """
    
    create_markdown_artifact(
        key=f"summary-{retriever_type.replace('_', '-')}-{timestamp.replace('_', '-')}",
        markdown=markdown_content,
        description=f"Summary of evaluation results for {retriever_type} retriever"
    )
    
    return {
        "retriever_type": retriever_type,
        "timestamp": timestamp,
        "file_path": result_path,
        "artifacts": {
            "table": artifact_key,
            "markdown": f"summary-{retriever_type.replace('_', '-')}-{timestamp.replace('_', '-')}",
            "link": f"results-file-{retriever_type.replace('_', '-')}-{timestamp.replace('_', '-')}"
        }
    } 