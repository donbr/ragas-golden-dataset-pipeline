"""
Evaluation results reporting utilities.

This module provides tasks for generating reports and
visualizations from evaluation results.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from prefect import task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact, create_link_artifact


@task(
    name="generate-evaluation-report",
    description="Creates a comprehensive evaluation report",
    retries=1,
    retry_delay_seconds=5,
    tags=["results", "reporting"],
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def generate_evaluation_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "evaluation_results/",
    include_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Creates a comprehensive evaluation report with comparisons.
    
    Args:
        results: Dictionary mapping retriever types to their metrics
        output_dir: Directory to store report
        include_visualizations: Whether to include visualizations
        
    Returns:
        Dictionary with report information
    """
    logger = get_run_logger()
    logger.info(f"Generating evaluation report for {len(results)} retrievers")
    
    # Create timestamp for report naming
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Use hyphens instead of underscores
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics dataframe for comparison
    metrics_rows = []
    
    # Log detailed retriever information
    logger.info("Detailed retriever configuration and performance:")
    for retriever_type, metrics in results.items():
        logger.info(f"Retriever: {retriever_type}")
        logger.info(f"  Success rate: {metrics.get('success_rate', 0):.2f}")
        logger.info(f"  Avg latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1: {metrics.get('f1', 0):.4f}")
        
        if metrics.get("retriever_info"):
            for key, value in metrics.get("retriever_info", {}).items():
                logger.info(f"  {key}: {value}")
        
        # Extract the metrics we want to compare
        row = {
            "retriever_type": retriever_type,
            "success_rate": metrics.get("success_rate", 0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
            "mrr": metrics.get("mrr", 0),
            "ndcg": metrics.get("ndcg", 0),
        }
        
        # Add RAGAS metrics if available
        if "ragas_available" in metrics and metrics["ragas_available"]:
            for key, value in metrics.items():
                if key.startswith("ragas_") and key != "ragas_available":
                    row[key] = value
        
        metrics_rows.append(row)
    
    # Create dataframe for comparison
    comparison_df = pd.DataFrame(metrics_rows)
    
    # Save comparison to CSV
    csv_path = os.path.join(output_dir, f"comparison-{timestamp}.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    # Create visualizations if requested
    visualization_paths = []
    
    if include_visualizations and len(results) > 1:
        visualization_paths = generate_visualizations(comparison_df, output_dir, timestamp)
    
    # Create comparison table artifact with proper format
    # Convert DataFrame to JSON serializable format for table artifact
    table_data = []
    for _, row in comparison_df.iterrows():
        table_row = []
        for col in comparison_df.columns:
            value = row[col]
            if isinstance(value, float):
                # Format floats to 4 decimal places
                table_row.append(f"{value:.4f}")
            else:
                table_row.append(str(value))
        table_data.append(table_row)
    
    create_table_artifact(
        key=f"comparison-{timestamp}",  # Now uses hyphens instead of underscores
        table={
            "columns": comparison_df.columns.tolist(),
            "data": table_data
        },
        description=f"Comparison of {len(results)} retrievers"
    )
    
    # Create comprehensive markdown report
    markdown_content = generate_markdown_report(comparison_df, visualization_paths, output_dir, timestamp)
    
    create_markdown_artifact(
        key=f"report-{timestamp}",  # Now uses hyphens instead of underscores
        markdown=markdown_content,
        description=f"Comprehensive evaluation report for {len(results)} retrievers"
    )
    
    # Create link to CSV comparison
    create_link_artifact(
        key=f"comparison-csv-{timestamp}",  # Now uses hyphens instead of underscores
        link=f"file://{os.path.abspath(csv_path)}",
        description=f"CSV comparison of {len(results)} retrievers"
    )
    
    logger.info(f"Generated evaluation report at {csv_path}")
    
    # Return JSON-serializable information (no pandas dataframes)
    return {
        "timestamp": timestamp,
        "comparison_csv": csv_path,
        "visualization_paths": visualization_paths,
        "retriever_count": len(results),
        "metrics_compared": comparison_df.columns.tolist(),
        "artifacts": {
            "table": f"comparison-{timestamp}",
            "markdown": f"report-{timestamp}",
            "link": f"comparison-csv-{timestamp}"
        }
    }


def generate_visualizations(
    comparison_df: pd.DataFrame,
    output_dir: str,
    timestamp: str
) -> List[str]:
    """
    Generates visualizations for comparing retrievers.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        output_dir: Directory to store visualizations
        timestamp: Timestamp for file naming
        
    Returns:
        List of visualization file paths
    """
    visualization_paths = []
    
    # Create directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Bar chart for precision, recall, F1
    try:
        plt.figure(figsize=(12, 8))
        
        # Set the positions for the bars
        x = range(len(comparison_df))
        width = 0.25
        
        # Create bars
        plt.bar([p - width for p in x], comparison_df["precision"], width, label="Precision")
        plt.bar(x, comparison_df["recall"], width, label="Recall")
        plt.bar([p + width for p in x], comparison_df["f1"], width, label="F1")
        
        # Add labels and title
        plt.xlabel("Retriever Type")
        plt.ylabel("Score")
        plt.title("Precision, Recall, and F1 by Retriever Type")
        plt.xticks(x, comparison_df["retriever_type"])
        plt.legend()
        
        # Add values on top of the bars
        for i, v in enumerate(comparison_df["precision"]):
            plt.text(i - width, v + 0.01, f"{v:.2f}", ha="center")
        for i, v in enumerate(comparison_df["recall"]):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        for i, v in enumerate(comparison_df["f1"]):
            plt.text(i + width, v + 0.01, f"{v:.2f}", ha="center")
        
        # Save the plot
        prf_path = os.path.join(viz_dir, f"precision_recall_f1_{timestamp}.png")
        plt.savefig(prf_path)
        plt.close()
        
        visualization_paths.append(prf_path)
    except Exception as e:
        print(f"Error generating precision/recall visualization: {str(e)}")
    
    # Bar chart for latency
    try:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df["retriever_type"], comparison_df["avg_latency_ms"])
        
        # Add title and labels
        plt.title("Average Latency by Retriever Type")
        plt.xlabel("Retriever Type")
        plt.ylabel("Latency (ms)")
        
        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom"
            )
        
        # Save the plot
        latency_path = os.path.join(viz_dir, f"latency_{timestamp}.png")
        plt.savefig(latency_path)
        plt.close()
        
        visualization_paths.append(latency_path)
    except Exception as e:
        print(f"Error generating latency visualization: {str(e)}")
    
    # Radar chart if we have MRR and NDCG
    try:
        if "mrr" in comparison_df.columns and "ndcg" in comparison_df.columns:
            # Get the metrics for the radar chart
            metrics = ["precision", "recall", "f1", "mrr", "ndcg"]
            
            # Number of variables
            num_vars = len(metrics)
            
            # Create a figure and a polar axis for the radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Calculate angles for each metric
            angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
            angles += angles[:1]  # Close the polygon
            
            # Draw one line per retriever and fill the area
            for i, retriever in enumerate(comparison_df["retriever_type"]):
                # Get values for this retriever
                values = comparison_df.loc[i, metrics].tolist()
                values += values[:1]  # Close the polygon
                
                # Plot the retriever line
                ax.plot(angles, values, linewidth=2, linestyle="solid", label=retriever)
                ax.fill(angles, values, alpha=0.1)
            
            # Set the labels for each metric
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            
            # Add legend and title
            plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
            plt.title("Retriever Performance Comparison", size=15)
            
            # Save the radar chart
            radar_path = os.path.join(viz_dir, f"radar_{timestamp}.png")
            plt.savefig(radar_path)
            plt.close()
            
            visualization_paths.append(radar_path)
    except Exception as e:
        print(f"Error generating radar visualization: {str(e)}")
    
    return visualization_paths


def generate_markdown_report(
    comparison_df: pd.DataFrame,
    visualization_paths: List[str],
    output_dir: str,
    timestamp: str
) -> str:
    """
    Generates a comprehensive markdown report.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        visualization_paths: List of paths to visualizations
        output_dir: Directory where results are stored
        timestamp: Timestamp for report ID
        
    Returns:
        Markdown content for the report
    """
    # Start with a header
    markdown = f"""
    # Retriever Evaluation Report
    
    **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    **Retrievers Compared**: {len(comparison_df)}
    
    ## Performance Comparison
    
    """
    
    # Add the comparison table using manual markdown generation instead of pandas to_markdown
    # This avoids the dependency on tabulate
    try:
        # First try pandas to_markdown if tabulate is available
        markdown += comparison_df.to_markdown(index=False)
    except ImportError:
        # Fallback to manual markdown table generation if tabulate is not available
        logger = get_run_logger()
        logger.info("Tabulate package not found, using manual markdown table generation")
        
        # Get column headers and add markdown table header
        columns = comparison_df.columns.tolist()
        header_row = "| " + " | ".join(columns) + " |"
        separator_row = "| " + " | ".join(["---" for _ in columns]) + " |"
        
        markdown += header_row + "\n" + separator_row + "\n"
        
        # Add each row of data
        for _, row in comparison_df.iterrows():
            row_values = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    # Format floats to 4 decimal places
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append(str(value))
            
            markdown += "| " + " | ".join(row_values) + " |\n"
    
    # Add visualizations if available
    if visualization_paths:
        markdown += "\n\n## Visualizations\n\n"
        
        for viz_path in visualization_paths:
            # Get the file name and create a local link
            viz_name = os.path.basename(viz_path)
            viz_url = f"file://{os.path.abspath(viz_path)}"
            markdown += f"### {viz_name.split('_')[0].title()} Comparison\n\n"
            markdown += f"![{viz_name}]({viz_url})\n\n"
    
    # Add detailed metrics for each retriever
    markdown += "\n\n## Detailed Metrics\n\n"
    
    for i, row in comparison_df.iterrows():
        retriever = row["retriever_type"]
        markdown += f"### {retriever}\n\n"
        
        # Add a table of all metrics for this retriever
        markdown += "| Metric | Value |\n| ------ | ----- |\n"
        
        for metric in comparison_df.columns:
            if metric != "retriever_type":
                value = row[metric]
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                markdown += f"| {metric} | {formatted_value} |\n"
        
        markdown += "\n"
    
    # Add conclusions section
    markdown += "\n\n## Conclusions\n\n"
    
    # Find the best performer for each metric
    best_performers = {}
    for metric in comparison_df.columns:
        if metric != "retriever_type":
            # For latency, lower is better
            if metric == "avg_latency_ms":
                best_idx = comparison_df[metric].idxmin()
            # For all other metrics, higher is better
            else:
                best_idx = comparison_df[metric].idxmax()
            
            best_value = comparison_df.loc[best_idx, metric]
            best_retriever = comparison_df.loc[best_idx, "retriever_type"]
            
            best_performers[metric] = (best_retriever, best_value)
    
    # Add the best performers to the conclusions
    markdown += "### Best Performers\n\n"
    markdown += "| Metric | Best Retriever | Value |\n| ------ | ------------- | ----- |\n"
    
    for metric, (retriever, value) in best_performers.items():
        if metric != "retriever_type":
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            markdown += f"| {metric} | {retriever} | {formatted_value} |\n"
    
    return markdown 