"""
Retrieval metrics calculation.

This module provides tasks for calculating various retrieval
evaluation metrics based on retriever results.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import timedelta

from prefect import task, get_run_logger
from prefect.tasks import task_input_hash

# Import the get_ragas_components function
from libs.evaluation_core.config import get_ragas_components


@task(
    name="calculate-metrics",
    description="Calculates performance metrics for retriever results",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=1,
    retry_delay_seconds=5,
    tags=["evaluation", "metrics"],
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def calculate_metrics(
    retriever_result: Dict[str, Any], 
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates performance metrics for retriever results.
    
    Args:
        retriever_result: Results from evaluate_retriever
        config: Optional global evaluation configuration
        
    Returns:
        Dictionary with calculated metrics
    """
    logger = get_run_logger()
    logger.info(f"Calculating metrics for {retriever_result['retriever_type']} retriever")
    
    # Track all metrics
    metrics = {
        "retriever_type": retriever_result["retriever_type"],
        "success_rate": retriever_result.get("success_rate", 0),
        "avg_latency_ms": retriever_result.get("avg_latency_ms", 0),
        "avg_document_count": retriever_result.get("avg_document_count", 0),
        "avg_document_score": retriever_result.get("avg_document_score", 0),
        "avg_token_count": retriever_result.get("avg_token_count", 0),
    }
    
    # Include retriever-specific information if available
    if "retriever_info" in retriever_result:
        metrics["retriever_info"] = retriever_result["retriever_info"]
        logger.info(f"Including detailed information for {retriever_result['retriever_type']} retriever")
        
        # Log the distinctive characteristics
        if "description" in retriever_result["retriever_info"]:
            logger.info(f"  Description: {retriever_result['retriever_info']['description']}")
        if "strength" in retriever_result["retriever_info"]:
            logger.info(f"  Strength: {retriever_result['retriever_info']['strength']}")
        if "weakness" in retriever_result["retriever_info"]:
            logger.info(f"  Weakness: {retriever_result['retriever_info']['weakness']}")
    
    # Calculate traditional retrieval metrics if ground truth is available
    if retriever_result.get("ground_truth"):
        logger.info("Calculating the following metrics: precision, recall, f1, mrr, ndcg, latency, success_rate, token_efficiency")
        
        # Calculate precision, recall, F1 from results
        # This would use the ground truth and retrieved documents
        
        # For this example, we'll mock up these calculations
        # TODO: Implement actual metrics calculations
        metrics.update({
            "precision": 0.0,  # Would be calculated using ground truth
            "recall": 0.0,     # Would be calculated using ground truth
            "f1": 0.0,         # Would be calculated using ground truth
            "mrr": 0.0,        # Would be calculated using ground truth
            "ndcg": 0.0        # Would be calculated using ground truth
        })
    else:
        logger.warning("No ground truth available, skipping relevance-based metrics")
    
    # Add RAGAS metrics if available
    if config and config.get("status") == "available":
        ragas_metrics = calculate_ragas_metrics(retriever_result, config)
        if ragas_metrics:
            metrics.update(ragas_metrics)
            metrics["ragas_available"] = True
    
    logger.info(f"Completed metrics calculation for {retriever_result['retriever_type']} retriever")
    
    return metrics


def calculate_relevance_metrics(results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
    """
    Calculates relevance-based metrics when ground truth is available.
    
    Args:
        results: List of results with ground truth
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with calculated metrics
    """
    metric_values = {}
    
    # Track metrics across all queries
    precision_values = []
    recall_values = []
    f1_values = []
    mrr_values = []
    ndcg_values = []
    
    for result in results:
        # Get retrieved documents
        retrieved_docs = result.get("documents", [])
        retrieved_ids = [doc.get("id") for doc in retrieved_docs if "id" in doc]
        
        # Get ground truth (relevant document IDs)
        relevant_ids = result.get("relevant_docs", result.get("ground_truth", []))
        
        if not relevant_ids or not retrieved_ids:
            continue
        
        # Calculate precision, recall, F1
        relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
        
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        
        # Calculate MRR
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        mrr_values.append(mrr)
        
        # Calculate NDCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids):
            relevance = 1 if doc_id in relevant_ids else 0
            dcg += relevance / np.log2(i + 2)  # i+2 because log_2(1) = 0
        
        # Calculate ideal DCG (IDCG)
        idcg = 0
        for i in range(min(len(relevant_ids), len(retrieved_ids))):
            idcg += 1.0 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_values.append(ndcg)
    
    # Calculate average metrics
    if "precision" in metrics and precision_values:
        metric_values["precision"] = float(np.mean(precision_values))
    
    if "recall" in metrics and recall_values:
        metric_values["recall"] = float(np.mean(recall_values))
    
    if "f1" in metrics and f1_values:
        metric_values["f1"] = float(np.mean(f1_values))
    
    if "mrr" in metrics and mrr_values:
        metric_values["mrr"] = float(np.mean(mrr_values))
    
    if "ndcg" in metrics and ndcg_values:
        metric_values["ndcg"] = float(np.mean(ndcg_values))
    
    return metric_values


def calculate_ragas_metrics(retriever_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate RAGAS metrics if RAGAS is available.
    
    This is a helper function that will be expanded when we implement 
    RAGAS-specific evaluation logic.
    
    Args:
        retriever_result: Results from evaluate_retriever
        config: Global evaluation configuration
        
    Returns:
        Dictionary with RAGAS metric results or empty dict if RAGAS unavailable
    """
    logger = get_run_logger()
    
    # Check if we have a valid RAGAS configuration
    if not config or config.get("status") != "available":
        logger.warning("RAGAS configuration not available - skipping RAGAS metrics")
        return {"ragas_available": False}
    
    # Check if we have metrics
    if "metrics" not in config:
        logger.warning("No RAGAS metrics in configuration - skipping RAGAS metrics")
        return {"ragas_available": False}
    
    try:
        # Check if RAGAS is available
        from ragas import evaluate
        
        # Log the metrics we're using
        metrics = config.get("metrics", [])
        metric_names = [type(metric).__name__ for metric in metrics]
        logger.info(f"Using RAGAS metrics: {', '.join(metric_names)}")
        
        # Just a placeholder result for now
        # In the full implementation, we would call the RAGAS evaluator with
        # the retrieved documents, questions, and potential answers
        
        # For now, create dummy results for each metric
        results = {"ragas_available": True}
        for metric_name in metric_names:
            results[f"ragas_{metric_name.lower()}"] = 0.0
            
        return results
        
    except ImportError:
        logger.warning("RAGAS not available - skipping RAGAS metrics")
        return {"ragas_available": False}
    except Exception as e:
        logger.error(f"Error calculating RAGAS metrics: {str(e)}")
        return {"ragas_available": False, "error": str(e)} 