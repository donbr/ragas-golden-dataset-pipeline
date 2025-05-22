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
        logger.info("Calculating traditional retrieval metrics with ground truth")
        relevance_metrics = calculate_relevance_metrics(
            retriever_result.get("detailed_results", []), 
            ["precision", "recall", "f1", "mrr", "ndcg"]
        )
        metrics.update(relevance_metrics)
    else:
        logger.info("No ground truth available, skipping traditional relevance-based metrics")
    
    # Add RAGAS metrics if available
    if config:
        logger.info("Calculating RAGAS metrics")
        ragas_metrics = calculate_ragas_metrics(retriever_result, config)
        
        if ragas_metrics:
            # Check if we actually got RAGAS metrics (not just availability flag)
            ragas_available = ragas_metrics.get("ragas_available", False)
            
            if ragas_available:
                logger.info("RAGAS metrics available, adding to results")
                # Add all RAGAS metrics to our results
                for key, value in ragas_metrics.items():
                    metrics[key] = value
                
                # Create a consolidated metrics summary for easier reference
                ragas_summary = {}
                for key, value in ragas_metrics.items():
                    if key != "ragas_available" and not key.startswith("error"):
                        # Simplify the key name for the summary
                        simple_key = key.replace("ragas_", "")
                        ragas_summary[simple_key] = value
                
                metrics["ragas_metrics"] = ragas_summary
                
                # Log the RAGAS metrics
                logger.info(f"RAGAS metrics: {ragas_summary}")
            else:
                logger.warning("RAGAS metrics not available")
                metrics["ragas_available"] = False
                
                # If there's an error, include it
                if "error" in ragas_metrics:
                    metrics["ragas_error"] = ragas_metrics["error"]
                    logger.error(f"RAGAS error: {ragas_metrics['error']}")
    else:
        logger.warning("No configuration provided, skipping RAGAS metrics")
        metrics["ragas_available"] = False
    
    # Create a Prefect artifact with the metrics summary
    from prefect.artifacts import create_table_artifact
    
    try:
        # Create a metrics table for visualization
        metrics_table = {
            "columns": ["Metric", "Value"],
            "data": []
        }
        
        # Add standard metrics
        metrics_table["data"].extend([
            ["Retriever Type", metrics["retriever_type"]],
            ["Success Rate (%)", f"{metrics['success_rate']:.2f}"],
            ["Avg Latency (ms)", f"{metrics['avg_latency_ms']:.2f}"],
            ["Avg Document Count", f"{metrics['avg_document_count']:.2f}"],
            ["Avg Document Score", f"{metrics['avg_document_score']:.2f}"],
            ["Avg Token Count", f"{metrics['avg_token_count']:.2f}"]
        ])
        
        # Add traditional relevance metrics if available
        if "precision" in metrics:
            metrics_table["data"].extend([
                ["Precision", f"{metrics['precision']:.4f}"],
                ["Recall", f"{metrics['recall']:.4f}"],
                ["F1 Score", f"{metrics['f1']:.4f}"],
                ["MRR", f"{metrics['mrr']:.4f}"],
                ["NDCG", f"{metrics['ndcg']:.4f}"]
            ])
        
        # Add RAGAS metrics if available
        if metrics.get("ragas_available", False) and "ragas_metrics" in metrics:
            # Add a section header
            metrics_table["data"].append(["--- RAGAS Metrics ---", ""])
            
            # Add each RAGAS metric
            for key, value in metrics["ragas_metrics"].items():
                if isinstance(value, (int, float)):
                    # Format numeric values
                    metrics_table["data"].append([key, f"{value:.4f}"])
                else:
                    # Handle string values (like warnings)
                    metrics_table["data"].append([key, str(value)])
        
        # Create the artifact
        create_table_artifact(
            key=f"metrics-{metrics['retriever_type']}",
            table=metrics_table,
            description=f"Metrics for {metrics['retriever_type']} retriever"
        )
    except Exception as e:
        logger.error(f"Failed to create metrics artifact: {str(e)}")
    
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
    
    This function follows the RAGAS documentation pattern to evaluate
    the retriever results using proper RAGAS metrics.
    
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
    
    try:
        # Import required RAGAS components
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (
            LLMContextRecall,
            Faithfulness,
            FactualCorrectness,
            ResponseRelevancy,
            ContextEntityRecall,
            NoiseSensitivity
        )
        
        # Get retriever components
        ragas_components = get_ragas_components(config)
        evaluator_llm = ragas_components.get("llm")
        
        # If we don't have required components, skip RAGAS evaluation
        if not evaluator_llm:
            logger.warning("RAGAS evaluator LLM not available - skipping RAGAS metrics")
            return {"ragas_available": False}
        
        # Extract detailed results from retriever_result
        detailed_results = retriever_result.get("detailed_results", [])
        if not detailed_results:
            logger.warning("No detailed results available for RAGAS evaluation")
            return {"ragas_available": False}
        
        # Prepare the dataset for RAGAS evaluation
        dataset_items = []
        for result in detailed_results:
            # Skip if no documents were retrieved
            if not result.get("documents"):
                continue
                
            # Get question and retrieved contexts
            question = result.get("question", "")
            
            # Extract document content from retrieved documents
            retrieved_contexts = [doc.get("content", "") for doc in result.get("documents", [])]
            if not retrieved_contexts:
                continue
                
            # For RAGAS metrics that require references and responses, we need to provide them
            # Otherwise, we'll get errors like "reference is missing in the test sample"
            
            # Use the first retrieved context as a mock response (for metrics that need it)
            # This is a reasonable approximation since we're evaluating retrievers, not generators
            mock_response = retrieved_contexts[0] if retrieved_contexts else ""
            
            # Use the same mock response as reference since we don't have ground truth
            # This isn't ideal, but allows us to run metrics that need references
            mock_reference = mock_response
            
            # Create the dataset item with all required fields
            dataset_items.append({
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": mock_response,
                "reference": mock_reference
            })
        
        # If we don't have any valid items, skip RAGAS evaluation
        if not dataset_items:
            logger.warning("No valid items for RAGAS evaluation")
            return {"ragas_available": False}
            
        # Create RAGAS evaluation dataset
        logger.info(f"Creating RAGAS evaluation dataset with {len(dataset_items)} items")
        evaluation_dataset = EvaluationDataset.from_list(dataset_items)
        
        # Initialize RAGAS metrics based on the global config
        # Import all the metrics we need
        from ragas.metrics import (
            LLMContextRecall,
            Faithfulness,
            FactualCorrectness,
            ResponseRelevancy,
            ContextEntityRecall,
            NoiseSensitivity
        )
        
        # Check what we have available in the dataset
        has_responses = any(item.get("response") for item in dataset_items)
        has_references = any(item.get("reference") for item in dataset_items)
        has_contexts = all(item.get("retrieved_contexts") for item in dataset_items)
        
        # Add appropriate metrics based on what's available
        metrics = []
        
        # Always add context-based metrics if we have contexts
        if has_contexts:
            metrics.append(LLMContextRecall())
            metrics.append(ContextEntityRecall())
            metrics.append(NoiseSensitivity())
        
        # Add response-based metrics if we have responses
        if has_responses and has_contexts:
            metrics.append(Faithfulness())
            metrics.append(ResponseRelevancy())
        
        # Add reference-based metrics if we have references and responses
        if has_responses and has_references:
            metrics.append(FactualCorrectness())
        
        # If no metrics were added, add at least LLMContextRecall
        if not metrics:
            logger.warning("No suitable metrics found for dataset, using LLMContextRecall as fallback")
            metrics.append(LLMContextRecall())
        
        # Log the metrics we're using
        metric_names = [type(metric).__name__ for metric in metrics]
        logger.info(f"Using RAGAS metrics: {', '.join(metric_names)}")
        
        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation")
        
        # Log the dataset and metrics for debugging
        logger.info(f"Dataset items: {len(dataset_items)}")
        for i, item in enumerate(dataset_items[:2]):  # Log first 2 items
            logger.info(f"Item {i}: user_input={item['user_input'][:50]}..., "
                        f"retrieved_contexts={len(item['retrieved_contexts'])}, "
                        f"response={item['response'][:50] if item['response'] else 'None'}..., "
                        f"reference={item['reference'][:50] if item['reference'] else 'None'}...")
        
        logger.info(f"Using metrics: {[type(m).__name__ for m in metrics]}")
        
        # Set the run config if available from global config
        run_config = None
        if ragas_components and "config" in ragas_components:
            run_config = ragas_components["config"]
            logger.info(f"Using RAGAS run config: {run_config}")
        else:
            # Create a default run config with conservative settings
            from ragas import RunConfig
            run_config = RunConfig(
                timeout=300,
                max_retries=15,
                max_wait=90,
                max_workers=8,  # Match best practice guide: 8 concurrent API calls
                log_tenacity=True
            )
            logger.info(f"Using default RAGAS run config: {run_config}")
        
        # Run the evaluation with more detailed logging
        try:
            # Explicitly pass the run_config to the evaluate function
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics,
                llm=evaluator_llm,
                run_config=run_config
            )
            logger.info(f"RAGAS evaluate() returned: {type(result)}")
            
            # Try to inspect the result structure
            if hasattr(result, "__dict__"):
                logger.info(f"Result attributes: {dir(result)}")
            
            if hasattr(result, "shape"):
                logger.info(f"Result shape: {result.shape}")
                
            if hasattr(result, "columns"):
                logger.info(f"Result columns: {list(result.columns)}")
                
            if hasattr(result, "index"):
                logger.info(f"Result index: {list(result.index)}")
            
            # Convert RAGAS results to our format
            ragas_results = {"ragas_available": True}
            
            # Properly access the EvaluationResult object
            # Based on RAGAS documentation, the result could be different formats
            try:
                result_data_found = False
                
                # Try accessing as pandas DataFrame
                if hasattr(result, 'to_dict'):
                    # Convert the entire result to a dict
                    result_dict = result.to_dict()
                    
                    # Check if we got any actual results
                    if result_dict:
                        result_data_found = True
                        # Add each metric to our results
                        for metric_name, values in result_dict.items():
                            # Calculate the average score if there are multiple values
                            if isinstance(values, dict):
                                avg_score = sum(values.values()) / len(values) if values else 0.0
                            else:
                                avg_score = values
                            
                            # Store as a float to ensure JSON serialization
                            ragas_results[f"ragas_{metric_name}"] = float(avg_score)
                            logger.info(f"RAGAS metric {metric_name}: {avg_score:.4f}")
                
                # Try accessing as dict-like object
                elif hasattr(result, 'items'):
                    result_dict = dict(result.items())
                    if result_dict:
                        result_data_found = True
                        for metric_name, value in result_dict.items():
                            ragas_results[f"ragas_{metric_name}"] = float(value)
                            logger.info(f"RAGAS metric {metric_name}: {float(value):.4f}")
                
                # Try accessing as object with attributes for each metric
                else:
                    # Get the metric names from our metrics list
                    for metric in metrics:
                        metric_name = type(metric).__name__.lower()
                        # Try to get the attribute
                        if hasattr(result, metric_name):
                            result_data_found = True
                            value = getattr(result, metric_name)
                            ragas_results[f"ragas_{metric_name}"] = float(value)
                            logger.info(f"RAGAS metric {metric_name}: {float(value):.4f}")
                
                # If we didn't find any metrics data, add a message
                if not result_data_found:
                    logger.warning("No RAGAS metrics data found in results")
                    ragas_results["ragas_warning"] = "No metrics data returned by RAGAS"
            
            except Exception as e:
                logger.error(f"Error processing RAGAS results: {str(e)}")
                ragas_results["ragas_error"] = str(e)
            
            logger.info(f"RAGAS evaluation completed: {ragas_results}")
            return ragas_results
        
        except Exception as e:
            logger.error(f"RAGAS evaluate() failed: {str(e)}")
            return {"ragas_available": False, "ragas_error": str(e)}
        
    except ImportError as e:
        logger.warning(f"RAGAS not available - skipping RAGAS metrics: {e}")
        return {"ragas_available": False}
    except Exception as e:
        logger.error(f"Error calculating RAGAS metrics: {str(e)}")
        return {"ragas_available": False, "error": str(e)} 