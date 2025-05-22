"""
Global evaluation configuration utilities.

This module provides tasks for setting up and managing global
evaluation configuration for RAGAS.
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta
from prefect import task, get_run_logger
from prefect.tasks import task_input_hash

# RAGAS imports with proper guards
try:
    from ragas import evaluate, RunConfig
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.metrics import (
        LLMContextRecall,
        Faithfulness,
        FactualCorrectness,
        ResponseRelevancy,
        ContextEntityRecall,
        NoiseSensitivity,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


@task(
    name="setup-global-evaluation-config",
    description="Configures global settings for RAGAS evaluations",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    tags=["setup", "ragas"],
    result_serializer="json",  # Use JSON serializer to avoid pickle issues
    persist_result=True  # Ensure results are persisted
)
def setup_global_evaluation_config(
    llm_model: str = "gpt-4.1-mini",
    timeout: int = 300,
    max_retries: int = 15,
    max_wait: int = 90,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Sets up global evaluation configuration for RAGAS.
    
    Args:
        llm_model: Model to use for evaluation
        timeout: Maximum timeout for operations (in seconds)
        max_retries: Maximum number of retries for API calls
        max_wait: Maximum wait time between retries (in seconds)
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dictionary with evaluation configuration metadata
        (not the actual objects, as they are not JSON-serializable)
    """
    logger = get_run_logger()
    
    if not RAGAS_AVAILABLE:
        logger.warning("RAGAS not available. Install ragas package to enable evaluation features.")
        return {
            "status": "unavailable",
            "message": "RAGAS not installed. Install with: pip install ragas"
        }
    
    logger.info(f"Setting up global evaluation configuration with model: {llm_model}")
    
    # We still create the objects but don't return them directly
    # Instead, we'll recreate them when needed using the settings
    
    # Configure standard metrics for evaluation
    metric_names = [
        "LLMContextRecall",
        "Faithfulness",
        "FactualCorrectness",
        "ResponseRelevancy",
        "ContextEntityRecall",
        "NoiseSensitivity"
    ]
    
    logger.info(f"Configured {len(metric_names)} RAGAS metrics with {max_workers} workers")
    
    # Return only serializable metadata
    return {
        "status": "available",
        "settings": {
            "llm_model": llm_model,
            "timeout": timeout,
            "max_retries": max_retries,
            "max_wait": max_wait,
            "max_workers": max_workers,
            "metric_names": metric_names
        }
    }


def get_ragas_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create actual RAGAS components from serializable configuration.
    
    This function recreates the actual RAGAS objects from the serializable
    configuration returned by setup_global_evaluation_config.
    
    Args:
        config: Configuration dictionary from setup_global_evaluation_config
        
    Returns:
        Dictionary with actual RAGAS components
    """
    if not RAGAS_AVAILABLE or config.get("status") != "available":
        return config
    
    settings = config.get("settings", {})
    llm_model = settings.get("llm_model", "gpt-4.1-mini")
    timeout = settings.get("timeout", 300)
    max_retries = settings.get("max_retries", 15)
    max_wait = settings.get("max_wait", 90)
    max_workers = settings.get("max_workers", 8)
    
    # Configure evaluation LLM
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    
    # Configure runtime settings to handle rate limits
    evaluator_config = RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_wait=max_wait,
        max_workers=max_workers,
        log_tenacity=True
    )
    
    # Create metrics based on names
    metric_classes = {
        "LLMContextRecall": LLMContextRecall,
        "Faithfulness": Faithfulness,
        "FactualCorrectness": FactualCorrectness,
        "ResponseRelevancy": ResponseRelevancy,
        "ContextEntityRecall": ContextEntityRecall,
        "NoiseSensitivity": NoiseSensitivity
    }
    
    evaluator_metrics = [
        metric_classes[name]() for name in settings.get("metric_names", [])
        if name in metric_classes
    ]
    
    return {
        "status": "available",
        "llm": evaluator_llm,
        "config": evaluator_config,
        "metrics": evaluator_metrics,
        "settings": settings
    }


def create_execution_metadata() -> Dict[str, Any]:
    """
    Create metadata about the execution environment.
    
    Returns:
        Dictionary with execution metadata
    """
    import sys
    import prefect
    from datetime import datetime
    from uuid import uuid4
    
    return {
        "timestamp": datetime.now().isoformat(),
        "execution_id": str(uuid4()),
        "python_version": sys.version,
        "prefect_version": prefect.__version__,
    } 