"""
Global evaluation configuration utilities.

This module provides tasks for setting up and managing global
evaluation configuration for RAGAS.
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta
from prefect import task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.artifacts import create_markdown_artifact

# Direct RAGAS imports without guards
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
    max_workers: int = 8,  # Match best practice guide: 8 concurrent API calls
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
    
    logger.info(f"Setting up global evaluation configuration with model: {llm_model}")
    
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
    
    # Create an artifact showing the RAGAS metrics configuration
    create_markdown_artifact(
        key="ragas-metrics-config",
        markdown=f"""# RAGAS Metrics Configuration

## Model
- LLM: `{llm_model}`

## Runtime Configuration
- Timeout: {timeout}s
- Max Retries: {max_retries}
- Max Wait: {max_wait}s
- Max Workers: {max_workers}
- Log Retries: True

## Configured Metrics
{chr(10).join(['- ' + metric for metric in metric_names])}
""",
        description="RAGAS metrics configuration"
    )
    
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
    settings = config.get("settings", {})
    llm_model = settings.get("llm_model", "gpt-4.1-mini")
    timeout = settings.get("timeout", 300)
    max_retries = settings.get("max_retries", 15)
    max_wait = settings.get("max_wait", 90)
    max_workers = settings.get("max_workers", 8)  # Use 8 workers by default to match best practice guide
    
    # Configure evaluation LLM
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    
    # Configure runtime settings to handle rate limits
    evaluator_config = RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_wait=max_wait,
        max_workers=max_workers,  # Use the configured max_workers value
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