"""
Test dataset loading utilities.

This module provides tasks for loading test datasets from
various sources.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import timedelta
from pathlib import Path

from prefect import task, get_run_logger
from prefect.tasks import task_input_hash


@task(
    name="load-test-dataset",
    description="Loads test dataset from file or HuggingFace",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=2,
    retry_delay_seconds=10,
    tags=["data", "loading"]
)
def load_test_dataset(
    test_dataset_path: str = "",
    hf_dataset_repo: str = "",
    dataset_size_limit: int = 0
) -> Dict[str, Any]:
    """
    Loads test dataset from file or HuggingFace.
    
    Args:
        test_dataset_path: Path to local test dataset file (JSON)
        hf_dataset_repo: HuggingFace dataset repository path (alternative to local)
        dataset_size_limit: Maximum number of examples to load (0 for all)
        
    Returns:
        Dictionary with loaded dataset and metadata
    """
    logger = get_run_logger()
    
    if not test_dataset_path and not hf_dataset_repo:
        raise ValueError("Either test_dataset_path or hf_dataset_repo must be provided")
    
    dataset = None
    source = "local"
    
    # Try loading from local file first
    if test_dataset_path:
        logger.info(f"Loading test dataset from local file: {test_dataset_path}")
        try:
            file_path = Path(test_dataset_path)
            
            # Log the resolved path
            logger.info(f"Using dataset at {file_path.resolve()}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(file_path, 'r') as f:
                dataset = json.load(f)
                
            logger.info(f"Successfully loaded dataset from {test_dataset_path}")
        except Exception as e:
            logger.error(f"Error loading local dataset: {str(e)}")
            if hf_dataset_repo:
                logger.info(f"Falling back to HuggingFace dataset")
            else:
                raise
    
    # Try loading from HuggingFace if local loading failed or wasn't specified
    if not dataset and hf_dataset_repo:
        logger.info(f"Loading test dataset from HuggingFace: {hf_dataset_repo}")
        source = "huggingface"
        
        try:
            # Try to import the datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                logger.error("HuggingFace datasets library not installed. Install with: pip install datasets")
                raise ImportError("HuggingFace datasets library required for loading from HuggingFace")
            
            # Load the dataset
            hf_dataset = load_dataset(hf_dataset_repo)
            
            # Convert to a compatible format
            if "train" in hf_dataset:
                dataset_split = hf_dataset["train"]
            else:
                dataset_split = next(iter(hf_dataset.values()))
            
            # Convert to a dictionary for consistency
            dataset = {"examples": []}
            
            for item in dataset_split:
                # Check for RAGAS format field names and map them to our expected format
                if "user_input" in item:
                    # RAGAS field mapping:
                    # user_input -> question
                    # reference_contexts -> context
                    # reference -> answer
                    example = {
                        "question": item.get("user_input", ""),
                        "context": item.get("reference_contexts", []),
                        "answer": item.get("reference", ""),
                    }
                else:
                    # Standard format or attempt other mappings
                    example = {
                        "question": item.get("question", ""),
                        "context": item.get("context", item.get("documents", [])),
                        "answer": item.get("answer", item.get("answers", {}).get("text", [""])[0] if isinstance(item.get("answers", {}), dict) else ""),
                    }
                dataset["examples"].append(example)
            
            logger.info(f"Successfully loaded dataset from HuggingFace: {hf_dataset_repo}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {str(e)}")
            raise
    
    # Apply size limit if specified
    if dataset and dataset_size_limit > 0:
        if "examples" in dataset:
            dataset["examples"] = dataset["examples"][:dataset_size_limit]
            logger.info(f"Limited dataset to {len(dataset['examples'])} examples")
        else:
            logger.warning("Dataset doesn't have an 'examples' key, could not apply size limit")
    
    # Get dataset size
    example_count = len(dataset.get("examples", [])) if dataset else 0
    
    # Create metadata
    metadata = {
        "source": source,
        "source_path": test_dataset_path if source == "local" else hf_dataset_repo,
        "example_count": example_count,
        "has_context": all("context" in ex for ex in dataset.get("examples", [])) if dataset else False,
        "has_answers": all("answer" in ex for ex in dataset.get("examples", [])) if dataset else False,
    }
    
    logger.info(f"Dataset loaded with {example_count} examples from {source}")
    
    return {
        "dataset": dataset,
        "metadata": metadata
    } 