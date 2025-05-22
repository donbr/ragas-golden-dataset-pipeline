import os
# Set Prefect to use ephemeral mode before importing Prefect
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "True"
# Configure Prefect to persist results by default for better caching
os.environ["PREFECT_RESULTS_PERSIST_BY_DEFAULT"] = "true"

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from datetime import datetime, timedelta
import argparse
import subprocess
import json
import pickle
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
# Tasks
# ------------------------------------------------------------------------------

@task(
    name="validate-environment",
    description="Validates all required environment variables are set",
    retries=3,
    retry_delay_seconds=5,
    tags=["setup", "validation"],
    task_run_name="Validate environment variables"
)
def validate_environment() -> Dict[str, List[str]]:
    """
    Validates that all required environment variables are set.
    
    Returns:
        Dict mapping environment variable categories to lists of set variables
    
    Raises:
        EnvironmentError: If any required variables are missing
    """
    logger = get_run_logger()
    logger.info("🔍 Validating environment variables...")
    execution_start = time.time()
    
    # Define required environment variables by category
    required_vars = {
        "OpenAI": ["OPENAI_API_KEY"],
        "HuggingFace": ["HF_TOKEN"] if os.environ.get("HF_TESTSET_REPO") else []
    }
    
    # Optional but recommended variables
    optional_vars = {
        "LangSmith": ["LANGSMITH_PROJECT", "LANGSMITH_TRACING"],
        "Pipeline Config": ["RAW_DIR", "TESTSET_SIZE", "KG_OUTPUT_PATH"]
    }
    
    # Check for missing required variables
    missing_vars = {}
    for category, vars_list in required_vars.items():
        category_missing = [var for var in vars_list if var not in os.environ or not os.environ[var]]
        if category_missing:
            missing_vars[category] = category_missing
    
    # If any required variables are missing, log and raise error
    if missing_vars:
        lines = ["❌ Missing required environment variables:"]
        for category, vars_list in missing_vars.items():
            lines.append(f"• {category}: {', '.join(vars_list)}")
        lines.append("Please set these in your environment or .env file.")
        msg = "\n".join(lines)
        logger.error(msg)
        
        # Create artifact for the error
        create_markdown_artifact(
            key="environment-validation-error",
            markdown=f"# ❌ Environment Validation Failed\n\n{msg}",
            description="Missing environment variables"
        )
        
        raise EnvironmentError(msg)
    
    # Check for missing optional variables and log warnings
    missing_optional = {}
    for category, vars_list in optional_vars.items():
        category_missing = [var for var in vars_list if var not in os.environ or not os.environ[var]]
        if category_missing:
            missing_optional[category] = category_missing
            logger.warning(f"Optional {category} variables not set: {', '.join(category_missing)}")
    
    # Build table of resolved values for artifact
    table = [["Category", "Variable", "Value", "Status"]]
    
    # Add required vars to table
    for category, vars_list in required_vars.items():
        for var in vars_list:
            raw = os.environ.get(var, "")
            val = mask_value(raw) if var in SENSITIVE_VARS else raw
            status = "✅" if raw else "❌ Missing"
            table.append([category, var, val, status])
            logger.info(f"{var} = {val}")
    
    # Add optional vars to table
    for category, vars_list in optional_vars.items():
        for var in vars_list:
            raw = os.environ.get(var, "")
            val = mask_value(raw) if var in SENSITIVE_VARS else raw or "Not set"
            status = "✅" if raw else "⚠️ Optional"
            table.append([category, var, val, status])
            logger.info(f"{var} = {val}")

    create_table_artifact(
        key="env-vars",
        table={"columns": table[0], "data": table[1:]},
        description="Resolved environment variables (sensitive masked)"
    )
    
    # Emit event with validation completion
    execution_time_ms = int((time.time() - execution_start) * 1000)
    emit_event(
        event="environment-validated",
        resource={"prefect.resource.id": "ragas-pipeline.environment"},
        payload={
            "status": "success",
            "execution_time_ms": execution_time_ms,
            "categories": list(required_vars.keys()) + list(optional_vars.keys()),
            "required_vars_count": sum(len(v) for v in required_vars.values()),
            "optional_vars_count": sum(len(v) for v in optional_vars.values())
        }
    )
    
    logger.info("✅ Environment validated successfully.")
    
    # Return a dictionary of set variables by category for potential use in tasks
    return {cat: [k for k in keys if os.environ.get(k)] 
            for cat, keys in {**required_vars, **optional_vars}.items()}

@task(
    name="download-pdfs",
    description="Downloads sample PDFs for RAGAS testset generation",
    retries=3,
    retry_delay_seconds=30,
    timeout_seconds=300,
    tags=["data", "download"],
    task_run_name="Download sample PDFs from arXiv"
)
def download_pdfs(data_path: str) -> str:
    """
    Download sample PDFs for RAGAS testset generation.
    
    Args:
        data_path: Directory path to save downloaded PDFs
        
    Returns:
        Path to the directory containing PDFs
        
    Raises:
        ValueError: If path validation fails
        RuntimeError: If all download attempts fail
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        # Validate and create directory if needed
        pdf_dir = validate_path(data_path, create_if_missing=True)
        logger.info(f"Downloading PDFs to directory: {pdf_dir}")
        
        # PDF URLs and filenames - latest AI agent papers from arXiv
        pdf_urls = [
            ("https://arxiv.org/pdf/2505.10468.pdf", "ai_agents_vs_agentic_ai_2505.10468.pdf"),
            ("https://arxiv.org/pdf/2505.06913.pdf", "redteamllm_agentic_ai_framework_2505.06913.pdf"),
            ("https://arxiv.org/pdf/2505.06817.pdf", "control_plane_scalable_design_pattern_2505.06817.pdf")
        ]
        
        successful_downloads = 0
        failed_downloads = 0
        skipped_downloads = 0
        failed_files = []
        
        for url, filename in pdf_urls:
            output_path = pdf_dir / filename
            
            # Skip download if file already exists
            if output_path.exists():
                logger.info(f"File {filename} already exists, skipping download")
                skipped_downloads += 1
                continue
                
            logger.info(f"Downloading {url} to {output_path}")
            
            # Try using curl first (with Windows compatibility)
            try:
                subprocess.run([
                    "curl", "-L", "--ssl-no-revoke", "--connect-timeout", "30",
                    "--max-time", "60", url, "-o", str(output_path)
                ], check=True)
                logger.info(f"✅ Successfully downloaded {filename} using curl")
                successful_downloads += 1
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # Fallback to requests if curl fails or isn't available
                logger.warning(f"Curl failed with error: {str(e)}. Using requests library as fallback")
                try:
                    # Disable insecure request warnings when verify=False
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    
                    # Add timeout to prevent hanging
                    response = requests.get(url, verify=False, timeout=60)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Successfully downloaded {filename} using requests")
                    successful_downloads += 1
                except Exception as inner_e:
                    logger.error(f"Failed to download {filename}: {str(inner_e)}")
                    failed_downloads += 1
                    failed_files.append((filename, url, str(inner_e)))
                    if output_path.exists() and output_path.stat().st_size == 0:
                        # Clean up empty files
                        output_path.unlink()
                        logger.info(f"Removed empty file: {output_path}")
        
        # Create artifact with download summary and metadata
        create_table_artifact(
            key="pdf-download-summary",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Successful Downloads", successful_downloads],
                    ["Failed Downloads", failed_downloads],
                    ["Skipped (Already Existed)", skipped_downloads],
                    ["Total PDFs", successful_downloads + skipped_downloads],
                    ["Output Directory", str(pdf_dir)],
                    ["Execution Time (s)", f"{time.time() - execution_start:.2f}"]
                ]
            },
            description="Summary of PDF download operations"
        )
        
        # If there were failures, create a warning artifact
        if failed_files:
            failures_md = "\n".join([f"- **{name}** ({url}): {error}" for name, url, error in failed_files])
            create_markdown_artifact(
                key="pdf-download-warnings",
                markdown=f"# ⚠️ Some PDF Downloads Failed\n\n{failures_md}",
                description="Details of failed PDF downloads"
            )
        
        # Emit event about download completion
        execution_time_ms = int((time.time() - execution_start) * 1000)
        emit_event(
            event="pdfs-downloaded",
            resource={"prefect.resource.id": f"ragas-pipeline.data.{data_path}"},
            payload={
                "successful": successful_downloads,
                "failed": failed_downloads,
                "skipped": skipped_downloads,
                "data_path": str(pdf_dir),
                "execution_time_ms": execution_time_ms
            }
        )
        
        # Check if we have at least one PDF
        if successful_downloads + skipped_downloads == 0:
            logger.error("No PDFs were downloaded or found in the directory.")
            create_markdown_artifact(
                key="pdf-download-error",
                markdown=f"# ❌ No PDFs Available\n\nNo PDFs were downloaded or found in directory: `{pdf_dir}`",
                description="PDF download critical error"
            )
            raise RuntimeError(f"No PDFs available in {pdf_dir}")
            
        return str(pdf_dir)
        
    except ValueError as e:
        # Path validation error
        logger.error(f"Path validation error: {str(e)}")
        create_markdown_artifact(
            key="pdf-download-error",
            markdown=f"# ❌ PDF Download Failed\n\n**Path Error:**\n```\n{str(e)}\n```",
            description="PDF download path validation error"
        )
        raise
        
    except Exception as e:
        # Generic error handler
        logger.error(f"Error in PDF download task: {str(e)}")
        create_markdown_artifact(
            key="pdf-download-error",
            markdown=f"# ❌ PDF Download Failed\n\n**Error:**\n```\n{str(e)}\n```",
            description="PDF download error"
        )
        raise

@task(
    name="load-documents",
    description="Loads PDF documents from a directory using LangChain's PyPDFDirectoryLoader",
    retries=2,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    timeout_seconds=300,
    tags=["data", "loading"],
    task_run_name="Load documents from {path}"
)
def load_documents(path: str) -> List[Document]:
    """
    Load PDF documents from a directory using LangChain's PyPDFDirectoryLoader.
    
    Args:
        path: Path to directory containing PDFs
        
    Returns:
        List of Document objects
        
    Raises:
        ValueError: If path validation fails
        RuntimeError: If document loading fails
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        # Validate the path
        pdf_dir = validate_path(path, must_exist=True)
        logger.info(f"Loading documents from {pdf_dir}...")
        
        # Check if there are any PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {path}")
            create_markdown_artifact(
                key="document-loading-warning",
                markdown=f"# ⚠️ No PDF Files Found\n\nNo PDF files were found in directory: `{path}`",
                description="Document loading warning"
            )
            return []
            
        # Log the PDF files found
        logger.info(f"Found {len(pdf_files)} PDF files: {', '.join(f.name for f in pdf_files)}")
        
        # Load documents
        loader = PyPDFDirectoryLoader(str(pdf_dir), glob="*.pdf", silent_errors=True)
        docs = loader.load()
        num_docs = len(docs)
        
        if num_docs == 0:
            logger.warning(f"No content extracted from PDFs in {path}. Check that files are valid PDFs.")
            create_markdown_artifact(
                key="document-loading-warning",
                markdown=f"# ⚠️ No Content Extracted\n\nNo content could be extracted from PDFs in: `{path}`.\nFiles may be corrupted, empty, or password-protected.",
                description="Document loading warning"
            )
            return []
            
        logger.info(f"Successfully loaded {num_docs} pages across {len(pdf_files)} PDF files")
        
        # Add metadata to each document
        for doc in docs:
            # Add loading timestamp if not present
            if "source" in doc.metadata and "timestamp" not in doc.metadata:
                doc.metadata["timestamp"] = datetime.now().isoformat()
        
        # Calculate average content length for statistics
        avg_length = sum(len(doc.page_content) for doc in docs) / num_docs if num_docs > 0 else 0
        
        # Create artifact with document loading summary
        create_table_artifact(
            key="document-loading-summary",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["PDF Files Found", len(pdf_files)],
                    ["Pages Loaded", num_docs],
                    ["Avg Content Length", f"{int(avg_length)} chars"],
                    ["Source Directory", str(pdf_dir)],
                    ["Execution Time (s)", f"{time.time() - execution_start:.2f}"]
                ]
            },
            description="Summary of document loading operation"
        )
        
        # Emit event about document loading
        execution_time_ms = int((time.time() - execution_start) * 1000)
        emit_event(
            event="documents-loaded",
            resource={"prefect.resource.id": f"ragas-pipeline.documents.{path}"},
            payload={
                "document_count": num_docs,
                "file_count": len(pdf_files),
                "source_path": str(pdf_dir),
                "execution_time_ms": execution_time_ms,
                "avg_content_length": int(avg_length)
            }
        )
        
        return docs
        
    except ValueError as e:
        # Path validation error
        logger.error(f"Path validation error: {str(e)}")
        create_markdown_artifact(
            key="document-loading-error",
            markdown=f"# ❌ Document Loading Failed\n\n**Path Error:**\n```\n{str(e)}\n```",
            description="Document loading path error"
        )
        raise
        
    except Exception as e:
        logger.error(f"Error loading documents from {path}: {str(e)}")
        create_markdown_artifact(
            key="document-loading-error",
            markdown=f"# ❌ Document Loading Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Source Directory:** {path}",
            description="Document loading error"
        )
        raise RuntimeError(f"Failed to load documents from {path}: {str(e)}")

@task(
    name="initialize-ragas-generator",
    description="Initializes the RAGAS TestsetGenerator with LLM and embedding models",
    retries=3,
    retry_delay_seconds=20,
    tags=["ragas", "setup"],
    persist_result=False,  # Prevent serialization errors with TestsetGenerator
    task_run_name="Initialize RAGAS generator with {llm_model}"
)
def initialize_ragas_generator(
    llm_model: str,
    embedding_model: str
) -> TestsetGenerator:
    """
    Initialize the RAGAS TestsetGenerator with specified models.
    
    Args:
        llm_model: OpenAI model identifier for generation
        embedding_model: OpenAI embedding model identifier
        
    Returns:
        Configured TestsetGenerator instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        logger.info(f"Initializing RAGAS TestsetGenerator with LLM: {llm_model}, Embeddings: {embedding_model}")
        
        # Initialize models and validate API connectivity
        llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
        
        # Create generator
        generator = TestsetGenerator(llm=llm, embedding_model=emb)
        
        # Log successful initialization
        execution_time_ms = int((time.time() - execution_start) * 1000)
        logger.info(f"✅ Successfully initialized RAGAS generator in {execution_time_ms}ms")
        
        # Create artifact
        create_table_artifact(
            key="ragas-initialization",
            table={
                "columns": ["Component", "Value"],
                "data": [
                    ["LLM Model", llm_model],
                    ["Embedding Model", embedding_model],
                    ["Initialization Time (ms)", execution_time_ms]
                ]
            },
            description="RAGAS TestsetGenerator initialization"
        )
        
        # Emit event
        emit_event(
            event="ragas-generator-initialized",
            resource={"prefect.resource.id": "ragas-pipeline.generator"},
            payload={
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "execution_time_ms": execution_time_ms
            }
        )
        
        return generator
        
    except Exception as e:
        logger.error(f"Error initializing RAGAS generator: {str(e)}")
        create_markdown_artifact(
            key="ragas-initialization-error",
            markdown=f"# ❌ RAGAS Initialization Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Parameters:**\n- LLM Model: {llm_model}\n- Embedding Model: {embedding_model}",
            description="RAGAS initialization error"
        )
        raise RuntimeError(f"Failed to initialize RAGAS generator: {str(e)}")

@task(
    name="generate-testset",
    description="Generates a RAGAS testset from documents",
    retries=1,
    retry_delay_seconds=60,
    # cache_policy=NO_CACHE,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    timeout_seconds=1800,  # 30 minutes
    tags=["ragas", "generation"],
    task_run_name="Generate testset with {size} samples"
)
def generate_testset(
    generator: TestsetGenerator,
    docs: List[Document],
    size: int
) -> tuple:  # (dataset, knowledge_graph)
    """
    Generate a RAGAS testset from the provided documents.
    
    Args:
        generator: Initialized TestsetGenerator
        docs: List of Document objects to use for generation
        size: Number of QA pairs to generate
        
    Returns:
        Tuple of (dataset, knowledge_graph)
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If generation fails
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        # Validate inputs
        if not docs:
            raise ValueError("No documents provided for testset generation")
        if size <= 0:
            raise ValueError(f"Testset size must be positive, got {size}")
            
        # Log generation start
        logger.info(f"Generating testset with {size} samples from {len(docs)} documents...")
        
        # Generate testset
        dataset = generator.generate_with_langchain_docs(docs, testset_size=size)
        
        # Get knowledge graph
        kg = generator.knowledge_graph
        
        # Calculate statistics
        execution_time_s = time.time() - execution_start
        
        # Log successful generation
        logger.info(f"✅ Successfully generated testset with {size} samples in {execution_time_s:.2f}s")
        
        # Create artifact
        create_table_artifact(
            key="testset-generation",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Testset Size", size],
                    ["Document Count", len(docs)],
                    ["Generation Time (s)", f"{execution_time_s:.2f}"]
                ]
            },
            description="RAGAS testset generation summary"
        )
        
        # Emit event
        emit_event(
            event="testset-generated",
            resource={"prefect.resource.id": "ragas-pipeline.testset"},
            payload={
                "testset_size": size,
                "document_count": len(docs),
                "execution_time_s": round(execution_time_s, 2)
            }
        )
        
        return dataset, kg
        
    except ValueError as e:
        # Input validation error
        logger.error(f"Validation error in testset generation: {str(e)}")
        create_markdown_artifact(
            key="testset-generation-error",
            markdown=f"# ❌ Testset Generation Failed\n\n**Validation Error:**\n```\n{str(e)}\n```\n\n**Parameters:**\n- Document Count: {len(docs) if docs else 0}\n- Requested Size: {size}",
            description="Testset generation validation error"
        )
        raise
        
    except Exception as e:
        # Generic error handler
        logger.error(f"Error generating testset: {str(e)}")
        create_markdown_artifact(
            key="testset-generation-error",
            markdown=f"# ❌ Testset Generation Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Parameters:**\n- Document Count: {len(docs) if docs else 0}\n- Requested Size: {size}",
            description="Testset generation error"
        )
        raise RuntimeError(f"Failed to generate testset: {str(e)}")

@task(
    name="save-knowledge-graph",
    description="Saves the RAGAS knowledge graph to a file",
    retries=2,
    retry_delay_seconds=10,
    tags=["ragas", "output"],
    task_run_name="Save knowledge graph to {path}"
)
def save_knowledge_graph(
    knowledge_graph: Any,
    path: str
) -> str:
    """
    Save the RAGAS knowledge graph to a file.
    
    Args:
        knowledge_graph: Knowledge graph from TestsetGenerator
        path: Path to save the knowledge graph
        
    Returns:
        Path to the saved knowledge graph
        
    Raises:
        ValueError: If path validation fails
        RuntimeError: If saving fails
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        # Validate and create directory if needed
        kg_path = validate_path(path, create_if_missing=True)
        logger.info(f"Saving knowledge graph to {kg_path}")
        
        # Check if output file exists before proceeding
        if kg_path.exists():
            # Create an artifact to document that the file already exists
            create_markdown_artifact(
                key="knowledge-graph-status",
                markdown=f"# Knowledge Graph File Found\nExisting knowledge graph found at `{kg_path}`.\nRemoving file to regenerate.",
                description="Status of knowledge graph save operation"
            )
            # Delete the existing file to ensure we regenerate
            kg_path.unlink()
            logger.info(f"Removed existing knowledge graph at {kg_path}")
        
        # Save the knowledge graph
        knowledge_graph.save(str(kg_path))
        
        # Calculate file size
        file_size = kg_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Log successful save
        execution_time_ms = int((time.time() - execution_start) * 1000)
        logger.info(f"✅ Successfully saved knowledge graph to {kg_path} ({file_size_mb:.2f} MB) in {execution_time_ms}ms")
        
        # Create success artifact
        create_markdown_artifact(
            key="knowledge-graph-status",
            markdown=f"# ✅ Knowledge Graph Saved\nSuccessfully saved knowledge graph to `{kg_path}`.\n\n**Details:**\n- File size: {file_size_mb:.2f} MB\n- Save time: {execution_time_ms}ms",
            description="Status of knowledge graph save operation"
        )
        
        # Create a link artifact for easy access
        file_uri = f"file://{os.path.abspath(kg_path)}"
        create_link_artifact(
            key="knowledge-graph-file",
            link=file_uri,
            link_text="Knowledge Graph JSON File",
            description="Link to the saved knowledge graph file"
        )
        
        # Emit event
        emit_event(
            event="knowledge-graph-saved",
            resource={"prefect.resource.id": f"ragas-pipeline.knowledge-graph.{path}"},
            payload={
                "path": str(kg_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "execution_time_ms": execution_time_ms
            }
        )
        
        return str(kg_path)
        
    except ValueError as e:
        # Path validation error
        logger.error(f"Path validation error: {str(e)}")
        create_markdown_artifact(
            key="knowledge-graph-save-error",
            markdown=f"# ❌ Knowledge Graph Save Failed\n\n**Path Error:**\n```\n{str(e)}\n```\n\n**Path:** {path}",
            description="Knowledge graph save path error"
        )
        raise
        
    except Exception as e:
        # Generic error handler
        logger.error(f"Error saving knowledge graph: {str(e)}")
        create_markdown_artifact(
            key="knowledge-graph-save-error",
            markdown=f"# ❌ Knowledge Graph Save Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Path:** {path}",
            description="Knowledge graph save error"
        )
        raise RuntimeError(f"Failed to save knowledge graph: {str(e)}")

@task(
    name="prepare-hf-dataset",
    description="Prepares RAGAS dataset for HuggingFace upload",
    retries=1,
    retry_delay_seconds=20,
    tags=["huggingface", "processing"],
    task_run_name="Prepare dataset for HF upload"
)
def prepare_hf_dataset(dataset: Any) -> Any:
    """
    Convert RAGAS dataset to HuggingFace dataset format.
    
    Args:
        dataset: RAGAS dataset object
        
    Returns:
        HuggingFace dataset object
        
    Raises:
        RuntimeError: If conversion fails
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        logger.info("Converting RAGAS dataset to HuggingFace format")
        
        # Convert to HF dataset
        hf_dataset = dataset.to_hf_dataset()
        
        # Calculate statistics
        sample_count = len(hf_dataset) if hasattr(hf_dataset, "__len__") else "Unknown"
        execution_time_ms = int((time.time() - execution_start) * 1000)
        
        logger.info(f"✅ Successfully converted dataset to HF format with {sample_count} samples in {execution_time_ms}ms")
        
        # Create artifact
        create_table_artifact(
            key="hf-dataset-preparation",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Sample Count", sample_count],
                    ["Format", "HuggingFace Dataset"],
                    ["Conversion Time (ms)", execution_time_ms]
                ]
            },
            description="HuggingFace dataset preparation"
        )
        
        # Emit event
        emit_event(
            event="hf-dataset-prepared",
            resource={"prefect.resource.id": "ragas-pipeline.hf-dataset"},
            payload={
                "sample_count": sample_count,
                "execution_time_ms": execution_time_ms
            }
        )
        
        return hf_dataset
        
    except Exception as e:
        logger.error(f"Error preparing HuggingFace dataset: {str(e)}")
        create_markdown_artifact(
            key="hf-dataset-preparation-error",
            markdown=f"# ❌ HuggingFace Dataset Preparation Failed\n\n**Error:**\n```\n{str(e)}\n```",
            description="HuggingFace dataset preparation error"
        )
        raise RuntimeError(f"Failed to prepare HuggingFace dataset: {str(e)}")

@task(
    name="push-to-hub",
    description="Pushes the generated testset to a Hugging Face repository",
    retries=2,
    retry_delay_seconds=60,
    timeout_seconds=600,
    tags=["huggingface", "publish"],
    task_run_name="Push dataset to {repo_name}"
)
def push_to_hub(dataset: Any, repo_name: str) -> str:
    """
    Push the generated testset to a Hugging Face repository.
    
    Args:
        dataset: HuggingFace dataset to push
        repo_name: Name of the target HuggingFace repository
        
    Returns:
        URL of the published dataset
        
    Raises:
        ValueError: If HF_TOKEN is not set
        RuntimeError: If push fails
    """
    logger = get_run_logger()
    execution_start = time.time()
    
    try:
        logger.info(f"Pushing dataset to Hugging Face repository: {repo_name}")
        
        # Get token
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set")
            
        # Login to HuggingFace
        login(token=token, add_to_git_credential=False)
        logger.info(f"Successfully authenticated with HuggingFace Hub")
        
        # Push dataset
        dataset.push_to_hub(repo_name)
        
        # Build URL for display
        url = f"https://huggingface.co/datasets/{repo_name}"
        execution_time_s = time.time() - execution_start
        
        logger.info(f"✅ Successfully pushed dataset to {url} in {execution_time_s:.2f}s")
        
        # Create artifact for successful push
        create_markdown_artifact(
            key="huggingface-push",
            markdown=f"# ✅ Dataset Published to Hugging Face\n\n**Repository:** [{repo_name}]({url})\n\n**Details:**\n- Push Time: {execution_time_s:.2f}s",
            description="HuggingFace repository information"
        )
        
        # Create link artifact
        create_link_artifact(
            key="huggingface-repo",
            link=url,
            link_text=f"HuggingFace Dataset: {repo_name}",
            description="Link to the published HuggingFace dataset"
        )
        
        # Emit event for successful push
        emit_event(
            event="dataset-published",
            resource={"prefect.resource.id": f"ragas-pipeline.dataset.{repo_name}"},
            payload={
                "repo_name": repo_name,
                "url": url,
                "platform": "huggingface",
                "execution_time_s": round(execution_time_s, 2)
            }
        )
        
        return url
        
    except ValueError as e:
        # Configuration error
        logger.error(f"HuggingFace configuration error: {str(e)}")
        create_markdown_artifact(
            key="huggingface-push-error",
            markdown=f"# ❌ Failed to Publish Dataset\n\n**Configuration Error:**\n```\n{str(e)}\n```\n\n**Repository:** {repo_name}",
            description="HuggingFace repository configuration error"
        )
        raise
        
    except Exception as e:
        # Generic error handler
        logger.error(f"Failed to push dataset to Hugging Face: {str(e)}")
        create_markdown_artifact(
            key="huggingface-push-error",
            markdown=f"# ❌ Failed to Publish Dataset\n\n**Error:**\n```\n{str(e)}\n```\n\n**Repository:** {repo_name}",
            description="HuggingFace repository error"
        )
        raise RuntimeError(f"Failed to push dataset to HuggingFace: {str(e)}")

@flow(
    name="RAGAS Golden Dataset Pipeline",
    description="Generates a RAG test dataset and knowledge graph from PDF documents",
    log_prints=True,
    version=os.environ.get("PIPELINE_VERSION", "1.0.0"),
    task_runner=ConcurrentTaskRunner(),
    validate_parameters=True
)
def ragas_pipeline(
    docs_path: str = "data/",
    testset_size: int = 10,
    knowledge_graph_path: str = "output/kg.json",
    hf_repo: str = "",
    llm_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    """
    Orchestrates the full pipeline:
      1. Validate environment
      2. Download PDFs (if needed)
      3. Load documents
      4. Generate testset & KG
      5. Save KG as JSON
      6. (Optional) Push testset to HF Hub
      
    Args:
        docs_path: Directory with source documents
        testset_size: Number of test samples to generate
        knowledge_graph_path: File path for the serialized knowledge graph
        hf_repo: HuggingFace repository to push dataset to (optional)
        llm_model: LLM model to use for generation
        embedding_model: Embedding model to use
        
    Returns:
        Dictionary with pipeline results and statistics
    """
    logger = get_run_logger()
    execution_start = time.time()
    logger.info(f"🚀 Starting RAGAS Golden Dataset Pipeline")
    
    try:
        # Validate input parameters
        validate_int_range(testset_size, "testset_size", min_val=1, max_val=100)
        
        # First validate environment variables
        env_vars = validate_environment()
        
        # Create artifact with pipeline configuration
        create_table_artifact(
            key="pipeline-configuration",
            table={
                "columns": ["Parameter", "Value"],
                "data": [
                    ["Documents Path", docs_path],
                    ["Testset Size", testset_size],
                    ["Knowledge Graph Path", knowledge_graph_path],
                    ["Hugging Face Repo", hf_repo or "Not specified"],
                    ["LLM Model", llm_model],
                    ["Embedding Model", embedding_model],
                    ["Pipeline Version", os.environ.get("PIPELINE_VERSION", "1.0.0")]
                ]
            },
            description="Pipeline configuration parameters"
        )
        
        # Download PDFs if directory is empty - submit for parallel execution
        docs_path_future = None
        if not any(Path(docs_path).glob("*.pdf")):
            logger.info(f"No PDFs found in {docs_path}, downloading samples...")
            docs_path_future = download_pdfs.submit(docs_path)
        else:
            logger.info(f"Found existing PDFs in {docs_path}, skipping download")
            
        # Wait for download if it was running, otherwise use the original path
        docs_path_resolved = docs_path_future.result() if docs_path_future else docs_path
        
        # Load documents
        logger.info(f"Loading documents from {docs_path_resolved}")
        docs = load_documents(docs_path_resolved)
        
        if not docs:
            msg = f"No documents were loaded from {docs_path_resolved}. Cannot proceed with testset generation."
            logger.error(msg)
            create_markdown_artifact(
                key="pipeline-error",
                markdown=f"# ❌ Pipeline Aborted\n\n{msg}",
                description="Pipeline error - no documents"
            )
            return {"status": "failed", "reason": "No documents loaded"}
        
        # Initialize RAGAS generator
        logger.info(f"Initializing RAGAS generator with {llm_model} and {embedding_model}")
        generator = initialize_ragas_generator(llm_model, embedding_model)
        
        # Generate testset
        logger.info(f"Generating testset of size {testset_size}")
        dataset, kg = generate_testset(generator, docs, testset_size)
        
        # Save knowledge graph
        logger.info(f"Saving knowledge graph to {knowledge_graph_path}")
        kg_path = save_knowledge_graph(kg, knowledge_graph_path)
        
        # Push to HuggingFace if repo specified
        hf_url = None
        if hf_repo:
            logger.info(f"Preparing dataset for HuggingFace")
            hf_dataset = prepare_hf_dataset.submit(dataset)
            
            logger.info(f"Pushing dataset to HuggingFace repo {hf_repo}")
            hf_url = push_to_hub(hf_dataset.result(), hf_repo)
        else:
            logger.info("No HuggingFace repository specified, skipping push step")
        
        # Calculate pipeline execution time
        execution_time_s = time.time() - execution_start
        logger.info(f"✅ RAGAS Golden Dataset Pipeline completed successfully in {execution_time_s:.2f}s")
        
        # Final success artifact
        create_markdown_artifact(
            key="pipeline-summary",
            markdown=f"""# Pipeline Execution Summary

## Success! ✅

The RAGAS Golden Dataset Pipeline completed successfully in {execution_time_s:.2f} seconds.

## Outputs
- Knowledge Graph: `{knowledge_graph_path}`
- Test Set Size: {testset_size}
- Documents Processed: {len(docs)} pages
{f'- Published to: [{hf_repo}]({hf_url})' if hf_url else ''}

## Models Used
- LLM: {llm_model}
- Embeddings: {embedding_model}
""",
            description="Pipeline execution summary"
        )
        
        # Return pipeline results
        return {
            "status": "success",
            "testset_size": testset_size,
            "document_count": len(docs),
            "knowledge_graph_path": kg_path,
            "huggingface_url": hf_url,
            "execution_time_s": round(execution_time_s, 2)
        }
        
    except Exception as e:
        # Generic error handler for the entire flow
        execution_time_s = time.time() - execution_start
        logger.error(f"Pipeline failed after {execution_time_s:.2f}s: {str(e)}")
        
        create_markdown_artifact(
            key="pipeline-failure",
            markdown=f"# ❌ Pipeline Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Execution Time:** {execution_time_s:.2f}s",
            description="Pipeline failure summary"
        )
        
        # Re-raise the exception to ensure Prefect marks the flow as failed
        raise


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="RAGAS Golden Dataset Pipeline")
    
    # Required parameters with environment variable defaults
    parser.add_argument("--docs-path", type=str, 
                        default=os.environ.get("RAW_DIR", "data/"),
                        help="Directory with source documents")
    parser.add_argument("--testset-size", type=int, 
                        default=int(os.environ.get("TESTSET_SIZE", "10")),
                        help="Number of test samples to generate")
    parser.add_argument("--kg-output", type=str, 
                        default=os.environ.get("KG_OUTPUT_PATH", "output/kg.json"),
                        help="File path for the serialized knowledge graph")
    parser.add_argument("--hf-repo", type=str, 
                        default=os.environ.get("HF_TESTSET_REPO_v1", ""),
                        help="(Optional) HF Hub repository name to push the dataset")
    parser.add_argument("--llm-model", type=str,
                        default=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
                        help="LLM model to use for testset generation")
    parser.add_argument("--embedding-model", type=str,
                        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                        help="Embedding model to use for testset generation")
    parser.add_argument("--version", action="store_true",
                        help="Show pipeline version and exit")
    
    args = parser.parse_args()
    
    # Get pipeline version from environment or default
    pipeline_version = os.environ.get("PIPELINE_VERSION", "1.0.0")
    
    # Handle --version flag
    if args.version:
        print(f"RAGAS Golden Dataset Pipeline version {pipeline_version}")
        exit(0)
    
    print("Starting RAGAS Golden Dataset Pipeline v" + pipeline_version)
    print("=========================================")
    print(f"- Documents Path: {args.docs_path}")
    print(f"- Testset Size: {args.testset_size}")
    print(f"- Knowledge Graph Output: {args.kg_output}")
    print(f"- Hugging Face Repo: {args.hf_repo or 'Not specified'}")
    print(f"- LLM Model: {args.llm_model}")
    print(f"- Embedding Model: {args.embedding_model}")
    print("=========================================")
    
    try:
        # Validate parameters before running
        validate_path(args.docs_path, create_if_missing=True)
        validate_int_range(args.testset_size, "testset_size", min_val=1, max_val=100)
        validate_path(args.kg_output, create_if_missing=True)
        
        # Run the pipeline with parsed arguments
        result = ragas_pipeline(
            docs_path=args.docs_path,
            testset_size=args.testset_size,
            knowledge_graph_path=args.kg_output,
            hf_repo=args.hf_repo,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model
        )
        
        # Print summary of results
        if result and result.get("status") == "success":
            print("\nPipeline completed successfully!")
            print(f"- Documents processed: {result.get('document_count', 0)}")
            print(f"- KG path: {result.get('knowledge_graph_path', args.kg_output)}")
            if result.get("huggingface_url"):
                print(f"- Published to: {result.get('huggingface_url')}")
            print(f"- Total execution time: {result.get('execution_time_s', 0):.2f}s")
        else:
            print("\nPipeline completed with issues:")
            print(f"- Status: {result.get('status', 'unknown')}")
            print(f"- Reason: {result.get('reason', 'unknown')}")
            exit(1)
            
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        exit(1)
