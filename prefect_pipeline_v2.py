import os
import subprocess
import json
import pickle
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.artifacts import (
    create_table_artifact,
    create_markdown_artifact,
    create_link_artifact,
)
from prefect.cache_policies import NO_CACHE
from prefect.tasks import task_input_hash
from prefect.events import emit_event

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.documents import Document
from huggingface_hub import login
import requests
import urllib3

load_dotenv()  # Load variables from .env into os.environ

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

SENSITIVE_VARS = {"OPENAI_API_KEY", "HF_TOKEN"}

def mask_value(value: str) -> str:
    """Mask sensitive values: show first 4 + last 4 characters."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]

# ------------------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------------------

@task(
    name="validate-environment",
    description="Validates all required environment variables are set",
    retries=1,
    retry_delay_seconds=5,
    tags=["setup", "validation"]
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

    required_vars_by_category = {
        "OpenAI": ["OPENAI_API_KEY"],
        "HuggingFace": ["HF_TOKEN", "HF_TESTSET_REPO_V2"],
        "LLM Config": ["LLM_MODEL", "EMBEDDING_MODEL"],
        "Pipeline Config": ["RAW_DIR", "TESTSET_SIZE", "KG_OUTPUT_PATH"],
        "Prefect": ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "PREFECT_API_URL"],
    }

    # Define any optional variable categories here if needed
    optional_vars_by_category = {
    }

    missing = {}
    for cat, keys in required_vars_by_category.items():
        miss = [k for k in keys if not os.environ.get(k)]
        if miss:
            missing[cat] = miss

    if missing:
        lines = ["❌ Missing required environment variables:"]
        for cat, keys in missing.items():
            lines.append(f"• {cat}: {', '.join(keys)}")
        lines.append("Please set them in your environment or .env file.")
        msg = "\n".join(lines)
        logger.error(msg)
        
        # Create artifact for the error
        create_markdown_artifact(
            key="environment-validation-error",
            markdown=f"# ❌ Environment Validation Failed\n\n{msg}",
            description="Missing environment variables"
        )
        
        raise EnvironmentError(msg)

    # Build table of resolved values
    table = [["Variable", "Value"]]
    # Add required vars to table
    for cat, keys in required_vars_by_category.items():
        for k in keys:
            raw = os.environ.get(k, "")
            val = mask_value(raw) if k in SENSITIVE_VARS else raw
            table.append([k, val])
            logger.info(f"{k} = {val}")
    
    # Add optional vars to table if they exist
    for cat, keys in optional_vars_by_category.items():
        for k in keys:
            raw = os.environ.get(k, "")
            if raw:  # Only include if the variable exists
                val = mask_value(raw) if k in SENSITIVE_VARS else raw
                table.append([k, val])
                logger.info(f"{k} = {val}")

    create_table_artifact(
        key="env-vars",
        table={"columns": table[0], "data": table[1:]},
        description="Resolved environment variables (sensitive masked)"
    )
    logger.info("✔️ Environment validated.")
    
    # Return a dictionary of set variables by category for potential use in tasks
    return {cat: [k for k in keys if os.environ.get(k)] 
            for cat, keys in {**required_vars_by_category, **optional_vars_by_category}.items()}

@task(
    name="validate-input-parameters",
    description="Validates input parameters for the pipeline",
    tags=["setup", "validation"]
)
def validate_input_parameters(
    docs_path: str,
    testset_size: int,
    kg_output_path: str,
    output_dir: str
) -> bool:
    """
    Validates input parameters for the pipeline.
    
    Args:
        docs_path: Directory path for input documents
        testset_size: Number of test samples to generate
        kg_output_path: Path to save knowledge graph
        output_dir: Directory for output files
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If any parameters are invalid
    """
    logger = get_run_logger()
    logger.info("🔍 Validating input parameters...")
    
    errors = []
    warnings = []
    
    # Validate docs_path
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        warnings.append(f"Documents path doesn't exist: {docs_path}")
        logger.warning(f"Documents path doesn't exist: {docs_path}. It will be created.")
        # Create the directory instead of failing
        docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate testset_size
    if testset_size <= 0:
        errors.append(f"Testset size must be positive: {testset_size}")
    
    # Validate output paths
    kg_output_dir = Path(kg_output_path).parent
    if not kg_output_dir.exists():
        warnings.append(f"Knowledge graph output directory doesn't exist: {kg_output_dir}")
        logger.warning(f"Knowledge graph output directory doesn't exist: {kg_output_dir}. It will be created when needed.")
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        warnings.append(f"Output directory doesn't exist: {output_dir}")
        logger.warning(f"Output directory doesn't exist: {output_dir}. It will be created when needed.")
    
    # Create validation summary artifact
    create_table_artifact(
        key="input-parameters",
        table={
            "columns": ["Parameter", "Value", "Status"],
            "data": [
                ["Documents Path", docs_path, "⚠️ Created" if not docs_dir.exists() else "✅"],
                ["Testset Size", str(testset_size), "❌" if testset_size <= 0 else "✅"],
                ["Knowledge Graph Path", kg_output_path, "⚠️ Parent directory will be created" if not kg_output_dir.exists() else "✅"],
                ["Output Directory", output_dir, "⚠️ Will be created" if not output_dir_path.exists() else "✅"]
            ]
        },
        description="Validation status of input parameters"
    )
    
    # If there are warnings but no errors, create a warning artifact
    if warnings and not errors:
        warning_msg = "\n".join(f"- {w}" for w in warnings)
        create_markdown_artifact(
            key="input-validation-warnings",
            markdown=f"# ⚠️ Input Validation Warnings\n\n{warning_msg}\n\nThese issues will be handled automatically.",
            description="Input validation warnings"
        )
    
    # If there are errors, create an error artifact and raise an exception
    if errors:
        error_msg = "\n".join(f"- {e}" for e in errors)
        logger.error(f"Input validation failed: {error_msg}")
        create_markdown_artifact(
            key="input-validation-error",
            markdown=f"# ❌ Input Validation Failed\n\n{error_msg}",
            description="Input validation errors"
        )
        raise ValueError(f"Input validation failed: {error_msg}")
    
    logger.info("✔️ Input validation successful.")
    return True

@task(
    name="download-pdfs", 
    description="Downloads sample PDF papers from arXiv",
    retries=3, 
    retry_delay_seconds=30,
    tags=["data", "download"]
)
def download_pdfs(docs_path: str) -> str:
    """
    Downloads sample PDF papers from arXiv.
    
    Args:
        docs_path: Directory to save PDFs to
        
    Returns:
        Path to the directory containing the PDFs
    """
    logger = get_run_logger()
    pdf_dir = Path(docs_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        ("https://arxiv.org/pdf/2505.10468.pdf", "ai_agents_vs_agentic_ai_2505.10468.pdf"),
        ("https://arxiv.org/pdf/2505.06913.pdf", "redteamllm_agentic_ai_framework_2505.06913.pdf"),
        ("https://arxiv.org/pdf/2505.06817.pdf", "control_plane_scalable_design_pattern_2505.06817.pdf"),
    ]

    success = failed = skipped = 0
    failed_files = []
    
    try:
        for url, filename in urls:
            out_path = pdf_dir / filename
            if out_path.exists():
                logger.info(f"Skipping existing {filename}")
                skipped += 1
                continue

            try:
                logger.info(f"Downloading via curl: {filename}")
                subprocess.run(
                    ["curl", "-L", "--ssl-no-revoke", url, "-o", str(out_path)],
                    check=True,
                )
                success += 1
                logger.info(f"✅ Successfully downloaded {filename}")
            except Exception as e:
                logger.warning(f"curl failed ({e}), falling back to requests")
                try:
                    # Disable insecure request warnings
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    
                    resp = requests.get(url, verify=False)
                    resp.raise_for_status()
                    out_path.write_bytes(resp.content)
                    success += 1
                    logger.info(f"✅ Successfully downloaded {filename} using requests")
                except Exception as inner_e:
                    logger.error(f"Failed to download {filename}: {str(inner_e)}")
                    failed += 1
                    failed_files.append((filename, str(inner_e)))

        # Create artifact with download results
        table = [
            ["Successful Downloads", success],
            ["Failed Downloads", failed],
            ["Skipped (Already Existed)", skipped],
            ["Total PDFs", success + skipped],
        ]
        create_table_artifact(
            key="pdf-download-summary",
            table={"columns": ["Metric", "Value"], "data": table},
            description="PDF download summary"
        )
        
        # If there were failures, create a warning artifact
        if failed > 0:
            failures_md = "\n".join([f"- **{name}**: {error}" for name, error in failed_files])
            create_markdown_artifact(
                key="pdf-download-warnings",
                markdown=f"# ⚠️ Some PDF Downloads Failed\n\n{failures_md}",
                description="Details of failed PDF downloads"
            )

        # Emit event about download completion
        emit_event(
            event="pdfs-downloaded",
            resource={"prefect.resource.id": f"ragas-pipeline.data.{docs_path}"},
            payload={
                "successful": success,
                "failed": failed,
                "skipped": skipped,
                "data_path": docs_path
            }
        )
        
        # Check if we have at least one PDF and warn if not
        if success + skipped == 0:
            logger.error("No PDFs were downloaded or found in the directory.")
            create_markdown_artifact(
                key="pdf-download-error",
                markdown=f"# ❌ No PDFs Available\n\nNo PDFs were downloaded or found in directory: `{docs_path}`",
                description="PDF download critical error"
            )
            raise RuntimeError(f"No PDFs available in {docs_path}")
            
        return docs_path
    
    except Exception as e:
        logger.error(f"Error in PDF download task: {str(e)}")
        create_markdown_artifact(
            key="pdf-download-error",
            markdown=f"# ❌ PDF Download Failed\n\n**Error:**\n```\n{str(e)}\n```",
            description="PDF download error"
        )
        raise

@task(
    name="load-documents",
    description="Loads PDF documents using LangChain's PyPDFDirectoryLoader",
    retries=2, 
    retry_delay_seconds=10,
    tags=["data", "loading"]
)
def load_documents(docs_path: str) -> List[Document]:
    """
    Loads PDF documents from a directory.
    
    Args:
        docs_path: Directory path containing PDF files
        
    Returns:
        List of Document objects
    """
    logger = get_run_logger()
    logger.info(f"Loading documents from {docs_path}")
    
    try:
        # Ensure directory exists
        Path(docs_path).mkdir(parents=True, exist_ok=True)
        
        # Check if there are any PDF files
        pdf_files = list(Path(docs_path).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {docs_path}")
            create_markdown_artifact(
                key="document-loading-warning",
                markdown=f"# ⚠️ No PDF Files Found\n\nNo PDF files were found in directory: `{docs_path}`",
                description="Document loading warning"
            )
            return []
        
        loader = PyPDFDirectoryLoader(docs_path, glob="*.pdf", silent_errors=True)
        docs = loader.load()
        count = len(docs)
        
        if count == 0:
            logger.warning(f"No content could be extracted from PDFs in {docs_path}")
            create_markdown_artifact(
                key="document-loading-warning",
                markdown=f"# ⚠️ No Content Extracted\n\nNo content could be extracted from PDFs in: `{docs_path}`",
                description="Document loading warning"
            )
            return []
            
        logger.info(f"Loaded {count} pages from {len(pdf_files)} files")

        # Add metadata to each document
        for doc in docs:
            # Add loading timestamp if not present
            if "source" in doc.metadata and "timestamp" not in doc.metadata:
                doc.metadata["timestamp"] = datetime.now().isoformat()
        
        # Create artifact with loading statistics
        create_table_artifact(
            key="load-summary",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["PDF Files Found", len(pdf_files)],
                    ["Pages Loaded", count],
                    ["Source Directory", docs_path]
                ]
            },
            description="Document load summary"
        )

        # Emit event about document loading
        emit_event(
            event="documents-loaded",
            resource={"prefect.resource.id": f"ragas-pipeline.documents.{docs_path}"},
            payload={
                "document_count": count,
                "file_count": len(pdf_files),
                "source_dir": docs_path
            }
        )
        
        return docs
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        create_markdown_artifact(
            key="document-loading-error",
            markdown=f"# ❌ Document Loading Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Source Directory:** {docs_path}",
            description="Document loading error"
        )
        raise

@task(
    name="save-documents",
    description="Saves documents to disk in multiple formats",
    tags=["data", "output"]
)
def save_documents(
    docs: List[Document], 
    output_dir: str, 
    formats: List[str] = ["pkl", "json"]
) -> List[str]:
    """
    Saves documents to disk in multiple formats.
    
    Args:
        docs: List of document objects to save
        output_dir: Directory to save documents to
        formats: List of formats to save in (pkl, json)
        
    Returns:
        List of paths to saved files
    """
    logger = get_run_logger()
    
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        # Save as pickle if requested
        if "pkl" in formats:
            try:
                pkl_path = output_dir_path / "documents.pkl"
                logger.info(f"Saving {len(docs)} documents to {pkl_path} (pickle format)")
                with open(pkl_path, "wb") as f:
                    pickle.dump(docs, f)
                saved_paths.append(str(pkl_path))
                
                create_link_artifact(
                    key="documents-pkl",
                    link=f"file://{os.path.abspath(pkl_path)}",
                    link_text="Saved Documents (Pickle)",
                    description="Link to the saved documents in pickle format"
                )
                logger.info(f"✅ Successfully saved documents to {pkl_path}")
            except Exception as e:
                logger.error(f"Failed to save pickle format: {str(e)}")
                create_markdown_artifact(
                    key="documents-pkl-error",
                    markdown=f"# ❌ Failed to Save Pickle\n\n**Error:**\n```\n{str(e)}\n```",
                    description="Error saving documents in pickle format"
                )
        
        # Save as JSON if requested
        if "json" in formats:
            try:
                json_path = output_dir_path / "documents.json"
                logger.info(f"Saving {len(docs)} documents to {json_path} (JSON format)")
                
                # Convert to a serializable format
                serializable_docs = []
                for doc in docs:
                    # Handle non-serializable metadata
                    clean_metadata = {}
                    for k, v in doc.metadata.items():
                        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                            clean_metadata[k] = v
                        else:
                            # Convert non-serializable types to string
                            clean_metadata[k] = str(v)
                    
                    serializable_docs.append({
                        "page_content": doc.page_content,
                        "metadata": clean_metadata
                    })
                
                # Save to JSON
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
                saved_paths.append(str(json_path))
                
                create_link_artifact(
                    key="documents-json",
                    link=f"file://{os.path.abspath(json_path)}",
                    link_text="Saved Documents (JSON)",
                    description="Link to the saved documents in JSON format"
                )
                logger.info(f"✅ Successfully saved documents to {json_path}")
            except Exception as e:
                logger.error(f"Failed to save JSON format: {str(e)}")
                create_markdown_artifact(
                    key="documents-json-error",
                    markdown=f"# ❌ Failed to Save JSON\n\n**Error:**\n```\n{str(e)}\n```",
                    description="Error saving documents in JSON format"
                )
        
        # Create artifact with saving statistics
        create_table_artifact(
            key="document-save-summary",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Documents Saved", len(docs)],
                    ["Formats", ", ".join(formats)],
                    ["Output Files", ", ".join([os.path.basename(p) for p in saved_paths])],
                    ["Output Directory", output_dir]
                ]
            },
            description="Document save summary"
        )
        
        # Emit event about document saving
        emit_event(
            event="documents-saved",
            resource={"prefect.resource.id": f"ragas-pipeline.documents.{output_dir}"},
            payload={
                "paths": saved_paths,
                "count": len(docs),
                "formats": formats
            }
        )
        
        logger.info(f"✅ Documents saved in formats: {', '.join(formats)}")
        return saved_paths
        
    except Exception as e:
        logger.error(f"Error saving documents: {str(e)}")
        create_markdown_artifact(
            key="document-save-error",
            markdown=f"# ❌ Document Saving Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Output Directory:** {output_dir}",
            description="Document saving error"
        )
        raise

@task(
    name="build-testset", 
    description="Builds a RAGAS testset and knowledge graph",
    retries=2, 
    retry_delay_seconds=30, 
    cache_policy=NO_CACHE,
    tags=["ragas", "generation"]
)
def build_testset(
    docs: List[Document],
    testset_size: int,
    kg_output_path: str,
    llm_model: str,
    embedding_model: str
) -> Any:  # Using Any because RAGAS dataset type is not importable here
    """
    Builds a RAGAS testset and knowledge graph.
    
    Args:
        docs: List of documents to use for testset generation
        testset_size: Number of QA pairs to generate
        kg_output_path: Path to save knowledge graph
        llm_model: OpenAI model to use for generation
        embedding_model: Embedding model to use
        
    Returns:
        Generated RAGAS dataset
    """
    logger = get_run_logger()
    logger.info(f"Building testset of size {testset_size} with {len(docs)} documents")
    
    try:
        # Initialize LLM and embedding models
        llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
        
        # Create TestsetGenerator
        logger.info(f"Initializing generator with LLM: {llm_model}, Embeddings: {embedding_model}")
        generator = TestsetGenerator(llm=llm, embedding_model=emb)

        # Generate testset
        logger.info(f"Generating testset with {testset_size} samples...")
        dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        
        # Save knowledge graph
        kg = generator.knowledge_graph
        kg_output_dir = Path(kg_output_path).parent
        kg_output_dir.mkdir(parents=True, exist_ok=True)
        kg.save(kg_output_path)
        logger.info(f"Knowledge graph saved to {kg_output_path}")

        # Create success artifacts
        create_markdown_artifact(
            key="testset-generation-success",
            markdown=f"# ✅ Testset Generation Successful\n\n**Generated:**\n- Testset with {testset_size} QA pairs\n- Knowledge graph saved to `{kg_output_path}`\n\n**Parameters:**\n- LLM: {llm_model}\n- Embeddings: {embedding_model}",
            description="Testset generation success"
        )
        
        create_link_artifact(
            key="kg-json",
            link=f"file://{os.path.abspath(kg_output_path)}",
            link_text="Knowledge Graph JSON File",
            description="Link to the generated knowledge graph"
        )

        # Emit event for successful generation
        emit_event(
            event="kg-generated",
            resource={"prefect.resource.id": f"ragas-pipeline.kg.{kg_output_path}"},
            payload={
                "path": kg_output_path,
                "testset_size": testset_size,
                "doc_count": len(docs),
                "llm_model": llm_model,
                "embedding_model": embedding_model
            }
        )
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error building testset: {str(e)}")
        create_markdown_artifact(
            key="testset-generation-error",
            markdown=f"# ❌ Testset Generation Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Parameters:**\n- Testset Size: {testset_size}\n- Document Count: {len(docs)}\n- LLM: {llm_model}\n- Embeddings: {embedding_model}",
            description="Testset generation error"
        )
        raise

@task(
    name="prepare-hf-dataset",
    description="Prepares RAGAS dataset for HuggingFace upload",
    tags=["huggingface", "processing"]
)
def prepare_hf_dataset(dataset: Any) -> Any:
    """
    Prepares RAGAS dataset for HuggingFace upload.
    
    Args:
        dataset: RAGAS dataset
        
    Returns:
        HuggingFace dataset ready for upload
    """
    logger = get_run_logger()
    
    try:
        logger.info("Converting RAGAS dataset to HuggingFace format")
        hf_dataset = dataset.to_hf_dataset()
        logger.info("Dataset successfully converted to HuggingFace format")
        
        # Create artifact with dataset statistics
        sample_count = len(hf_dataset) if hasattr(hf_dataset, "__len__") else "Unknown"
        create_table_artifact(
            key="hf-dataset-stats",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Sample Count", sample_count],
                    ["Format", "HuggingFace Dataset"]
                ]
            },
            description="HuggingFace dataset statistics"
        )
        
        return hf_dataset
        
    except Exception as e:
        logger.error(f"Error preparing HuggingFace dataset: {str(e)}")
        create_markdown_artifact(
            key="hf-dataset-preparation-error",
            markdown=f"# ❌ HuggingFace Dataset Preparation Failed\n\n**Error:**\n```\n{str(e)}\n```",
            description="HuggingFace dataset preparation error"
        )
        raise

@task(
    name="push-to-hub",
    description="Pushes dataset to HuggingFace Hub",
    retries=2,
    retry_delay_seconds=60,
    tags=["huggingface", "publish"]
)
def push_to_hub(hf_dataset: Any, repo_name: str) -> str:
    """
    Pushes a dataset to HuggingFace Hub.
    
    Args:
        hf_dataset: HuggingFace dataset to push
        repo_name: Name of the HuggingFace repository
        
    Returns:
        URL of the pushed dataset
    """
    logger = get_run_logger()
    
    try:
        # Get HF token
        token = os.environ["HF_TOKEN"]
        logger.info(f"Logging in to HuggingFace Hub with token")
        login(token=token, add_to_git_credential=False)
        
        # Push dataset
        logger.info(f"Pushing dataset to {repo_name}")
        hf_dataset.push_to_hub(repo_name)
        
        # Create URL and artifact
        url = f"https://huggingface.co/datasets/{repo_name}"
        logger.info(f"Successfully pushed dataset to {url}")

        create_markdown_artifact(
            key="hf-push-success",
            markdown=f"# ✅ Dataset Published\n\n**Repository:** [{repo_name}]({url})",
            description="HuggingFace publication success",
        )

        # Emit event
        emit_event(
            event="dataset-published",
            resource={"prefect.resource.id": f"ragas-pipeline.dataset.{repo_name}"},
            payload={"repo_name": repo_name, "url": url}
        )
        
        return repo_name
        
    except Exception as e:
        logger.error(f"Error pushing to HuggingFace: {str(e)}")
        create_markdown_artifact(
            key="hf-push-error",
            markdown=f"# ❌ HuggingFace Push Failed\n\n**Error:**\n```\n{str(e)}\n```\n\n**Repository:** {repo_name}",
            description="HuggingFace push error"
        )
        raise

# ------------------------------------------------------------------------------
# Main Flow
# ------------------------------------------------------------------------------

@flow(
    name="RAGAS Golden Dataset Pipeline", 
    description="Generates a RAG test dataset and knowledge graph from PDF documents",
    log_prints=True,
    version=os.environ.get("PIPELINE_VERSION", "1.0.0")
)
def ragas_pipeline(
    RAW_DIR: str = os.environ.get("RAW_DIR"),
    TESTSET_SIZE: int = int(os.environ.get("TESTSET_SIZE", "10")),
    KG_OUTPUT_PATH: str = os.environ.get("KG_OUTPUT_PATH", "output/kg.json"),
    PROCESSED_DIR: str = os.environ.get("PROCESSED_DIR", "output/"),
    HF_TESTSET_REPO_V2: str = os.environ.get("HF_TESTSET_REPO_V2", ""),
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", ""),
    HF_TOKEN: str = os.environ.get("HF_TOKEN", ""),
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE: str = os.environ.get("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "True"),
    PREFECT_API_URL: str = os.environ.get("PREFECT_API_URL", ""),
) -> Dict[str, Any]:
    """
    RAGAS Golden Dataset Pipeline that generates a test dataset and knowledge graph.
    
    Args:
        RAW_DIR: Directory with source documents
        TESTSET_SIZE: Number of test samples to generate
        KG_OUTPUT_PATH: File path for the serialized knowledge graph
        PROCESSED_DIR: Directory for output files
        HF_TESTSET_REPO_V2: HuggingFace repo name to push the dataset
        LLM_MODEL: LLM model to use for generation
        EMBEDDING_MODEL: Embedding model to use
        
    Returns:
        Dictionary with statistics about the pipeline run
    """
    logger = get_run_logger()
    logger.info("🚀 Starting RAGAS Golden Dataset Pipeline")

    try:
        # 1. Validate environment and input parameters
        validate_environment()
        validate_input_parameters(RAW_DIR, TESTSET_SIZE, KG_OUTPUT_PATH, PROCESSED_DIR)

        # 2. Record flow parameters
        params_table = [
            ["RAW_DIR", RAW_DIR],
            ["TESTSET_SIZE", TESTSET_SIZE],
            ["KG_OUTPUT_PATH", KG_OUTPUT_PATH],
            ["PROCESSED_DIR", PROCESSED_DIR],
            ["HF_TESTSET_REPO_V2", HF_TESTSET_REPO_V2 or "Not specified"],
            ["LLM_MODEL", LLM_MODEL],
            ["EMBEDDING_MODEL", EMBEDDING_MODEL],
            ["PREFECT_API_URL", PREFECT_API_URL or "Not specified"],
            ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", PREFECT_SERVER_ALLOW_EPHEMERAL_MODE],
        ]

        create_table_artifact(
            key="flow-parameters",
            table={"columns": ["Parameter", "Value"], "data": params_table},
            description="Pipeline flow parameters"
        )

        # 3. Submit PDF download task if needed (concurrent execution with other setup)
        docs_path_future = None
        if not any(Path(RAW_DIR).glob("*.pdf")):
            docs_path_future = download_pdfs.submit(RAW_DIR)
            logger.info(f"Submitting task to download PDFs to {RAW_DIR}")
        else:
            logger.info(f"Using existing PDFs in {RAW_DIR}")
            docs_path = RAW_DIR

        # 4. Resolve PDF path (wait for download if it was running)
        if docs_path_future:
            docs_path = docs_path_future.result()
            logger.info(f"PDF download completed to {docs_path}")

        # 5. Submit document loading task
        docs_future = load_documents.submit(docs_path)

        # 6. Get loaded documents and check if empty
        docs = docs_future.result()
        if not docs:
            logger.error("No documents loaded — aborting pipeline.")
            create_markdown_artifact(
                key="pipeline-error",
                markdown="# ❌ Pipeline Aborted\n\nNo documents were loaded. Cannot proceed with testset generation.",
                description="Pipeline error"
            )
            return {"status": "failed", "reason": "No documents loaded"}
            
        # 7. Submit document saving task in parallel with other tasks
        save_future = save_documents.submit(docs, PROCESSED_DIR)

        # 8. Submit testset generation task
        dataset_future = build_testset.submit(
            docs, TESTSET_SIZE, KG_OUTPUT_PATH, LLM_MODEL, EMBEDDING_MODEL
        )

        # 9. Get saved paths and testset results
        saved_paths = save_future.result()
        dataset = dataset_future.result()

        # 10. Push to Hugging Face if specified
        hf_url = None
        if HF_TESTSET_REPO_V2:
            # Submit HuggingFace preparation and push tasks
            hf_dataset_future = prepare_hf_dataset.submit(dataset)
            hf_dataset = hf_dataset_future.result()
            
            repo_name = push_to_hub(hf_dataset, HF_TESTSET_REPO_V2)
            hf_url = f"https://huggingface.co/datasets/{repo_name}"
        else:
            logger.info("No HF_TESTSET_REPO_V2 specified; skipping push.")

        # 11. Create final success artifact
        create_markdown_artifact(
            key="pipeline-summary",
            markdown=f"""# Pipeline Execution Summary

## Success! ✅

The RAGAS Golden Dataset Pipeline completed successfully.

## Outputs
- Documents processed: {len(docs)} pages
- Test set size: {TESTSET_SIZE}
- Knowledge graph: `{KG_OUTPUT_PATH}`
- Document JSONs: {', '.join([os.path.basename(p) for p in saved_paths])}
{f'- Published to: [{HF_TESTSET_REPO_V2}]({hf_url})' if HF_TESTSET_REPO_V2 else ''}

## Models Used
- LLM: {LLM_MODEL}
- Embeddings: {EMBEDDING_MODEL}
""",
            description="Pipeline execution summary"
        )

        logger.info("✅ Pipeline completed successfully.")
        
        # Return statistics dictionary
        return {
            "status": "success",
            "document_count": len(docs),
            "testset_size": TESTSET_SIZE,
            "kg_path": KG_OUTPUT_PATH,
            "saved_paths": saved_paths,
            "hf_repo": HF_TESTSET_REPO_V2 if HF_TESTSET_REPO_V2 else None
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        create_markdown_artifact(
            key="pipeline-failure",
            markdown=f"# ❌ Pipeline Failed\n\n**Error:**\n```\n{str(e)}\n```",
            description="Pipeline failure summary"
        )
        # Re-raise to ensure Prefect marks the flow as failed
        raise

if __name__ == "__main__":
    ragas_pipeline()
