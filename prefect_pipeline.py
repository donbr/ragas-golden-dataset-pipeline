import os
# Set Prefect to use ephemeral mode before importing Prefect
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "True"
# Configure Prefect to persist results by default for better caching
os.environ["PREFECT_RESULTS_PERSIST_BY_DEFAULT"] = "true"

from pathlib import Path
from typing import List, Tuple, Dict
from datetime import timedelta
import argparse
import subprocess
import requests
from uuid import uuid4
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.cache_policies import NO_CACHE
from prefect.artifacts import create_markdown_artifact, create_link_artifact, create_table_artifact
from prefect.events import emit_event
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from huggingface_hub import HfApi, login

# Load environment variables from .env file
load_dotenv()

@task(
    name="validate-environment",
    description="Validates all required environment variables are set",
    retries=3,
    retry_delay_seconds=5,
    tags=["setup", "validation"]
)
def validate_environment() -> Dict[str, List[str]]:
    """
    Validates that all required environment variables are set.
    Returns a dictionary of environment categories and their values if successful,
    raises an EnvironmentError if critical variables are missing.
    """
    logger = get_run_logger()
    logger.info("Validating environment variables...")
    
    # Define required environment variables by category
    required_vars = {
        "OpenAI": ["OPENAI_API_KEY"],
        "HuggingFace": ["HF_TOKEN"] if os.environ.get("HF_TESTSET_REPO") else []
    }
    
    # Optional but recommended variables
    optional_vars = {
        "LangSmith": ["LANGSMITH_PROJECT", "LANGSMITH_TRACING"]
    }
    
    # Check for missing required variables
    missing_vars = {}
    for category, vars_list in required_vars.items():
        category_missing = [var for var in vars_list if var not in os.environ or not os.environ[var]]
        if category_missing:
            missing_vars[category] = category_missing
    
    # If any required variables are missing, log and raise error
    if missing_vars:
        error_msg = "Missing required environment variables:\n"
        for category, vars_list in missing_vars.items():
            error_msg += f"- {category}: {', '.join(vars_list)}\n"
        error_msg += "Please set these in your environment or .env file."
        
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    
    # Check for missing optional variables and log warnings
    missing_optional = {}
    for category, vars_list in optional_vars.items():
        category_missing = [var for var in vars_list if var not in os.environ or not os.environ[var]]
        if category_missing:
            missing_optional[category] = category_missing
            logger.warning(f"Optional {category} variables not set: {', '.join(category_missing)}")
            
    # Create a result dictionary with all variables that are set
    result = {}
    for category, vars_list in {**required_vars, **optional_vars}.items():
        available_vars = [var for var in vars_list if var in os.environ and os.environ[var]]
        if available_vars:
            # Just store the names, not the values (for security)
            result[category] = available_vars
    
    logger.info(f"Environment validation complete. Found {sum(len(v) for v in result.values())} variables across {len(result)} categories.")
    return result

@task(
    name="download-pdfs",
    description="Downloads sample PDFs for RAGAS testset generation",
    retries=3,
    retry_delay_seconds=10,
    tags=["data", "download"]
)
def download_pdfs(data_path: str) -> str:
    """
    Download sample PDFs for RAGAS testset generation.
    """
    logger = get_run_logger()
    pdf_dir = Path(data_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # PDF URLs and filenames
    pdf_urls = [
        ("https://arxiv.org/pdf/2505.10468.pdf", "ai_agents_vs_agentic_ai_2505.10468.pdf"),
        ("https://arxiv.org/pdf/2505.06913.pdf", "redteamllm_agentic_ai_framework_2505.06913.pdf"),
        ("https://arxiv.org/pdf/2505.06817.pdf", "control_plane_scalable_design_pattern_2505.06817.pdf")
    ]
    
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    
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
                "curl", "-L", "--ssl-no-revoke", url, "-o", str(output_path)
            ], check=True)
            logger.info(f"Successfully downloaded {filename} using curl")
            successful_downloads += 1
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback to requests if curl fails or isn't available
            logger.warning(f"Curl failed with error: {str(e)}. Using requests library as fallback")
            try:
                # Disable insecure request warnings when verify=False
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                response = requests.get(url, verify=False)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded {filename} using requests")
                successful_downloads += 1
            except Exception as e:
                logger.error(f"Failed to download {url}: {str(e)}")
                failed_downloads += 1
                continue
    
    # Create artifact with download summary
    create_table_artifact(
        key="pdf-download-summary",
        table={
            "columns": ["Metric", "Value"],
            "data": [
                ["Successful Downloads", successful_downloads],
                ["Failed Downloads", failed_downloads],
                ["Skipped (Already Existed)", skipped_downloads],
                ["Total PDFs", successful_downloads + skipped_downloads]
            ]
        },
        description="Summary of PDF download operations"
    )
    
    # Emit event about download completion
    emit_event(
        event="pdfs-downloaded",
        resource={"prefect.resource.id": f"ragas-pipeline.data.{data_path}"},
        payload={
            "successful": successful_downloads,
            "failed": failed_downloads,
            "skipped": skipped_downloads,
            "data_path": data_path
        }
    )
    
    return data_path

@task(
    name="load-documents",
    description="Loads PDF documents from a directory using LangChain's PyPDFDirectoryLoader",
    tags=["data", "loading"]
)
def load_documents(path: str) -> List:
    """
    Load PDF documents from a directory using LangChain's PyPDFDirectoryLoader.
    """
    logger = get_run_logger()
    logger.info(f"Loading documents from {path}...")
    
    try:
        loader = PyPDFDirectoryLoader(path, glob="*.pdf", silent_errors=True)
        docs = loader.load()
        num_docs = len(docs)
        
        if num_docs == 0:
            logger.warning(f"No documents were loaded from {path}. Check that valid PDFs exist in this directory.")
            return []
            
        logger.info(f"Successfully loaded {num_docs} pages across all PDFs")
        
        # Create artifact with document loading summary
        create_table_artifact(
            key="document-loading-summary",
            table={
                "columns": ["Metric", "Value"],
                "data": [
                    ["Documents Loaded", num_docs],
                    ["Source Directory", path]
                ]
            },
            description="Summary of document loading operation"
        )
        
        # Emit event about document loading
        emit_event(
            event="documents-loaded",
            resource={"prefect.resource.id": f"ragas-pipeline.documents.{path}"},
            payload={
                "document_count": num_docs,
                "source_path": path
            }
        )
        
        return docs
    except Exception as e:
        logger.error(f"Error loading documents from {path}: {str(e)}")
        raise

@task(
    name="build-testset",
    description="Builds RAGAS testset and generates knowledge graph",
    retries=3,
    retry_delay_seconds=10,
    cache_policy=NO_CACHE,
    tags=["ragas", "generation"]
)
def build_testset(
    docs: List,
    size: int,
    knowledge_graph_output_path: str,
    llm_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small"
) -> object:  # returns dataset
    """
    Instantiate the RAGAS TestsetGenerator, build a testset, and save the knowledge graph.
    No caching to ensure knowledge graph is always generated.
    """
    logger = get_run_logger()
    logger.info(f"Building testset of size {size} with {len(docs)} documents")
    
    kg_path = Path(knowledge_graph_output_path)
    
    # Check if output file exists before proceeding
    if kg_path.exists():
        # Create an artifact to document that the file already exists
        create_markdown_artifact(
            key="knowledge-graph-status",
            markdown=f"# Knowledge Graph File Found\nExisting knowledge graph found at `{knowledge_graph_output_path}`.\nRemoving file to regenerate.",
            description="Status of knowledge graph generation"
        )
        # Delete the existing file to ensure we regenerate
        kg_path.unlink()
        logger.info(f"Removed existing knowledge graph at {knowledge_graph_output_path}")
    
    try:
        # Initialize testset generator
        logger.debug(f"Initializing RAGAS TestsetGenerator with {llm_model} and {embedding_model}")
        llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
        generator = TestsetGenerator(llm=llm, embedding_model=emb)
        
        # Generate testset
        logger.info(f"Generating testset with {size} samples...")
        dataset = generator.generate_with_langchain_docs(docs, testset_size=size)
        logger.info(f"Successfully generated testset with {size} samples")
        
        # Save the knowledge graph
        kg = generator.knowledge_graph
        kg_path.parent.mkdir(parents=True, exist_ok=True)
        kg.save(str(kg_path))
        logger.info(f"Knowledge graph saved to {knowledge_graph_output_path}")
        
        # Create an artifact to document successful generation
        create_markdown_artifact(
            key="knowledge-graph-status",
            markdown=f"# Knowledge Graph Generated\nSuccessfully generated knowledge graph and saved to `{knowledge_graph_output_path}`.\n\n**Details:**\n- Test set size: {size}\n- Documents used: {len(docs)}\n- LLM model: {llm_model}\n- Embedding model: {embedding_model}",
            description="Status of knowledge graph generation"
        )
        
        # Create a link artifact to make it easy to access the file
        file_uri = f"file://{os.path.abspath(knowledge_graph_output_path)}"
        create_link_artifact(
            key="knowledge-graph-file",
            link=file_uri,
            link_text="Knowledge Graph JSON File",
            description="Link to the generated knowledge graph file"
        )
        
        # Emit event for successful generation
        emit_event(
            event="knowledge-graph-generated",
            resource={"prefect.resource.id": f"ragas-pipeline.knowledge-graph.{knowledge_graph_output_path}"},
            payload={
                "path": knowledge_graph_output_path,
                "testset_size": size,
                "documents_count": len(docs),
                "llm_model": llm_model,
                "embedding_model": embedding_model
            }
        )
        
        return dataset
    except Exception as e:
        logger.error(f"Error building testset: {str(e)}")
        
        # Create artifact for failed generation
        create_markdown_artifact(
            key="knowledge-graph-status",
            markdown=f"# ❌ Knowledge Graph Generation Failed\nFailed to generate knowledge graph at `{knowledge_graph_output_path}`.\n\n**Error:**\n```\n{str(e)}\n```\n\n**Parameters:**\n- Test set size: {size}\n- Documents used: {len(docs)}\n- LLM model: {llm_model}\n- Embedding model: {embedding_model}",
            description="Status of knowledge graph generation"
        )
        
        # Re-raise the exception
        raise

@task(
    name="push-to-hub",
    description="Pushes the generated testset to a Hugging Face repository",
    retries=2,
    retry_delay_seconds=30,
    tags=["huggingface", "publish"]
)
def push_to_hub(dataset: object, repo_name: str) -> str:
    """
    Push the generated testset to a Hugging Face repository.
    
    Note: When deployed with Prefect, the HF_TOKEN should be set
    as an environment variable or using Prefect secrets.
    """
    logger = get_run_logger()
    logger.info(f"Pushing dataset to Hugging Face repository: {repo_name}")
    
    try:
        # Prefect will automatically substitute the HF_TOKEN variable
        # when running as a deployment
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set")
            
        login(token=token, add_to_git_credential=False)
        hf_dataset = dataset.to_hf_dataset()  # Convert RAGAS dataset to HF dataset
        logger.info(f"Converted RAGAS dataset to Hugging Face dataset format")
        
        hf_dataset.push_to_hub(repo_name)
        logger.info(f"Successfully pushed dataset to {repo_name}")
        
        # Create artifact for successful push
        create_markdown_artifact(
            key="huggingface-push",
            markdown=f"# Dataset Published to Hugging Face\nSuccessfully pushed dataset to repository: [{repo_name}](https://huggingface.co/datasets/{repo_name})",
            description="Hugging Face repository information"
        )
        
        # Emit event for successful push
        emit_event(
            event="dataset-published",
            resource={"prefect.resource.id": f"ragas-pipeline.dataset.{repo_name}"},
            payload={
                "repo_name": repo_name,
                "platform": "huggingface"
            }
        )
        
        return repo_name
    except Exception as e:
        logger.error(f"Failed to push dataset to Hugging Face: {str(e)}")
        
        # Create artifact for failed push
        create_markdown_artifact(
            key="huggingface-push",
            markdown=f"# ❌ Failed to Publish Dataset\nFailed to push dataset to Hugging Face repository: {repo_name}\n\n**Error:**\n```\n{str(e)}\n```",
            description="Hugging Face repository error"
        )
        
        # Re-raise the exception
        raise

@flow(
    name="RAGAS Golden Dataset Pipeline",
    description="Generates a RAG test dataset and knowledge graph from PDF documents",
    log_prints=True,
    version=os.environ.get("PIPELINE_VERSION", "1.0.0")
)
def ragas_pipeline(
    docs_path: str = "data/",
    testset_size: int = 10,
    knowledge_graph_path: str = "output/kg.json",
    HF_TESTSET_REPO: str = "",
    llm_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small"
) -> None:
    """
    Orchestrates the full pipeline:
      1. Validate environment
      2. Download PDFs (if needed)
      3. Load documents
      4. Generate testset & KG
      5. Save KG as JSON
      6. (Optional) Push testset to HF Hub
    """
    logger = get_run_logger()
    logger.info(f"Starting RAGAS Golden Dataset Pipeline")
    logger.info(f"Parameters: docs_path={docs_path}, testset_size={testset_size}, knowledge_graph_path={knowledge_graph_path}")
    logger.info(f"Models: LLM={llm_model}, Embeddings={embedding_model}")
    
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
                ["Hugging Face Repo", HF_TESTSET_REPO or "Not specified"],
                ["LLM Model", llm_model],
                ["Embedding Model", embedding_model]
            ]
        },
        description="Pipeline configuration parameters"
    )
    
    # Download PDFs if directory is empty
    if not any(Path(docs_path).glob("*.pdf")):
        logger.info(f"No PDFs found in {docs_path}, downloading samples...")
        docs_path = download_pdfs(docs_path)
    else:
        logger.info(f"Found existing PDFs in {docs_path}, skipping download")
    
    logger.info(f"Loading documents from {docs_path}")
    docs = load_documents(docs_path)
    
    if not docs:
        logger.error(f"No documents were loaded from {docs_path}. Cannot proceed with testset generation.")
        return None

    logger.info(f"Generating testset of size {testset_size} and saving KG to {knowledge_graph_path}")
    dataset = build_testset(
        docs, 
        testset_size,
        knowledge_graph_path,
        llm_model=llm_model,
        embedding_model=embedding_model
    )

    if HF_TESTSET_REPO:
        logger.info(f"Pushing dataset to Hugging Face repo {HF_TESTSET_REPO}")
        push_to_hub(dataset, HF_TESTSET_REPO)
    else:
        logger.info("No Hugging Face repository specified, skipping push step")
    
    logger.info("RAGAS Golden Dataset Pipeline completed successfully")
    
    # Final success artifact
    create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"# Pipeline Execution Summary\n\n## Success! ✅\n\nThe RAGAS Golden Dataset Pipeline completed successfully.\n\n## Outputs\n- Knowledge Graph: `{knowledge_graph_path}`\n- Test Set Size: {testset_size}\n- Documents Processed: {len(docs)} pages\n{f'- Published to: [{HF_TESTSET_REPO}](https://huggingface.co/datasets/{HF_TESTSET_REPO})' if HF_TESTSET_REPO else ''}",
        description="Pipeline execution summary"
    )
    
    return dataset


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="RAGAS Golden Dataset Pipeline")
    
    # Required parameters with environment variable defaults
    parser.add_argument("--docs-path", type=str, 
                        default=os.environ.get("DOCS_PATH", "data/"),
                        help="Directory with source documents")
    parser.add_argument("--testset-size", type=int, 
                        default=int(os.environ.get("TESTSET_SIZE", "10")),
                        help="Number of test samples to generate")
    parser.add_argument("--kg-output", type=str, 
                        default=os.environ.get("KG_OUTPUT_PATH", "output/kg.json"),
                        help="File path for the serialized knowledge graph")
    parser.add_argument("--hf-repo", type=str, 
                        default=os.environ.get("HF_TESTSET_REPO", ""),
                        help="(Optional) HF Hub repository name to push the dataset")
    parser.add_argument("--llm-model", type=str,
                        default=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
                        help="LLM model to use for testset generation")
    parser.add_argument("--embedding-model", type=str,
                        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                        help="Embedding model to use for testset generation")
    
    args = parser.parse_args()
    
    print("Starting RAGAS Golden Dataset Pipeline with the following parameters:")
    print(f"- Documents Path: {args.docs_path}")
    print(f"- Testset Size: {args.testset_size}")
    print(f"- Knowledge Graph Output: {args.kg_output}")
    print(f"- Hugging Face Repo: {args.HF_TESTSET_REPO or 'Not specified'}")
    print(f"- LLM Model: {args.llm_model}")
    print(f"- Embedding Model: {args.embedding_model}")
    
    # Run the pipeline with parsed arguments
    ragas_pipeline(
        docs_path=args.docs_path,
        testset_size=args.testset_size,
        knowledge_graph_path=args.kg_output,
        HF_TESTSET_REPO=args.HF_TESTSET_REPO,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model
    )
