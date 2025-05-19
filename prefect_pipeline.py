import os
# Set Prefect to use ephemeral mode before importing Prefect
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "True"

from pathlib import Path
from typing import List, Tuple
from datetime import timedelta
import argparse
import subprocess
import requests
from uuid import uuid4
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.cache_policies import NO_CACHE
from prefect.artifacts import create_markdown_artifact, create_link_artifact
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from huggingface_hub import HfApi, login

# Load environment variables from .env file
load_dotenv()

# Validate required LangSmith environment variables
# These are prerequisites and should fail if not present
def validate_langsmith_env():
    """Validate that required LangSmith environment variables are set"""
    required_vars = ["LANGSMITH_PROJECT", "LANGSMITH_TRACING"]
    missing_vars = [var for var in required_vars if var not in os.environ or not os.environ[var]]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required LangSmith environment variables: {', '.join(missing_vars)}. "
            f"Please set these in your environment or .env file."
        )

# Validate environment at startup
validate_langsmith_env()

@task
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
    
    for url, filename in pdf_urls:
        output_path = pdf_dir / filename
        
        # Skip download if file already exists
        if output_path.exists():
            logger.info(f"File {filename} already exists, skipping download")
            continue
            
        logger.info(f"Downloading {url} to {output_path}")
        
        # Try using curl first (with Windows compatibility)
        try:
            subprocess.run([
                "curl", "-L", "--ssl-no-revoke", url, "-o", str(output_path)
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to requests if curl fails or isn't available
            logger.info("Curl failed, using requests library as fallback")
            try:
                # Disable insecure request warnings when verify=False
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                response = requests.get(url, verify=False)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                continue
    
    return data_path

@task
def load_documents(path: str) -> List:
    """
    Load PDF documents from a directory using LangChain's PyPDFDirectoryLoader.
    """
    logger = get_run_logger()
    loader = PyPDFDirectoryLoader(path, glob="*.pdf", silent_errors=True)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages across all PDFs")
    return docs

@task(
    retries=3,
    retry_delay_seconds=10,
    cache_policy=NO_CACHE
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
    
    llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    generator = TestsetGenerator(llm=llm, embedding_model=emb)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=size)
    
    # Save the knowledge graph
    kg = generator.knowledge_graph
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    kg.save(str(kg_path))
    logger.info(f"Knowledge graph saved to {knowledge_graph_output_path} from build_testset task")
    
    # Create an artifact to document successful generation
    create_markdown_artifact(
        key="knowledge-graph-status",
        markdown=f"# Knowledge Graph Generated\nSuccessfully generated knowledge graph and saved to `{knowledge_graph_output_path}`.\n\nTest set size: {size}\nLLM model: {llm_model}\nEmbedding model: {embedding_model}",
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
    
    return dataset

@task(cache_policy=NO_CACHE)
def save_knowledge_graph(
    generator: TestsetGenerator,
    output_path: str
) -> str:
    """
    Serialize the RAGAS knowledge graph to JSON.
    """
    kg = generator.knowledge_graph
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kg.save(str(out))
    return str(out)

@task
def push_to_hub(dataset: object, repo_name: str) -> str:
    """
    Push the generated testset to a Hugging Face repository.
    
    Note: When deployed with Prefect, the HF_TOKEN should be set
    as an environment variable or using Prefect secrets.
    """
    # Prefect will automatically substitute the HF_TOKEN variable
    # when running as a deployment
    login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)
    hf_dataset = dataset.to_hf_dataset()  # Convert RAGAS dataset to HF dataset
    hf_dataset.push_to_hub(repo_name)
    return repo_name

@flow(name="RAGAS Golden Dataset Pipeline")
def ragas_pipeline(
    docs_path: str = "data/",
    testset_size: int = 10,
    knowledge_graph_path: str = "output/kg.json",
    hf_repo: str = "",
    llm_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small"
) -> None:
    """
    Orchestrates the full pipeline:
      1. Download PDFs (if needed)
      2. Load documents
      3. Generate testset & KG
      4. Save KG as JSON
      5. (Optional) Push testset to HF Hub
    """
    logger = get_run_logger()
    
    # Download PDFs if directory is empty
    if not any(Path(docs_path).glob("*.pdf")):
        logger.info(f"No PDFs found in {docs_path}, downloading samples...")
        docs_path = download_pdfs(docs_path)
    
    logger.info(f"Loading documents from %s", docs_path)
    docs = load_documents(docs_path)

    logger.info(f"Generating testset of size %d and saving KG to %s", testset_size, knowledge_graph_path)
    dataset = build_testset(
        docs, 
        testset_size,
        knowledge_graph_path,
        llm_model=llm_model,
        embedding_model=embedding_model
    )

    if hf_repo:
        logger.info(f"Pushing dataset to Hugging Face repo %s", hf_repo)
        push_to_hub(dataset, hf_repo)


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
                        default=os.environ.get("HF_REPO", ""),
                        help="(Optional) HF Hub repository name to push the dataset")
    parser.add_argument("--llm-model", type=str,
                        default=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
                        help="LLM model to use for testset generation")
    parser.add_argument("--embedding-model", type=str,
                        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                        help="Embedding model to use for testset generation")
    
    args = parser.parse_args()
    
    # Run the pipeline with parsed arguments
    ragas_pipeline(
        docs_path=args.docs_path,
        testset_size=args.testset_size,
        knowledge_graph_path=args.kg_output,
        hf_repo=args.hf_repo,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model
    )
