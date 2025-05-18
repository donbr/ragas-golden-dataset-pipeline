import os
# Set Prefect to use ephemeral mode before importing Prefect
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "True"

from pathlib import Path
from typing import List, Tuple
from datetime import timedelta
import argparse
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from huggingface_hub import HfApi, login

# Load environment variables from .env file
load_dotenv()

@task
def load_documents(path: str) -> List:
    """
    Load documents from a directory using LangChain's DirectoryLoader.
    """
    loader = DirectoryLoader(path, glob="*.*")
    docs = loader.load()
    return docs


@task(
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1)
)
def build_testset(
    docs: List,
    size: int,
    llm_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small"
) -> Tuple[TestsetGenerator, object]:  # returns (generator, dataset)
    """
    Instantiate the RAGAS TestsetGenerator and build a testset.
    Cached for 1 day to avoid re-generation.
    """
    llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    generator = TestsetGenerator(llm=llm, embedding_model=emb)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=size)
    return generator, dataset


@task
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
    dataset.push_to_hub(repo_name)
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
      1. Load documents
      2. Generate testset & KG
      3. Save KG as JSON
      4. (Optional) Push testset to HF Hub
    """
    logger = get_run_logger()
    logger.info(f"Loading documents from %s", docs_path)
    docs = load_documents(docs_path)

    logger.info(f"Generating testset of size %d", testset_size)
    generator, dataset = build_testset(
        docs, 
        testset_size,
        llm_model=llm_model,
        embedding_model=embedding_model
    )

    logger.info(f"Saving knowledge graph to %s", knowledge_graph_path)
    save_knowledge_graph(generator, knowledge_graph_path)

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
