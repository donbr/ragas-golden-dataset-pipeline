import os
import subprocess
import json
import pickle
from pathlib import Path
from uuid import uuid4
from datetime import timedelta

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

@task(name="validate-environment")
def validate_environment() -> None:
    logger = get_run_logger()
    logger.info("🔍 Validating environment variables...")

    required_vars_by_category = {
        "OpenAI": ["OPENAI_API_KEY"],
        "HuggingFace": ["HF_TOKEN", "HF_REPO"],
        "LLM Config": ["LLM_MODEL", "EMBEDDING_MODEL"],
        "Pipeline Config": ["DOCS_PATH", "TESTSET_SIZE", "KG_OUTPUT_PATH"],
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

@task(name="download-pdfs", retries=2, retry_delay_seconds=10)
def download_pdfs(docs_path: str) -> str:
    logger = get_run_logger()
    pdf_dir = Path(docs_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        ("https://arxiv.org/pdf/2505.10468.pdf", "ai_agents_vs_agentic_ai_2505.10468.pdf"),
        ("https://arxiv.org/pdf/2505.06913.pdf", "redteamllm_agentic_ai_framework_2505.06913.pdf"),
        ("https://arxiv.org/pdf/2505.06817.pdf", "control_plane_scalable_design_pattern_2505.06817.pdf"),
    ]

    success = failed = skipped = 0
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
        except Exception as e:
            logger.warning(f"curl failed ({e}), falling back to requests")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(url, verify=False)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            success += 1

    table = [
        ["Successful Downloads", success],
        ["Skipped (Already Existed)", skipped],
        ["Total PDFs", success + skipped],
    ]
    create_table_artifact(
        key="pdf-download-summary",
        table={"columns": ["Metric", "Value"], "data": table},
        description="PDF download summary"
    )

    emit_event(
        event="pdfs-downloaded",
        resource={"prefect.resource.id": f"ragas-pipeline.data.{docs_path}"},
        payload={"successful": success, "skipped": skipped},
    )
    return docs_path

@task(name="load-documents")
def load_documents(docs_path: str):
    logger = get_run_logger()
    logger.info(f"Loading documents from {docs_path}")
    loader = PyPDFDirectoryLoader(docs_path, glob="*.pdf", silent_errors=True)
    docs = loader.load()
    count = len(docs)
    logger.info(f"Loaded {count} pages")

    create_table_artifact(
        key="load-summary",
        table={"columns": ["Metric", "Value"], "data": [["Pages Loaded", count]]},
        description="Document load summary"
    )

    emit_event(
        event="documents-loaded",
        resource={"prefect.resource.id": f"ragas-pipeline.documents.{docs_path}"},
        payload={"document_count": count},
    )
    return docs

@task(name="save-documents")
def save_documents(docs, output_dir: str, formats=["pkl", "json"]):
    logger = get_run_logger()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    # Save as pickle if requested
    if "pkl" in formats:
        pkl_path = output_dir_path / "documents.pkl"
        logger.info(f"Saving {len(docs)} documents to {pkl_path} (pickle format)")
        with open(pkl_path, "wb") as f:
            pickle.dump(docs, f)
        saved_paths.append(pkl_path)
        
        create_link_artifact(
            key="documents-pkl",
            link=f"file://{os.path.abspath(pkl_path)}",
            link_text="Saved Documents (Pickle)",
            description="Link to the saved documents in pickle format"
        )
    
    # Save as JSON if requested
    if "json" in formats:
        json_path = output_dir_path / "documents.json"
        logger.info(f"Saving {len(docs)} documents to {json_path} (JSON format)")
        
        # Convert to a serializable format
        serializable_docs = []
        for doc in docs:
            serializable_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Save to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        saved_paths.append(json_path)
        
        create_link_artifact(
            key="documents-json",
            link=f"file://{os.path.abspath(json_path)}",
            link_text="Saved Documents (JSON)",
            description="Link to the saved documents in JSON format"
        )
    
    emit_event(
        event="documents-saved",
        resource={"prefect.resource.id": f"ragas-pipeline.documents.{output_dir}"},
        payload={"paths": [str(p) for p in saved_paths], "count": len(docs)},
    )
    
    logger.info(f"✅ Documents saved in formats: {', '.join(formats)}")
    return saved_paths

@task(name="build-testset", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
def build_testset(
    docs,
    testset_size: int,
    kg_output_path: str,
    llm_model: str,
    embedding_model: str
):
    logger = get_run_logger()
    logger.info(f"Building testset of size {testset_size} with {len(docs)} documents")

    llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    generator = TestsetGenerator(llm=llm, embedding_model=emb)

    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    kg = generator.knowledge_graph

    Path(kg_output_path).parent.mkdir(parents=True, exist_ok=True)
    kg.save(kg_output_path)
    logger.info(f"Knowledge graph saved to {kg_output_path}")

    create_link_artifact(
        key="kg-json",
        link=f"file://{os.path.abspath(kg_output_path)}",
        link_text="Knowledge Graph JSON File",
        description="Link to the generated knowledge graph"
    )

    emit_event(
        event="kg-generated",
        resource={"prefect.resource.id": f"ragas-pipeline.kg.{kg_output_path}"},
        payload={"path": kg_output_path},
    )
    return dataset

@task(name="push-to-hub")
def push_to_hub(dataset, hf_repo: str) -> str:
    logger = get_run_logger()
    token = os.environ["HF_TOKEN"]
    login(token=token, add_to_git_credential=False)
    hf_dataset = dataset.to_hf_dataset()
    hf_dataset.push_to_hub(hf_repo)
    url = f"https://huggingface.co/datasets/{hf_repo}"
    logger.info(f"Pushed dataset to {url}")

    create_markdown_artifact(
        key="hf-push",
        markdown=f"**Dataset published**: [{hf_repo}]({url})",
        description="Hugging Face repository link",
    )

    emit_event(
        event="dataset-published",
        resource={"prefect.resource.id": f"ragas-pipeline.dataset.{hf_repo}"},
        payload={"repo_name": hf_repo},
    )
    return hf_repo

# ------------------------------------------------------------------------------
# Main Flow
# ------------------------------------------------------------------------------

@flow(name="RAGAS Golden Dataset Pipeline", log_prints=False)
def ragas_pipeline(
    DOCS_PATH: str = os.environ.get("DOCS_PATH", "data/"),
    TESTSET_SIZE: int = int(os.environ.get("TESTSET_SIZE", "10")),
    KG_OUTPUT_PATH: str = os.environ.get("KG_OUTPUT_PATH", "output/kg.json"),
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output/"),
    HF_REPO: str = os.environ.get("HF_REPO", ""),
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", ""),
    HF_TOKEN: str = os.environ.get("HF_TOKEN", ""),
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE: str = os.environ.get("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "True"),
    PREFECT_API_URL: str = os.environ.get("PREFECT_API_URL", ""),
):
    logger = get_run_logger()
    logger.info("🚀 Starting RAGAS Golden Dataset Pipeline")

    # 1. Validate environment
    validate_environment()

    # 2. Record flow parameters
    params_table = [
        ["DOCS_PATH", DOCS_PATH],
        ["TESTSET_SIZE", TESTSET_SIZE],
        ["KG_OUTPUT_PATH", KG_OUTPUT_PATH],
        ["OUTPUT_DIR", OUTPUT_DIR],
        ["HF_REPO", HF_REPO or "Not specified"],
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

    # 3. Download PDFs if none present
    docs_path = DOCS_PATH
    if not any(Path(DOCS_PATH).glob("*.pdf")):
        docs_path = download_pdfs(DOCS_PATH)
    else:
        logger.info("Using existing PDFs in %s", DOCS_PATH)

    # 4. Load documents
    docs = load_documents(docs_path)
    if not docs:
        logger.error("No documents loaded — aborting pipeline.")
        return
        
    # 5. Save the documents to output directory
    save_documents(docs, OUTPUT_DIR)

    # 6. Build testset & knowledge graph
    dataset = build_testset(
        docs, TESTSET_SIZE, KG_OUTPUT_PATH, LLM_MODEL, EMBEDDING_MODEL
    )

    # 7. Push to Hugging Face if specified
    if HF_REPO:
        push_to_hub(dataset, HF_REPO)
    else:
        logger.info("No HF_REPO specified; skipping push.")

    logger.info("✅ Pipeline completed successfully.")

if __name__ == "__main__":
    ragas_pipeline()
