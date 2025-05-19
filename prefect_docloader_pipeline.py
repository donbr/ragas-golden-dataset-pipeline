import os
import json
from pathlib import Path
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact, create_link_artifact
from prefect.events import emit_event
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import ArxivLoader, WebBaseLoader
from langchain_core.documents import Document
from huggingface_hub import login
from datasets import Dataset

import requests
import urllib3
import datetime

load_dotenv()

# --- Loader Tasks ---

@task(name="load-pdf-documents")
def load_pdf_documents(pdf_dir: str):
    logger = get_run_logger()
    loader = PyPDFDirectoryLoader(pdf_dir, glob="*.pdf", silent_errors=True)
    docs = loader.load()
    for doc in docs:
        doc.metadata["loader_type"] = "pdf"
    logger.info(f"Loaded {len(docs)} PDF pages from {pdf_dir}")
    return docs

@task(name="load-arxiv-metadata")
def load_arxiv_metadata(arxiv_ids: list):
    logger = get_run_logger()
    all_docs = []
    for arxiv_id in arxiv_ids:
        loader = ArxivLoader(query=arxiv_id)
        docs = loader.get_summaries_as_docs()
        for doc in docs:
            doc.metadata["loader_type"] = "arxiv"
            doc.metadata["arxiv_id"] = arxiv_id
        all_docs.extend(docs)
    logger.info(f"Loaded {len(all_docs)} arXiv metadata docs for {arxiv_ids}")
    return all_docs

@task(name="load-webbase-html")
def load_webbase_html(urls: list):
    logger = get_run_logger()
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["loader_type"] = "webbase"
            doc.metadata["source_url"] = url
        all_docs.extend(docs)
    logger.info(f"Loaded {len(all_docs)} HTML docs from {urls}")
    return all_docs

@task(name="save-docs-json")
def save_docs_json(docs, filename, output_dir="output"):
    logger = get_run_logger()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / filename
    serializable_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    def json_default(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2, default=json_default)
    logger.info(f"Saved {len(docs)} documents to {out_path}")
    # Sanitize key: remove extension, replace underscores with dashes
    sanitized_key = filename.rsplit('.', 1)[0].replace('_', '-')
    create_link_artifact(
        key=f"{sanitized_key}-json",
        link=f"file://{os.path.abspath(out_path)}",
        link_text=f"Saved {filename}",
        description=f"Link to the saved {filename} in JSON format"
    )
    return str(out_path)

# --- Save and Push to HuggingFace ---

@task(name="save-and-push-to-hf")
def save_and_push_to_hf(all_docs, HF_DOCLOADER_REPO: str):
    logger = get_run_logger()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in environment.")
    login(token=token, add_to_git_credential=False)

    def prune_and_serialize(meta: dict) -> str:
        clean = {}
        for k, v in meta.items():
            # drop empty
            if v in (None, "", [], {}):
                continue
            # normalize dates
            if isinstance(v, (datetime.date, datetime.datetime)):
                v = v.isoformat()
            clean[k] = v
        return json.dumps(clean, ensure_ascii=False)

    records = []
    for doc in all_docs:
        # support both dicts and Document-like objects
        pc   = doc.get("page_content") if isinstance(doc, dict) else getattr(doc, "page_content", "")
        meta = doc.get("metadata", {})   if isinstance(doc, dict) else getattr(doc, "metadata", {})
        records.append({
            "page_content": pc,
            "metadata_json": prune_and_serialize(meta)
        })

    ds = Dataset.from_list(records)
    ds.push_to_hub(HF_DOCLOADER_REPO)

    url = f"https://huggingface.co/datasets/{HF_DOCLOADER_REPO}"
    create_markdown_artifact(
        key="hf-push",
        markdown=f"**Dataset published**: [{HF_DOCLOADER_REPO}]({url})",
        description="Hugging Face repository link",
    )
    emit_event(
        event="dataset-published",
        resource={"prefect.resource.id": f"docloader-pipeline.dataset.{HF_DOCLOADER_REPO}"},
        payload={"repo_name": HF_DOCLOADER_REPO},
    )
    logger.info(f"Pushed dataset to {url}")
    return url

# --- Main Flow ---

@flow(name="Document Loader Structure Comparison Pipeline", log_prints=True)
def docloader_pipeline(
    PDF_DIR: str = os.environ.get("DOCS_PATH", "data/"),
    ARXIV_IDS: list = ["2505.10468", "2505.06913", "2505.06817"],
    HTML_URLS: list = [
        "https://arxiv.org/html/2505.10468v1",
        "https://arxiv.org/html/2505.06913v1",
        "https://arxiv.org/html/2505.06817v1",
    ],
    HF_DOCLOADER_REPO: str = os.environ.get("HF_DOCLOADER_REPO", ""),
):
    logger = get_run_logger()
    logger.info("Starting Document Loader Structure Comparison Pipeline")
    pdf_docs = load_pdf_documents(PDF_DIR)
    arxiv_docs = load_arxiv_metadata(ARXIV_IDS)
    webbase_docs = load_webbase_html(HTML_URLS)

    save_docs_json(pdf_docs, "pdf_docs.json")
    save_docs_json(arxiv_docs, "arxiv_docs.json")
    save_docs_json(webbase_docs, "webbase_docs.json")

    all_docs = pdf_docs + arxiv_docs + webbase_docs
    if HF_DOCLOADER_REPO:
        save_and_push_to_hf(all_docs, HF_DOCLOADER_REPO)
    else:
        logger.info("No HF_DOCLOADER_REPO specified; skipping push.")
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    docloader_pipeline()
