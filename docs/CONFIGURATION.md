# Configuration Guide

This document provides detailed information about all configuration options for the RAGAS Golden Dataset Pipeline.

## Environment Variables

Configure the pipeline through environment variables in a `.env` file:

### Required Variables

- `OPENAI_API_KEY`: Your OpenAI API key

### Optional Variables

- **Hugging Face Configuration**:
  - `HF_TOKEN`: Your Hugging Face API token
  - `HF_TESTSET_REPO_V1`: Repository for main pipeline results
  - `HF_TESTSET_REPO_V2`: Repository for V2 pipeline results
  - `HF_DOCLOADER_REPO`: Repository for document loader pipeline results

- **LLM Configuration**:
  - `LLM_MODEL`: OpenAI model to use (default: "gpt-4.1-mini")
  - `EMBEDDING_MODEL`: Embedding model to use (default: "text-embedding-3-small")

- **Project Configuration**:
  - `TESTSET_SIZE`: Number of test samples to generate (default: 10)
  - `DOCS_PATH`: Directory with source documents (default: "data/")
  - `OUTPUT_DIR`: Directory for outputs (default: "output/")
  - `KG_OUTPUT_PATH`: Path for knowledge graph output (default: "output/kg.json")

- **DocLoader Configuration**:
  - `ARXIV_IDS`: Comma-separated list of arXiv IDs to fetch
  - `HTML_URLS`: Comma-separated list of HTML URLs to fetch

- **Prefect Settings**:
  - `PREFECT_API_URL`: URL for Prefect API
  - `PREFECT_SERVER_ALLOW_EPHEMERAL_MODE`: Enable ephemeral mode
  - `PREFECT_RESULTS_PERSIST_BY_DEFAULT`: Persist results for better caching

- **LangSmith Settings**:
  - `LANGSMITH_TRACING`: Enable LangSmith tracing (true/false)
  - `LANGSMITH_PROJECT`: Project name for LangSmith
  - `LANGSMITH_API_KEY`: Your LangSmith API key

## Example .env File

```
# Required
OPENAI_API_KEY=sk-xxx...

# Hugging Face
HF_TOKEN=hf_xxx...
HF_TESTSET_REPO_V1=your-username/ragas-golden-dataset
HF_TESTSET_REPO_V2=your-username/ragas-golden-dataset-v2
HF_DOCLOADER_REPO=your-username/ragas-golden-dataset-documents

# LLM Configuration
LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

# Project Settings
TESTSET_SIZE=10
DOCS_PATH=data/
OUTPUT_DIR=output/
KG_OUTPUT_PATH=output/kg.json

# DocLoader Settings
ARXIV_IDS=2303.08774,2304.03442
HTML_URLS=https://example.com/doc1,https://example.com/doc2

# Prefect Settings
PREFECT_API_URL=http://127.0.0.1:4200/api
PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=True
PREFECT_RESULTS_PERSIST_BY_DEFAULT=True

# LangSmith Settings
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ragas-golden-dataset
LANGSMITH_API_KEY=ls_xxx...
```

## Using Configuration Values

Configuration values can be accessed in the pipeline code using the `os.environ` dictionary or the `dotenv` package:

```python
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Access configuration values
openai_api_key = os.environ.get("OPENAI_API_KEY")
llm_model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")  # Default if not set
testset_size = int(os.environ.get("TESTSET_SIZE", 10))  # Convert to int
```

## Overriding Environment Variables

You can override environment variables using command-line arguments for the main pipeline:

```bash
python prefect_pipeline.py \
  --docs-path custom_data/ \
  --testset-size 20 \
  --kg-output custom_output/kg.json \
  --hf-repo your-username/custom-repo \
  --llm-model gpt-4o \
  --embedding-model text-embedding-3-large
``` 