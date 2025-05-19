# RAGAS Golden Dataset Pipeline

This repository contains a Prefect v3 flow for generating a RAGAS testset and knowledge graph from a collection of documents, serializing the graph to JSON, and optionally pushing the testset to the Hugging Face Hub.

---

## Table of Contents

- [RAGAS Golden Dataset Pipeline](#ragas-golden-dataset-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Running Locally](#running-locally)
    - [Using Prefect Server](#using-prefect-server)
      - [Verify Server Connection](#verify-server-connection)
    - [Inspecting in Prefect UI](#inspecting-in-prefect-ui)
  - [Scheduling](#scheduling)
  - [Directory Structure](#directory-structure)
  - [License](#license)
- [Ragas Knowledge Graph Utilities](#ragas-knowledge-graph-utilities)
  - [Overview](#overview)
  - [Usage](#usage-1)
    - [Removing Embeddings from a Single File](#removing-embeddings-from-a-single-file)
    - [Processing All JSON Files in a Directory](#processing-all-json-files-in-a-directory)
    - [Checking JSON Structure](#checking-json-structure)
    - [Visualizing Knowledge Graphs](#visualizing-knowledge-graphs)
  - [Dependencies](#dependencies)
  - [File Size Reduction](#file-size-reduction)
  - [License](#license-1)

---

## Features

* **Automatic PDF Download**: Downloads sample research papers if no PDFs are present in the data directory.
* **PDF Document Processing**: Loads PDF files using LangChain's `PyPDFDirectoryLoader`.
* **Testset Generation**: Leverage RAGAS `TestsetGenerator` with configurable LLM & embedding models.
* **Caching & Retries**: Built-in task-level caching (1 day) and retry policies for robustness.
* **Knowledge Graph Export**: Serialize the RAGAS knowledge graph to a JSON file.
* **Hugging Face Integration**: Push the generated dataset to your Hugging Face Hub repository.
* **LangSmith Tracing**: Built-in LangSmith integration for monitoring and debugging LLM interactions.

---

## Prerequisites

* Python 3.8 or newer
* Prefect v3
* RAGAS, LangChain, OpenAI, and other Python libraries (see `requirements.txt`)
* An OpenAI API key for LLM and embedding access
* A LangSmith account and API key (optional, for tracing)
* A Hugging Face account and [API token](https://huggingface.co/settings/tokens) (if you plan to push to HF Hub)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ragas-golden-pipeline.git
cd ragas-golden-pipeline

# Create and activate a virtual environment
uv venv
source .venv/bin/activate    # On Windows: `.venv\Scripts\activate`

# Install dependencies
uv pip install -r requirements.txt
```

---

## Configuration

Set the following environment variables or create a `.env` file:

* `OPENAI_API_KEY`: Your OpenAI API key for model access.
* `HF_TOKEN`: Your Hugging Face API token for dataset uploads.
* `HF_REPO`: Your Hugging Face repository name (optional).
* `LANGSMITH_API_KEY`: Your LangSmith API key for tracing (optional).
* `LANGSMITH_TRACING`: Set to "true" to enable tracing (optional).
* `LANGSMITH_PROJECT`: Project name for LangSmith tracing (optional).

Example `.env` file:

```
OPENAI_API_KEY=sk-xxx...
HF_TOKEN=hf_xxx...
HF_REPO=your-username/ragas-golden-dataset
LANGSMITH_API_KEY=ls_xxx...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ragas-golden-dataset
```

See the `.env-example` file for a complete list of configurable options.

---

## Usage

### Running Locally

**Important**: Despite the PREFECT_SERVER_ALLOW_EPHEMERAL_MODE setting in the code and .env file, the Prefect server must be running before executing the pipeline locally.

```bash
# Start the Prefect server in a separate terminal
prefect server start
```

This will start a local server at `http://127.0.0.1:4200`. You can access the UI by opening this URL in your browser.

Once the server is running, you can execute the pipeline with default parameters:

```bash
python prefect_pipeline.py
```

You can customize the execution with command-line arguments:

```bash
python prefect_pipeline.py \
  --docs-path data/ \
  --testset-size 10 \
  --kg-output output/kg.json \
  --hf-repo your-username/ragas-golden-dataset \
  --llm-model gpt-4.1-mini \
  --embedding-model text-embedding-3-small
```

* `--docs-path`: Directory with source documents (default: "data/")
* `--testset-size`: Number of test samples to generate (default: 10)
* `--kg-output`: File path for the serialized knowledge graph (default: "output/kg.json")
* `--hf-repo`: HF Hub repository name to push the dataset (default: "")
* `--llm-model`: LLM model to use (default: "gpt-4.1-mini")
* `--embedding-model`: Embedding model to use (default: "text-embedding-3-small")

### Using Prefect Server

The pipeline requires a running Prefect server as described above. Here are additional details about the server configuration:

#### Verify Server Connection

```bash
# Check if the server is running
prefect config view
```

You should see that `PREFECT_API_URL` is set to `http://127.0.0.1:4200/api` or similar.

> **Note**: While the code and .env-example file include the `PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=True` setting, which should theoretically allow running without a server, our testing has found that starting the Prefect server is still required.

### Inspecting in Prefect UI

Once the Prefect server is running:

```bash
# Build and register a deployment
prefect deployment build prefect_pipeline.py:ragas_pipeline --name ragas-golden
prefect deployment apply ragas-golden-deployment.yaml

# Trigger a run
prefect deployment run ragas-pipeline/ragas-golden
```

---

## Scheduling

Schedule daily runs at 06:00 UTC using a cron-like schedule:

```bash
prefect deployment build prefect_pipeline.py:ragas_pipeline \
  --name daily-ragas \
  --cron "0 6 * * *" \
  --param docs_path=data/ \
  --param testset_size=10
prefect deployment apply daily-ragas-deployment.yaml
```

---

## Directory Structure

```
├── data/                  # Input documents (PDFs will be downloaded here if empty)
├── output/                # Generated outputs (KG JSON, logs)
├── prefect_pipeline.py    # Prefect v3 flow definition
├── requirements.txt       # Python dependencies
├── README.md              # This documentation
├── .env-example           # Example environment variable configuration
└── .gitignore
```

---

## License

This project is released under the [MIT License](LICENSE).

# Ragas Knowledge Graph Utilities

This repository contains utilities for working with Ragas knowledge graph files, including tools to remove embedding fields from JSON files and visualize knowledge graphs.

## Overview

The following utilities are available:

1. `remove_embeddings.py` - Removes embedding fields from a single JSON file
2. `process_all_json.py` - Processes all JSON files in a directory to remove embedding fields
3. `check_structure.py` - Analyzes a JSON file to identify embedding fields
4. `ragas_kg_visualization.py` - Visualizes a knowledge graph using Plotly
5. `simple_kg_visualization.py` - Creates a simplified visualization with fewer nodes for better clarity

## Usage

### Removing Embeddings from a Single File

```bash
python remove_embeddings.py <input_file> [output_file]
```

If no output file is specified, the script will save the result to `<input_file>_no_embeddings.json`.

### Processing All JSON Files in a Directory

```bash
python process_all_json.py <directory_path>
```

This will process all JSON files in the specified directory, skipping any files that already have "_no_embeddings" in their name.

### Checking JSON Structure

```bash
python check_structure.py <json_file>
```

This utility analyzes the structure of a JSON file and identifies any embedding fields present.

### Visualizing Knowledge Graphs

```bash
python ragas_kg_visualization.py
```

Creates an interactive visualization of the full knowledge graph and saves it as `ragas_kg_visualization.html`.

```bash
python simple_kg_visualization.py
```

Creates a simplified visualization with only 30 nodes for better clarity and saves it as `simple_kg_visualization.html`.

## Dependencies

This project uses the following dependencies:

- json (standard library)
- os (standard library)
- sys (standard library)
- glob (standard library)
- networkx
- plotly

You can install the required dependencies using:

```bash
pip install networkx plotly
```

Or with UV (recommended):

```bash
uv pip install networkx plotly
```

## File Size Reduction

Removing embedding fields can significantly reduce the size of knowledge graph JSON files, making them easier to work with and share.

## License

This project is open source and available under the MIT license.
