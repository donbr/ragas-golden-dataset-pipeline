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
      - [Starting Prefect Server](#starting-prefect-server)
      - [Verify Server Connection](#verify-server-connection)
    - [Inspecting in Prefect UI](#inspecting-in-prefect-ui)
  - [Scheduling](#scheduling)
  - [Directory Structure](#directory-structure)
  - [License](#license)

---

## Features

* **Document Ingestion**: Load files of any supported format via LangChain's `DirectoryLoader`.
* **Testset Generation**: Leverage RAGAS `TestsetGenerator` with configurable LLM & embedding models.
* **Caching & Retries**: Built-in task-level caching (1 day) and retry policies for robustness.
* **Knowledge Graph Export**: Serialize the RAGAS knowledge graph to a JSON file.
* **Hugging Face Integration**: Push the generated dataset to your Hugging Face Hub repository.

---

## Prerequisites

* Python 3.8 or newer
* Prefect v3
* RAGAS, LangChain, OpenAI, and other Python libraries (see `requirements.txt`)
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

Set the following environment variables:

* `HF_TOKEN`: Your Hugging Face API token for dataset uploads.

Example:

```bash
export HF_TOKEN="hf_xxx..."
```

You may also load variables from a `.env` file using tools like `python-dotenv`.

---

## Usage

### Running Locally

To run the pipeline with default parameters:

```bash
python prefect_pipeline.py
```

You can customize the execution with command-line arguments:

```bash
python prefect_pipeline.py \
  --docs-path data/ \
  --testset-size 10 \
  --kg-output output/kg.json \
  --hf-repo your-username/ragas-golden-dataset
```

* `--docs-path`: Directory with source documents (default: "data/")
* `--testset-size`: Number of test samples to generate (default: 10)
* `--kg-output`: File path for the serialized knowledge graph (default: "output/kg.json")
* `--hf-repo`: HF Hub repository name to push the dataset (default: "your-username/ragas-golden")

### Using Prefect Server

By default, the pipeline will run with Prefect's ephemeral mode, which doesn't require a separate server process. However, for monitoring and managing workflows, you may want to use Prefect Server.

#### Starting Prefect Server

```bash
# Start the Prefect server in a separate terminal
prefect server start
```

This will start a local server at `http://127.0.0.1:4200`. You can access the UI by opening this URL in your browser.

#### Verify Server Connection

```bash
# Check if the server is running
prefect config view
```

You should see that `PREFECT_API_URL` is set to `http://127.0.0.1:4200/api` or similar.

If you want to use the ephemeral mode instead (no server needed), set this environment variable:

```bash
export PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=True
```

Or add it to your `.env` file:
```
PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=True
```

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
├── data/                  # Input documents
├── output/                # Generated outputs (KG JSON, logs)
├── prefect_pipeline.py    # Prefect v3 flow definition
├── requirements.txt       # Python dependencies
├── README.md              # This documentation
└── .gitignore
```

---

## License

This project is released under the [MIT License](LICENSE).
