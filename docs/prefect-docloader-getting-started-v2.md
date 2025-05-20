# Document Loader Structure Comparison Pipeline: Getting Started

This guide provides a step-by-step introduction to the LangChain Document Loader Structure Comparison Pipeline, a specialized Prefect workflow that helps you compare document formats from different sources.

## What This Prefect Pipeline Actually Does

The Document Loader Structure Comparison Pipeline:

1. Loads documents from various sources leveraging [LangChain document loaders](https://python.langchain.com/docs/integrations/document_loaders/):

   * PDFs (using PyPDFDirectoryLoader)
   * arXiv papers (using ArxivLoader)
   * Web pages (using WebBaseLoader)

2. Processes document content and metadata from each source

3. Saves the documents in consistent JSON format for comparison

4. Analyzes differences in document structures across sources

5. Pushes the combined dataset to Hugging Face Hub

This pipeline is particularly useful for understanding the slight nuances in document loader behavior across different libraries and formats.

## Sequential Pipeline Workflow

The pipeline follows this workflow:

1. **Validation**: Checks environment variables and input parameters
2. **Loading**: Concurrently loads documents from all three sources
3. **Processing**: Normalizes document content and metadata
4. **Saving**: Saves each source's documents to separate JSON files
5. **Analysis**: Creates comparison artifacts in the Prefect UI
6. **Publishing**: Optionally pushes to Hugging Face Hub

## Prerequisites

* Python 3.11+
* Prefect 3.4.1+
* A Hugging Face account and User Access Token (see below for additional detail)
* The packages listed in `requirements.txt`, particularly:

  * `prefect`
  * `langchain` and related packages
  * `pypdf` (for PDF processing)
  * `arxiv` (for arXiv API access)
  * `pymupdf` (for PDF processing)
  * `huggingface_hub` (for pushing to Hugging Face)

## Setting Up Your Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ragas-golden-dataset-pipeline.git
   cd ragas-golden-dataset-pipeline
   ```

2. Create and activate a virtual environment:

   ```bash
   uv venv

   # On Windows:
   .venv\Scripts\activate

   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install dependencies using uv (for faster installation):

   ```bash
   # Install uv if you don't have it
   # follow the instructions here:  https://docs.astral.sh/uv/#installation

   # Install dependencies
   uv pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file based on `.env-example`:

```bash
cp .env-example .env
```

For basic usage, you only need to configure:

```ini
# Required for pushing to Hugging Face
HF_TOKEN=your_hugging_face_token_here
HF_DOCLOADER_REPO=your-username/document-loader-comparison
```

> **Note:** To get `HF_TOKEN`, sign up at [https://huggingface.co/join](https://huggingface.co/join) and create a "User Access Token" under **Settings → Access Tokens** ([Hugging Face Tokens Documentation](https://huggingface.co/docs/hub/security-tokens)).

This will run the pipeline based on the following defaults in the `.env` file:

```ini
# Required document sources
DOCS_PATH=data/
ARXIV_IDS=2505.10468,2505.06913,2505.06817
HTML_URLS=https://arxiv.org/html/2505.10468v1,https://arxiv.org/html/2505.06913v1,https://arxiv.org/html/2505.06817v1
```

> **Note:** Unlike the main RAGAS pipeline, the document loader pipeline does NOT require an OpenAI API key.

## Running the Pipeline

1. Start a Prefect server in a separate terminal:

   ```bash
   prefect server start
   ```

   This will start a server at [http://127.0.0.1:4200/](http://127.0.0.1:4200/)

2. Run the pipeline with default settings:

   ```bash
   python prefect_docloader_pipeline.py
   ```

## Understanding the Pipeline Output

The pipeline produces three main outputs:

### 1. JSON Files

Three JSON files are saved in the `output/` directory:

* `pdf_docs.json`: Documents loaded from PDFs
* `arxiv_docs.json`: Documents loaded from arXiv
* `webbase_docs.json`: Documents loaded from web pages

Each JSON file contains document content and metadata that you can analyze to understand structural differences.

### 2. Prefect Artifacts

The pipeline creates several artifacts in the Prefect UI:

* **Loading statistics**: Number of documents loaded from each source
* **Metadata field analysis**: Comparison of metadata fields across sources
* **Success/failure reports**: Detailed execution information

These artifacts help you understand differences in document structure across sources.

### 3. Hugging Face Dataset

If you configured a Hugging Face repository, the pipeline pushes a combined dataset containing all documents.

A [sample Hugging Face dataset](https://huggingface.co/datasets/dwb2023/ragas-golden-dataset-documents) created using this Prefect pipeline provides some great starter queries for understanding the structure differences between LangChain document loaders.

* [Loader Type Analysis](https://huggingface.co/datasets/dwb2023/ragas-golden-dataset-documents/viewer/default/train?views%5B%5D=train&sql=--+Compare+fields+included+in+metadata_json+for+each+loader_type%0AWITH+extracted_keys+AS+%28%0A++++SELECT+%0A++++++++json_extract_string%28metadata_json%2C+%27%24.loader_type%27%29+AS+loader_type%2C%0A++++++++json_keys%28metadata_json%29+AS+metadata_fields%2C%0A++++++++COUNT%28*%29+OVER+%28PARTITION+BY+json_extract_string%28metadata_json%2C+%27%24.loader_type%27%29%29+AS+record_count%0A++++FROM+%0A++++++++train%0A++++WHERE+%0A++++++++json_extract_string%28metadata_json%2C+%27%24.loader_type%27%29+IS+NOT+NULL%0A%29%2C%0Aflattened_fields+AS+%28%0A++++SELECT+%0A++++++++loader_type%2C%0A++++++++UNNEST%28metadata_fields%29+AS+field%2C%0A++++++++record_count%0A++++FROM+%0A++++++++extracted_keys%0A%29%0ASELECT+%0A++++loader_type%2C%0A++++COUNT%28DISTINCT+field%29+AS+distinct_field_count%2C%0A++++STRING_AGG%28DISTINCT+field%2C+%27%2C+%27%29+AS+all_fields%2C%0A++++MAX%28record_count%29+AS+record_count%0AFROM+%0A++++flattened_fields%0AGROUP+BY+%0A++++loader_type%0AORDER+BY+%0A++++distinct_field_count+DESC%3B)
* [Loader Type Metadata Fields](https://huggingface.co/datasets/dwb2023/ragas-golden-dataset-documents/viewer/default/train?views%5B%5D=train&sql=--+Compare+fields+included+in+metadata_json+for+each+loader_type%0AWITH+extracted_keys+AS+%28%0A++++SELECT+%0A++++++++json_extract_string%28metadata_json%2C+%27%24.loader_type%27%29+AS+loader_type%2C%0A++++++++json_keys%28metadata_json%29+AS+metadata_fields%2C%0A++++++++COUNT%28*%29+OVER+%28PARTITION+BY+json_extract_string%28metadata_json%2C+%27%24.loader_type%27%29%29+AS+record_count%0A++++FROM+%0A++++++++train%0A++++WHERE+%0A++++++++json_extract_string%28metadata_json%2C+%27%24.loader_type%27%29+IS+NOT+NULL%0A%29%2C%0Aflattened_fields+AS+%28%0A++++SELECT+%0A++++++++loader_type%2C%0A++++++++UNNEST%28metadata_fields%29+AS+field%2C%0A++++++++record_count%0A++++FROM+%0A++++++++extracted_keys%0A%29%0ASELECT+%0A++++loader_type%2C%0A++++COUNT%28DISTINCT+field%29+AS+distinct_field_count%2C%0A++++STRING_AGG%28DISTINCT+field%2C+%27%2C+%27%29+AS+all_fields%2C%0A++++MAX%28record_count%29+AS+record_count%0AFROM+%0A++++flattened_fields%0AGROUP+BY+%0A++++loader_type%0AORDER+BY+%0A++++distinct_field_count+DESC%3B)

## Analyzing Document Structures

To compare document structures across sources:

1. Examine the JSON files to see differences in content and metadata
2. Look at the Prefect UI artifacts, particularly the metadata field analysis
3. Note how different loaders handle the same content differently
4. Identify which format preserves information most effectively

## Troubleshooting

### Common Issues

| Issue              | Solution                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------ |
| No PDF files found | Place PDF files in the specified directory. The pipeline will continue with other sources. |
| Invalid arXiv IDs | Ensure IDs follow proper format (e.g., "2505.10468") and are publicly available. |
| Invalid HTML URLs | Verify URLs start with "http" and are accessible.                                          |
| HF_TOKEN missing | For pushing to Hugging Face add your token to the `.env` file. |
| Package errors | Ensure all dependencies are installed with `uv pip install -r requirements.txt`. |
| Connection errors  | Check that the Prefect server is running with `prefect server start`. |

### Checking Logs

If the pipeline fails, check:

1. The console output for error messages
2. The Prefect UI for detailed task logs
3. The artifacts in the Prefect UI for validation information

## Next Steps

* Improve the quality of the Hugging Face dataset metadata
* Customize the pipeline for your specific document analysis needs
* Connect the output to other document processing pipelines
* Integrate with your own RAG workflows

## Additional Resources

### uv Package Manager

The guide recommends using uv for faster dependency installation. Learn more about uv:

* [uv Documentation](https://docs.astral.sh/uv/) - Comprehensive guide to uv, a fast Python package installer
* [uv Installation](https://docs.astral.sh/uv/installation/) - Multiple ways to install uv on your system

### Hugging Face Resources

For working with Hugging Face datasets and authentication:

* [Hugging Face User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens) - Learn how to create and manage your HF\_TOKEN
* [Hugging Face Datasets Quickstart](https://huggingface.co/docs/datasets/en/quickstart) - Guide to working with datasets
* [Creating Datasets](https://huggingface.co/docs/datasets/en/create_dataset) - Learn how to create and customize your own datasets
* [Python Library Quick Start](https://huggingface.co/docs/huggingface_hub/en/quick-start) - Guide to using the huggingface\_hub Python library

For detailed information on configuration options and pipeline architecture, see:

* [CONFIGURATION.md](./CONFIGURATION.md)
* [PIPELINES.md](./PIPELINES.md)
