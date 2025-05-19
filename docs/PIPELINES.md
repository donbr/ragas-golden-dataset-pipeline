# Prefect Pipelines

This document provides detailed information about the different pipeline options available in the RAGAS Golden Dataset Pipeline.

## Pipeline Overview

The project offers three specialized Prefect v3 flows:

1. **Main Pipeline** (`prefect_pipeline.py`): The standard pipeline for generating RAGAS testsets and knowledge graphs
2. **V2 Pipeline** (`prefect_pipeline_v2.py`): Enhanced version with additional validation, monitoring, and error handling
3. **Document Loader Pipeline** (`prefect_docloader_pipeline.py`): Specialized for loading documents from multiple sources

## Pipeline Workflow Diagrams

### Main Pipeline Flow

```mermaid
graph TD
    A[Start] --> B[Load Documents]
    B --> C[Process Documents]
    C --> D[Generate Questions]
    D --> E[Generate Answers]
    E --> F[Construct Knowledge Graph]
    F --> G[Create Test Dataset]
    G --> H1[Save JSON Output]
    G --> H2[Save Knowledge Graph]
    
    subgraph "Optional Steps"
        H2 --> I[Push to HuggingFace]
        H2 --> J[Visualize Knowledge Graph]
    end
    
    classDef critical fill:#f96,stroke:#333,stroke-width:2px
    classDef optional fill:#69f,stroke:#333,stroke-width:1px
    
    class B,C,D,E,F,G,H1,H2 critical
    class I,J optional
```

### V2 Pipeline Flow

```mermaid
graph TD
    A[Start] --> B[Validate Environment]
    B --> C[Load Documents]
    C --> D[Validate Documents]
    D --> E[Process Documents]
    E --> F[Generate Questions]
    F --> G[Generate Answers]
    G --> H[Construct Knowledge Graph]
    H --> I[Create Test Dataset]
    I --> J1[Save JSON Output]
    I --> J2[Save PKL Output]
    I --> J3[Save Knowledge Graph]
    
    subgraph "Error Handling"
        B -- Failure --> B1[Log Error Details]
        C -- Failure --> C1[Download Sample Docs]
        D -- Failure --> D1[Report Invalid Documents]
    end
    
    subgraph "Optional Steps"
        J3 --> K[Push to HuggingFace]
        J3 --> L[Visualize Knowledge Graph]
    end
    
    classDef critical fill:#f96,stroke:#333,stroke-width:2px
    classDef optional fill:#69f,stroke:#333,stroke-width:1px
    classDef error fill:#f66,stroke:#333,stroke-width:1px
    
    class B,C,D,E,F,G,H,I,J1,J2,J3 critical
    class K,L optional
    class B1,C1,D1 error
```

### Document Loader Pipeline Flow

```mermaid
graph TD
    A[Start] --> B[Configure Sources]
    
    subgraph "Parallel Loading"
        B --> C1[Load PDF Documents]
        B --> C2[Fetch arXiv Papers]
        B --> C3[Scrape Web Content]
    end
    
    C1 --> D1[Process PDFs]
    C2 --> D2[Process arXiv]
    C3 --> D3[Process Web]
    
    D1 --> E[Merge Documents]
    D2 --> E
    D3 --> E
    
    E --> F[Validate Structure]
    F --> G[Save Merged Documents]
    G --> H[Push to HuggingFace]
    
    classDef parallel fill:#9f6,stroke:#333,stroke-width:2px
    classDef sequential fill:#f96,stroke:#333,stroke-width:2px
    classDef output fill:#69f,stroke:#333,stroke-width:1px
    
    class C1,C2,C3,D1,D2,D3 parallel
    class B,E,F,G sequential
    class H output
```

### Task Dependencies and Concurrency

```mermaid
flowchart TD
    subgraph "Sequential Tasks"
        A[Environment Setup] --> B[Document Loading]
        B --> C[Processing]
        C --> F[Knowledge Graph Creation]
        F --> G[Output Generation]
    end
    
    subgraph "Parallel Tasks"
        C --> D1[Question Generation Task 1]
        C --> D2[Question Generation Task 2]
        C --> D3[Question Generation Task 3]
        
        D1 --> E1[Answer Generation Task 1]
        D2 --> E2[Answer Generation Task 2]
        D3 --> E3[Answer Generation Task 3]
        
        E1 --> F
        E2 --> F
        E3 --> F
    end
    
    classDef sequential fill:#f96,stroke:#333,stroke-width:2px
    classDef parallel fill:#9f6,stroke:#333,stroke-width:2px
    
    class A,B,C,F,G sequential
    class D1,D2,D3,E1,E2,E3 parallel
```

## Main Pipeline

The standard pipeline for generating RAGAS testsets and knowledge graphs.

### Features

- Comprehensive error handling and validation
- Artifact creation for observability
- Concurrent task execution
- Well-documented code with proper type annotations

### Command-line Arguments

```bash
python prefect_pipeline.py \
  --docs-path data/ \
  --testset-size 10 \
  --kg-output output/kg.json \
  --hf-repo your-username/ragas-golden-dataset \
  --llm-model gpt-4.1-mini \
  --embedding-model text-embedding-3-small
```

### Available Arguments

- `--docs-path`: Directory with source documents (default: "data/")
- `--testset-size`: Number of test samples to generate (default: 10)
- `--kg-output`: File path for the knowledge graph (default: "output/kg.json")
- `--hf-repo`: HF Hub repository name (default: from env or empty)
- `--llm-model`: LLM model to use (default: from env or "gpt-4.1-mini")
- `--embedding-model`: Embedding model to use (default: from env or "text-embedding-3-small")

### Expected Output

When running the main pipeline successfully:

```bash
python prefect_pipeline.py
```

You should see output similar to:

```
Flow run 'crimson-sailfish' - Created by deployment 'ragas-pipeline/default'
Flow run 'crimson-sailfish' - ✅ Loading documents from data/ [3 documents found]
Flow run 'crimson-sailfish' - ✅ Generating RAGAS testset with 10 samples
Flow run 'crimson-sailfish' - ✅ Knowledge graph created with 87 nodes and 342 edges
Flow run 'crimson-sailfish' - ✅ Knowledge graph saved to output/kg.json
Flow run 'crimson-sailfish' - Completed successfully
```

## V2 Pipeline

Enhanced version with additional validation, monitoring, and error handling.

### Features

- Improved document saving with multiple format support
- Additional validation mechanisms
- More granular task structure with better retry configuration

This pipeline uses environment variables for configuration rather than command-line arguments.

### Expected Output

The V2 pipeline provides more detailed output with additional validation steps:

```
Flow run 'amber-moose' - Created by deployment 'ragas-pipeline-v2/default'
Flow run 'amber-moose' - ✅ Environment validated successfully
Flow run 'amber-moose' - ✅ Loading documents from data/ [3 documents found]
Flow run 'amber-moose' - ℹ️ Document validation in progress...
Flow run 'amber-moose' - ✅ All documents validated successfully
Flow run 'amber-moose' - ✅ Generating RAGAS testset with 10 samples
Flow run 'amber-moose' - ✅ Knowledge graph created with 92 nodes and 367 edges
Flow run 'amber-moose' - ✅ Documents saved to multiple formats in output/
Flow run 'amber-moose' - ✅ Knowledge graph saved to output/kg.json
Flow run 'amber-moose' - Completed successfully
```

## Document Loader Pipeline

Specialized for loading and comparing documents from multiple sources.

### Features

- Support for PDF files, arXiv papers, and web content
- Structure comparison and standardized output formats
- Metadata extraction and analysis

### Environment Variables

- `DOCS_PATH`: Directory containing PDF files (default: "data/")
- `HF_DOCLOADER_REPO`: HuggingFace repository for pushing the dataset (optional)
- `ARXIV_IDS`: Comma-separated list of arXiv IDs to fetch
- `HTML_URLS`: Comma-separated list of HTML URLs to fetch

## Using Prefect Server

### Starting the Server

```bash
# Start the Prefect server in a separate terminal
prefect server start
```

This starts a server at `http://127.0.0.1:4200` with a web UI.

### Verifying Server Connection

```bash
# Check if the server is running
prefect config view
```

You should see `PREFECT_API_URL` set to `http://127.0.0.1:4200/api` or similar.

> **Note**: Even with `PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=True`, our testing shows that starting a Prefect server is still required.

### Building and Registering Deployments

```bash
# Main pipeline
prefect deployment build prefect_pipeline.py:ragas_pipeline --name ragas-golden
prefect deployment apply ragas-golden-deployment.yaml

# V2 pipeline
prefect deployment build prefect_pipeline_v2.py:ragas_pipeline --name ragas-golden-v2
prefect deployment apply ragas-golden-v2-deployment.yaml

# Document loader pipeline
prefect deployment build prefect_docloader_pipeline.py:docloader_pipeline --name document-loader
prefect deployment apply document-loader-deployment.yaml

# Trigger a run
prefect deployment run ragas-pipeline/ragas-golden
```

## Scheduling

Schedule pipeline runs using Prefect's cron-like scheduling:

```bash
# Schedule daily runs at 06:00 UTC
prefect deployment build prefect_pipeline.py:ragas_pipeline \
  --name daily-ragas \
  --cron "0 6 * * *" \
  --param docs_path=data/ \
  --param testset_size=10
prefect deployment apply daily-ragas-deployment.yaml

# Start a worker to execute scheduled runs
prefect worker start -p default-agent-pool
```

## Troubleshooting

### Common Issues

1. **Prefect Server Connection Issues**:
   - Ensure the Prefect server is running with `prefect server start`
   - Verify connection with `prefect config view`
   - Check that `PREFECT_API_URL` is correctly set

2. **No PDFs in Data Directory**:
   - The pipeline will download sample research papers if no PDFs are present
   - Ensure your internet connection is active

3. **API Key Issues**:
   - Verify your OpenAI API key is valid and has sufficient credits
   - Check that environment variables are correctly set in `.env`

4. **Missing Dependencies**:
   - Run `uv pip install -r requirements.txt` (recommended) or `pip install -r requirements.txt` to ensure all dependencies are installed
   - Use a virtual environment to avoid conflicts

5. **HuggingFace Upload Failures**:
   - Verify your HF token has write permissions
   - Check that your repository exists or can be created

### Getting Help

- Check the Prefect [documentation](https://docs.prefect.io/latest/) for Prefect-specific issues
- Visit the RAGAS [GitHub repository](https://github.com/explodinggradients/ragas) for RAGAS-related questions
- Explore utility scripts in the `utilities/` directory for debugging and analysis 