# RAGAS Golden Dataset Evaluation Pipeline

A modular, extensible pipeline for evaluating RAG (Retrieval Augmented Generation) retriever systems using Prefect for orchestration.

## Overview

This pipeline enables systematic evaluation of different retrieval strategies against test datasets. It provides features for:

- Evaluating multiple retriever implementations
- Calculating relevant performance metrics
- Generating comparative reports and visualizations
- Storing results as Prefect artifacts and local files
- Generating and using RAGAS TestSets for comprehensive evaluation

The architecture emphasizes modularity, allowing selective execution of pipeline components.

## Architecture

The pipeline consists of these key components:

- **Evaluation Core**: Environment validation and global configuration
- **Evaluation Data**: Test dataset loading and management
- **Evaluation Retrieval**: API server management and retriever evaluation
- **Evaluation Results**: Results storage, reporting, and visualization

```
libs/
├── evaluation_core/         # Core functionality
│   ├── __init__.py
│   └── config.py            # Global configuration
├── evaluation_data/         # Data management
│   ├── __init__.py
│   └── loaders.py           # Dataset loading
├── evaluation_retrieval/    # Retrieval evaluation
│   ├── __init__.py
│   ├── api.py               # API server management
│   └── metrics.py           # Metrics calculation
└── evaluation_results/      # Results handling
    ├── __init__.py
    ├── storage.py           # Results storage
    └── reporting.py         # Report generation
pipeline.py                  # Main orchestration flow
run.py                       # Example API server for testing
prefect_pipeline_v2.py       # TestSet generation pipeline
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ragas-golden-dataset-pipeline.git
cd ragas-golden-dataset-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the evaluation pipeline with default settings:

```bash
python pipeline.py naive bm25 contextual_compression multi_query parent_document ensemble semantic --test_dataset_path data/test_questions.json
```

### Command Line Arguments

- **positional arguments**: List of retriever types to evaluate
- `--test_dataset_path`: Path to local test dataset file (JSON)
- `--hf_dataset_repo`: HuggingFace dataset repository path
- `--dataset_size_limit`: Maximum number of examples to evaluate (0 for all)
- `--api_port`: Port for the API server
- `--output_dir`: Directory to store evaluation results
- `--llm_model`: Model to use for RAGAS evaluation metrics
- `--disable_components`: Components to disable (e.g., validation api_server)
- `--no_visualizations`: Disable visualizations in the report

### Dataset Format

The expected dataset format is:

```json
{
  "examples": [
    {
      "question": "What is RAG?",
      "context": ["Document about RAG architecture", "Another relevant document"],
      "answer": "RAG stands for Retrieval Augmented Generation..."
    },
    ...
  ]
}
```

## TestSet Generation and Evaluation Workflow

The pipeline supports a complete workflow from test dataset generation to evaluation:

### 1. Generate a Test Dataset with RAGAS

Use the TestSet generation pipeline to create a synthetic evaluation dataset:

```bash
python prefect_pipeline_v2.py --raw_dir data/raw --testset_size 20 --kg_output_path output/kg.json --processed_dir output/
```

This will:
- Load documents from the specified directory
- Generate a knowledge graph and test questions
- Save the test dataset to disk

#### Behind the Scenes: How RAGAS TestSet Generation Works

RAGAS generates test datasets using a knowledge graph-based approach:

1. **Knowledge Graph Creation**: Documents are processed to extract entities, relationships, and key information.
   ```python
   # Initialize knowledge graph
   from ragas.testset.graph import KnowledgeGraph
   kg = KnowledgeGraph()
   
   # Add documents to knowledge graph and apply transforms
   from ragas.testset.transforms import default_transforms, apply_transforms
   transforms = default_transforms(documents=docs, llm=llm, embedding_model=embedding_model)
   apply_transforms(kg, transforms)
   ```

2. **Query Distribution**: RAGAS generates different types of queries based on a distribution:
   ```python
   # Default query distribution
   from ragas.testset.synthesizers import default_query_distribution
   query_distribution = default_query_distribution(llm)
   # Includes single-hop specific (50%), multi-hop abstract (25%), and multi-hop specific (25%)
   ```

3. **TestSet Generation**: The generator uses the knowledge graph to create realistic questions, contexts, and answers:
   ```python
   from ragas.testset import TestsetGenerator
   from ragas.llms import LangchainLLMWrapper
   from ragas.embeddings import LangchainEmbeddingsWrapper
   
   # Initialize generator with LangChain components
   generator = TestsetGenerator(
       llm=LangchainLLMWrapper(llm_model),
       embedding_model=LangchainEmbeddingsWrapper(embedding_model)
   )
   
   # Generate testset
   testset = generator.generate_with_langchain_docs(docs, testset_size=20)
   ```

### 2. Use the Generated TestSet for Evaluation

The generated test dataset can be used directly with the evaluation pipeline:

```bash
python pipeline.py naive bm25 contextual_compression --test_dataset_path output/testset.json
```

### 3. End-to-End Workflow Example

Here's a complete workflow example:

```bash
# Step 1: Generate TestSet
python prefect_pipeline_v2.py --raw_dir data/pdfs --testset_size 25 --kg_output_path output/kg.json --processed_dir output/

# Step 2: Evaluate retrievers using the generated TestSet
python pipeline.py naive bm25 contextual_compression multi_query --test_dataset_path output/testset.json
```

The generated TestSet includes various question types:
- Single-hop specific questions (factual)
- Single-hop abstract questions (conceptual)
- Multi-hop specific questions (requiring multiple facts)
- Multi-hop abstract questions (requiring synthesis across documents)

This diverse question set provides a more comprehensive evaluation of retriever capabilities.

### 4. Using a HuggingFace-Hosted TestSet

If you've pushed your TestSet to HuggingFace:

```bash
python pipeline.py naive bm25 semantic --hf_dataset_repo dwb2023/ragas-golden-dataset-v2
```

### 5. Customizing TestSet Generation

You can customize the TestSet generation process:

#### Choosing Different LLM and Embedding Models

```python
# OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
generator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    azure_deployment="your-deployment-name",
    model="your-model-name"
))

# Any LangChain-supported LLM
generator_llm = LangchainLLMWrapper(your_llm_instance)
```

#### Adjusting Query Distribution

You can create a custom query distribution to focus on specific question types:

```python
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)

# Custom distribution with only specific queries
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.7),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.3)
]

testset = generator.generate(testset_size=20, query_distribution=query_distribution)
```

### Example: Evaluating Multiple Retrievers

```bash
python pipeline.py naive bm25 contextual_compression multi_query parent_document ensemble semantic --test_dataset_path data/eval_questions.json --dataset_size_limit 100
```

### Example: Using HuggingFace Dataset

```bash
python pipeline.py naive bm25 --hf_dataset_repo huggingface/squad --dataset_size_limit 50
```

### Example: Disabling Components

Run only the evaluation without starting the API server (if it's already running):

```bash
python pipeline.py bm25 --test_dataset_path data/eval_questions.json --disable_components api_server
```

## API Server

The pipeline includes an example API server (`run.py`) that simulates different retriever implementations for testing purposes. You can:

1. Use the included example server for testing:
   ```bash
   python run.py --port 8000
   ```

2. Use your own API server by disabling the API server component:
   ```bash
   python pipeline.py bm25 --test_dataset_path data/test_questions.json --disable_components api_server
   ```
   In this case, ensure your API server has compatible endpoints:
   - `GET /health`: Health check endpoint that returns a 200 status
   - `GET /retrievers`: Endpoint that returns a list of available retrievers
   - `POST /retrieve/{retriever_type}`: Retrieval endpoint that accepts a query and returns documents

## Converting RAGAS TestSet Format

The RAGAS TestSet generator outputs data in a specific format that can be converted for evaluation:

```python
# Manual conversion example
import json
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Initialize generator
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
generator = TestsetGenerator(llm=llm, embedding_model=emb)

# Generate testset (assuming docs is a list of langchain Documents)
testset = generator.generate_with_langchain_docs(docs, testset_size=20)

# Convert to evaluation format
eval_data = {
    "examples": []
}

for item in testset.test_data:
    eval_data["examples"].append({
        "question": item.question,
        "context": item.contexts if hasattr(item, "contexts") else [],
        "answer": item.ground_truth if hasattr(item, "ground_truth") else ""
    })

# Save converted data
with open("data/eval_questions.json", "w") as f:
    json.dump(eval_data, f, indent=2)
```

The evaluation pipeline also supports direct loading of RAGAS TestSet format with automatic conversion.

## References

### RAGAS Documentation
- [RAGAS TestSet Generation Guide](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/) - Comprehensive guide to generating TestSets for RAG evaluation
- [Single-Hop TestSet Generation](https://docs.ragas.io/en/stable/howtos/applications/singlehop_testset_gen/#testset-generation) - Guide to generating TestSets for single-hop queries
- [LangChain Integration](https://docs.ragas.io/en/stable/howtos/integrations/langchain/) - Documentation on using RAGAS with LangChain

### LangChain Resources
- [LangChain Document Transformers](https://github.com/langchain-ai/langchain/tree/master/docs/docs/integrations/document_transformers) - Documentation on document transformers in LangChain

## Extending the Pipeline

### Adding New Retrievers

New retrievers should be implemented in your API server and exposed through the `/retrieve/{retriever_type}` endpoint. The retriever type is passed as a parameter to identify which implementation to use.

### Adding New Metrics

To add custom metrics, modify the `calculate_metrics` function in `libs/evaluation_retrieval/metrics.py`.

### Adding RAGAS Metrics

RAGAS metrics are automatically included if the RAGAS package is installed. Ensure you have the necessary LLM API keys configured as environment variables.

## RAGAS Metrics Integration

The evaluation pipeline incorporates [RAGAS](https://docs.ragas.io/) metrics for comprehensive retrieval evaluation. These metrics provide deeper insights into retriever performance beyond traditional IR metrics.

### Available RAGAS Metrics

The pipeline supports these key RAGAS metrics:

- **LLMContextRecall**: Evaluates how well retrieved contexts align with expected information needs
- **Faithfulness**: Assesses whether retrieved information is accurate and reliable
- **FactualCorrectness**: Checks if retrieved content is factually correct compared to references
- **ResponseRelevancy**: Measures how relevant the retrievals are to the query
- **ContextEntityRecall**: Evaluates whether important entities are captured in retrieval results
- **NoiseSensitivity**: Tests retriever robustness against query variations and noise

### Implementing RAGAS Evaluation

The evaluation pipeline creates a RAGAS `EvaluationDataset` from retriever results:

```python
# Example from our implementation
dataset_items = []
for result in detailed_results:
    # Get question and retrieved contexts
    question = result.get("question", "")
    retrieved_contexts = [doc.get("content", "") for doc in result.get("documents", [])]
    
    dataset_items.append({
        "user_input": question,
        "retrieved_contexts": retrieved_contexts,
        "response": "",  # Would contain response if using a QA system
        "reference": ""  # Would contain ground truth if available
    })

# Create RAGAS evaluation dataset
evaluation_dataset = EvaluationDataset.from_list(dataset_items)

# Apply RAGAS metrics
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)
```

### Viewing RAGAS Results in Prefect

The evaluation results are displayed in Prefect as artifacts, making it easy to compare different retrievers:

1. Run the pipeline with RAGAS enabled
2. Open the Prefect UI
3. Navigate to the flow run
4. Check the artifacts tab to see detailed RAGAS metrics

For more details on RAGAS metrics, see the [RAGAS documentation](https://docs.ragas.io/en/stable/concepts/metrics/).

## Results and Artifacts

Results are stored in two formats:

1. **Local Files**: JSON, CSV, and PNG files in the output directory
2. **Prefect Artifacts**: Markdown reports, tables, and links viewable in the Prefect UI

## Required Dependencies

- Python 3.8+
- prefect
- pandas
- matplotlib
- requests
- fastapi (for the example API server)
- uvicorn (for the example API server)
- ragas (optional, for advanced metrics)
- datasets (optional, for HuggingFace dataset loading)
- langchain_openai (for TestSet generation)

## Serialization Pattern

The pipeline uses a specific pattern to handle serialization of complex objects in Prefect tasks:

1. Tasks that return complex objects use `result_serializer="json"` in their decorator to avoid pickle serialization issues
2. Non-JSON-serializable objects (like subprocess.Popen) are converted to simple dictionaries with relevant metadata
3. Table artifacts use the proper dictionary format with "columns" and "data" keys instead of pandas DataFrames
4. When needed, helper functions like `get_ragas_components()` are used to recreate objects from their serialized metadata

Example:

```python
@task(
    name="my-task",
    description="Task description",
    result_serializer="json"  # Use JSON serializer
)
def my_task():
    # Non-serializable object
    process = subprocess.Popen(...)
    
    # Return only serializable metadata
    return {
        "pid": process.pid,
        "status": "running"
    }
```

This approach ensures all task results can be properly cached and persisted by Prefect.

## License

[MIT License](LICENSE)
