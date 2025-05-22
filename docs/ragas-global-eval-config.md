# Leveraging Global Configuration for RAGAS Evaluations

## define global settings

- define settings once at the start of your notebook or script
- `evaluator_config` settings shown below will minimize impacts of LLM rate limiting and let you run a stronger evaluation model (such as `gpt-4o`)
- to make your comparisons more accurate you should use the same model for all evaluations.

Evaluations
    • naive retriever (baseline)
    • bm25 retriever
    • contextual compression retriever (re-ranker)
    • multi-query retriever
    • parent document retriever
    • ensemble retriever
    • semantic retriever


```python
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)

## set global evaluation settings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
evaluator_config = RunConfig(
    timeout=300,          # 5 minutes max for operations
    max_retries=15,       # More retries for rate limits
    max_wait=90,          # Longer wait between retries
    max_workers=8,        # Fewer concurrent API calls
    log_tenacity=True     # Log retry attempts
)
evaluator_metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()]
```

## Prepare Response Datasets

To evaluate different retriever strategies using RAGAS, we'll use a synthetic golden dataset that was specifically created for RAG evaluation. This dataset was generated using the RAGAS TestsetGenerator, which creates high-quality question-answer pairs with relevant contexts.

```python
import tempfile
import os
from datasets import load_dataset
from ragas import EvaluationDataset
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load the golden dataset from Hugging Face
golden_dataset = load_dataset("dwb2023/ragas-golden-dataset-v2", split="train")

print(f"Loaded golden dataset with {len(golden_dataset)} examples")
print(f"Dataset features: {golden_dataset.features}")
print(f"Sample question: {golden_dataset[0]['user_input']}")

# Step 2: Load your documents for retrieval
documents = []
loader = TextLoader("./data/source_documents.txt")
documents.extend(loader.load())

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# Step 4: Create embeddings and vector store
# Create a temporary directory for OpenAI embeddings
openai_dir = tempfile.mkdtemp(prefix="chroma_openai_")
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_model,
    persist_directory=openai_dir
)

# Step 5: Define a common prompt template for all retrievers
prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Step 6: Function to create evaluation dataset for a specific retriever
def create_response_dataset(retriever, retriever_name):
    # Create retrieval chain
    retrieval_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    
    # Create full chain
    chain = retrieval_chain | prompt | llm | StrOutputParser()
    
    # Extract test questions from the golden dataset
    test_questions = golden_dataset["user_input"]
    reference_contexts = golden_dataset["reference_contexts"]
    ground_truths = golden_dataset["reference"]
    
    # Generate answers and collect contexts
    answers = []
    contexts = []
    
    for question in test_questions:
        # Get answer from the chain
        answer = chain.invoke(question)
        answers.append(answer)
        
        # Get retrieved contexts
        retrieved_docs = retriever.get_relevant_documents(question)
        context_texts = [doc.page_content for doc in retrieved_docs]
        contexts.append(context_texts)
    
    # Create dataset dictionary
    dataset_dict = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    # Convert to RAGAS EvaluationDataset
    from ragas import EvaluationDataset
    
    # Create a dataset suitable for RAGAS evaluation
    evaluation_samples = []
    for i in range(len(test_questions)):
        sample = {
            "user_input": test_questions[i],
            "response": answers[i],
            "retrieved_contexts": contexts[i],
            "reference": ground_truths[i]
        }
        evaluation_samples.append(sample)
    
    evaluation_dataset = EvaluationDataset.from_list(evaluation_samples)
    
    return evaluation_dataset
```

Now you can create evaluation datasets for each retriever type:

```python
# Create naive retriever (basic similarity search)
naive_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
naive_retriever_response_dataset = create_response_dataset(naive_retriever, "naive")

# Create BM25 retriever
from langchain_community.retrievers import BM25Retriever
bm25_retriever = BM25Retriever.from_documents(splits, k=3)
bm25_retriever_response_dataset = create_response_dataset(bm25_retriever, "bm25")

# Create contextual compression retriever (using reranker)
# Import CohereRerank from the correct module
try:
    from langchain_cohere import CohereRerank
except ImportError:
    print("CohereRerank not available. Install with: pip install langchain-cohere")
    CohereRerank = None
    
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# Initialize the compressor if available
if CohereRerank is not None:
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=naive_retriever
    )
    contextual_compression_retriever_response_dataset = create_response_dataset(compression_retriever, "contextual_compression")
else:
    print("Skipping contextual compression retriever due to missing dependencies")

# Create multi-query retriever
from langchain.retrievers import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever,
    llm=llm
)
multi_query_retriever_response_dataset = create_response_dataset(multi_query_retriever, "multi_query")

# Create parent document retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.storage import InMemoryStore

# Function to batch documents
def batch_documents(documents, batch_size=100):
    """Split a list of documents into batches of specified size"""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

# Create a separate vectorstore for parent document retriever
parent_vs_dir = tempfile.mkdtemp(prefix="chroma_parent_")
parent_vs = Chroma(
    collection_name="parent_documents",
    embedding_function=embedding_model,
    persist_directory=parent_vs_dir
)

# Create document store
doc_store = InMemoryStore()

# Create parent document retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=parent_vs,
    docstore=doc_store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
)

# Add documents in smaller batches to avoid "batch size too large" error
print(f"Adding {len(splits)} documents to parent document retriever in batches")
for i, batch in enumerate(batch_documents(splits)):
    print(f"Adding batch {i+1} to parent document retriever")
    parent_retriever.add_documents(batch)

parent_document_retriever_response_dataset = create_response_dataset(parent_retriever, "parent_document")

# Create ensemble retriever
from langchain.retrievers import EnsembleRetriever

# Create ensemble of BM25 and vector retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[naive_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
ensemble_retriever_response_dataset = create_response_dataset(ensemble_retriever, "ensemble")

# Create semantic retriever (using a different embedding model)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create a separate directory for the HuggingFace embeddings
hf_dir = tempfile.mkdtemp(prefix="chroma_huggingface_")
semantic_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
semantic_vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=semantic_embedding,
    persist_directory=hf_dir
)
semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 3})
semantic_retriever_response_dataset = create_response_dataset(semantic_retriever, "semantic")
```

## Evaluate naive retriever

```python
naive_retriever_evaluation_results = evaluate(
    dataset=naive_retriever_response_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```

## Evaluate bm25 retriever

```python
bm25_retriever_evaluation_results = evaluate(
    dataset=bm25_retriever_response_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```

## Evaluate contextual compression retriever

```python
# Only run if CohereRerank is available
if CohereRerank is not None:
    contextual_compression_retriever_evaluation_results = evaluate(
        dataset=contextual_compression_retriever_response_dataset,
        metrics=evaluator_metrics,
        llm=evaluator_llm,
        run_config=evaluator_config
    )
```

## Evaluate multi query retriever

```python
multi_query_retriever_evaluation_results = evaluate(
    dataset=multi_query_retriever_response_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```

## Evaluate parent document retriever

```python
parent_document_retriever_evaluation_results = evaluate(
    dataset=parent_document_retriever_response_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```

## Evaluate ensemble retriever

```python
ensemble_retriever_evaluation_results = evaluate(
    dataset=ensemble_retriever_response_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```

## Evaluate semantic retriever

```python
semantic_retriever_evaluation_results = evaluate(
    dataset=semantic_retriever_response_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```
