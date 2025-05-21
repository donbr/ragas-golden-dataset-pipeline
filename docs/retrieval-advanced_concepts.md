# Advanced RAG Techniques and Optimizations

This document covers cutting-edge techniques, optimizations, and advanced implementations for Retrieval Augmented Generation (RAG) systems. It's designed for those who already understand the [basic RAG concepts](./concepts.md) and want to explore more sophisticated approaches.

> 📚 **Learning Path**: This document is part of a three-part learning progression:
>
> 1. [README.md](./README.md): Practical guide for setup and usage
> 2. [concepts.md](./concepts.md): Core RAG concepts and component explanations
> 3. **advanced_concepts.md (this file)**: Advanced techniques and optimizations

> **Last Updated:** May 2025 - Reflecting current state of RAG technology and practices.

## 1. Advanced Retrieval Techniques

### Hypothetical Document Embeddings (HyDE)

HyDE is an approach that uses the LLM to generate a hypothetical document that would answer the query, then uses that document's embedding (rather than the query's embedding) for retrieval.

```mermaid
graph LR
    A[User Query] --> B[LLM]
    B --> C[Hypothetical Document]
    C --> D[Embedding Model]
    D --> E[Document Vector]
    E --> F[Vector Search]
    F --> G[Similar Real Documents]
    G --> H[Context]
    H --> I[Final LLM]
    A --> I
    I --> J[Answer]
```

**Why It Works**: Query embeddings often lack the specificity and context needed for optimal retrieval. By having the LLM generate a hypothetical answer first, we get a richer representation containing the kinds of details and language that would likely appear in a good source document.

**Sample Implementation**:

```python
from langchain.retrievers import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI

# Base retriever
base_retriever = vectorstore.as_retriever()

# LLM for generating hypothetical documents
llm = ChatOpenAI(temperature=0)

# Create the HyDE retriever
hyde_retriever = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_retriever=base_retriever,
    prompt_template="""Given this question, generate a hypothetical document that would answer it:
    Question: {question}
    Hypothetical document:"""
)

# Using the HyDE retriever
documents = hyde_retriever.get_relevant_documents(
    "What are the rules of the Continental Hotel in John Wick?"
)
```

### Query Transformation

Query transformation involves modifying the original query to improve retrieval performance. This can include:

1. **Query Expansion**: Adding related terms or synonyms
2. **Query Rewriting**: Rephrasing for clarity or completeness
3. **Query Decomposition**: Breaking complex queries into simpler sub-queries

**Implementation Example**:

```python
from langchain.retrievers.query_transformation import PromptQueryTransformation
from langchain.schema.runnable import RunnablePassthrough

# Create a query transformer
query_transformer = PromptQueryTransformation(
    llm=ChatOpenAI(),
    template="""You are an expert at transforming user questions into the most effective search queries. 
    Transform the following user question into a search query that will help find the most relevant documents:
    User question: {query}
    Search query:"""
)

# Create the transformed retrieval chain
transformed_retrieval_chain = (
    {"query": RunnablePassthrough()} 
    | {"transformed_query": query_transformer, "original_query": lambda x: x["query"]}
    | {"documents": lambda x: vectorstore.similarity_search(x["transformed_query"]), "query": lambda x: x["original_query"]}
)

# Using the transformed retrieval chain
result = transformed_retrieval_chain.invoke({"query": "Why is John Wick so angry about his dog?"})
```

### Self-Query Retriever

Self-query retrievers use an LLM to parse a natural language query into both a search query and structured filters, enabling more precise retrieval.

```python
from langchain.retrievers.self_query import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define metadata fields and their types
metadata_field_info = [
    AttributeInfo(
        name="Movie_Title",
        description="The title of the John Wick movie being reviewed",
        type="string",
    ),
    AttributeInfo(
        name="Rating",
        description="The numerical rating given to the movie (1-10)",
        type="integer",
    ),
]

# Create the self-query retriever
self_querying_retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(),
    vectorstore=vectorstore,
    document_contents="John Wick movie reviews",
    metadata_field_info=metadata_field_info,
)

# Example query combining search and filters
documents = self_querying_retriever.get_relevant_documents(
    "What do reviews with ratings above 8 say about John Wick 3's action scenes?"
)
```

### GraphRAG

GraphRAG is a powerful approach that emerged in 2024-2025 which leverages graph-based knowledge representations for more sophisticated reasoning and retrieval. Instead of treating documents as independent units, GraphRAG models the relationships between entities, enabling multi-hop reasoning and more complex information synthesis.

```mermaid
graph TD
    A[User Query] --> B[Query Analysis]
    B --> C[Entity Extraction]
    C --> D[Graph Navigation]
    D --> E[Subgraph Retrieval]
    E --> F[Context Generation]
    F --> G[LLM Response]
    H[(Knowledge Graph)] --> D
```

**Why It Works**: Traditional RAG struggles with questions requiring connections across multiple documents or reasoning chains. GraphRAG excels at answering questions that involve relationships between entities, cause-effect reasoning, or multi-step processes.

**Sample Implementation**:

```python
from langchain.retrievers import KnowledgeGraphRetriever
from langchain_community.graphs import Neo4jGraph

# Connect to a knowledge graph database
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Create a GraphRAG retriever
graph_retriever = KnowledgeGraphRetriever(
    graph=graph,
    query_template="""
    Given the question: {query}
    Generate a Cypher query to retrieve relevant information from the knowledge graph.
    """,
    llm=ChatOpenAI(model="gpt-4o")
)

# Get graph-based context
results = graph_retriever.get_relevant_documents("What relationships exist between John Wick and the Continental Hotel management?")
```

### MiniRAG and Lightweight RAG

Developed in response to the computational demands of traditional RAG, MiniRAG represents a family of techniques for implementing RAG in resource-constrained environments. These approaches focus on optimizing each component of the RAG pipeline for efficiency.

**Key Techniques**:

1. **Quantized embeddings**: Using 8-bit or even 4-bit quantization for embedding vectors to reduce memory footprint
2. **Sparse retrieval optimization**: Implementing more efficient retrieval algorithms with minimal accuracy loss
3. **Context distillation**: Pre-processing documents to extract only essential information
4. **Caching and precomputation**: Strategic caching of common queries and precomputed retrievals

**Implementation Example**:

```python
from langchain.retrievers import CompactRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Use a smaller, optimized embedding model
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    quantize=True  # Enable quantized embeddings
)

# Create a memory-efficient vector store
compact_vectorstore = CompactVectorStore.from_documents(
    documents,
    embeddings,
    index_type="hnsw",  # Hierarchical Navigable Small World for faster retrieval
    compression_level=4  # Higher compression for smaller index size
)

# Create a lightweight retriever
compact_retriever = CompactRetriever(
    vectorstore=compact_vectorstore,
    search_kwargs={"k": 3},
    cache_strategy="lru"  # Least Recently Used caching for common queries
)

# Use the lightweight retriever
results = compact_retriever.get_relevant_documents("What are the main themes in John Wick?")
```

## 2. Advanced Vector Search Techniques

### Hybrid Search

Hybrid search combines the strengths of sparse retrieval (keyword-based like BM25) and dense retrieval (embedding-based semantic search).

```mermaid
graph TD
    A[User Query] --> B[BM25 Search]
    A --> C[Vector Search]
    B --> D[Keyword Results]
    C --> E[Semantic Results]
    D --> F[Result Fusion]
    E --> F
    F --> G[Combined Results]
```

**Implementation**:

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.ensemble import WeightedEnsembleRetriever

# Create the keyword retriever
bm25_retriever = BM25Retriever.from_documents(documents)

# Create the semantic retriever
vector_retriever = vectorstore.as_retriever()

# Create the weighted ensemble hybrid retriever
hybrid_retriever = WeightedEnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # Weights should sum to 1
    verbose=True
)

# Get results
hybrid_docs = hybrid_retriever.get_relevant_documents("Continental Hotel gold coins")
```

### Cross-Encoder Reranking

Cross-encoders analyze query and document pairs together rather than separately, providing more accurate relevance scoring but at higher computational cost. They're often used to rerank an initial set of retrieved documents.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers.document_compressors import CohereRerank
from langchain_openai import OpenAIEmbeddings

# Create base retriever that gets a larger initial set
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Create the cross-encoder reranker
reranker = CohereRerank()

# Create the compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    compressor=reranker,
    top_k=5  # Return top 5 after reranking
)

# Get more precisely ranked results
documents = compression_retriever.get_relevant_documents("Who is Winston in John Wick?")
```

### Vector Search with Maximum Marginal Relevance (MMR)

MMR balances relevance with diversity in the results, ensuring that the retrieved documents cover different aspects of the query.

```python
# Using MMR instead of similarity search
diverse_docs = vectorstore.max_marginal_relevance_search(
    query="What are the main themes in John Wick?",
    k=5,  # Return 5 documents
    fetch_k=15,  # Fetch 15 initial candidates
    lambda_mult=0.5  # Diversity vs relevance tradeoff (0-1)
)
```

## 3. Advanced RAG Architectures

### Recursive Retrieval RAG

Recursive retrieval uses the initial retrieval results to fuel additional retrieval steps, iteratively refining the context.

```mermaid
graph TD
    A[User Query] --> B[Initial Retrieval]
    B --> C[Initial Documents]
    C --> D[LLM Analysis]
    D --> E[Follow-up Queries]
    E --> F[Additional Retrieval]
    F --> G[Combined Context]
    G --> H[Final LLM]
    A --> H
    H --> I[Answer]
```

**Implementation**:

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create document chain for initial query
document_chain = create_stuff_documents_chain(llm, prompt)

# Define a recursive retrieval chain
recursive_chain = create_retrieval_chain(
    retriever, 
    document_chain,
    # Define how to handle the recursive aspect
    callbacks=[recursive_analyzer]
)

# Use the recursive retriever
result = recursive_chain.invoke({"input": "Explain how the High Table works in John Wick."})
```

### Agent-Based RAG

Agent-based RAG systems use an LLM to decide which retrieval actions to take, what queries to run, or how to filter results, effectively acting as a "research agent" that strategically gathers information.

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Create tools the agent can use
retrieval_tool = Tool(
    name="JohnWickRetriever",
    func=vectorstore.similarity_search,
    description="Search for information in the John Wick reviews database. Input should be a search query."
)

filter_tool = Tool(
    name="FilterByRating",
    func=lambda x: filter_by_rating(documents, int(x)),
    description="Filter documents by minimum rating (1-10). Input should be a number."
)

# Create the agent with access to tools
react_agent = create_react_agent(
    llm=ChatOpenAI(temperature=0),
    tools=[retrieval_tool, filter_tool],
    prompt=agent_prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=[retrieval_tool, filter_tool],
    verbose=True
)

# Run the agent
result = agent_executor.invoke(
    {"input": "What do the most positive reviews say about Keanu Reeves in John Wick 3?"}
)
```

### Multi-Index RAG

Multi-index RAG involves querying multiple vector stores, each containing different types of content or structured differently, then combining the results.

```python
from langchain.retrievers import MultiVectorRetriever
from langchain_community.storage import InMemoryStore

# Parent documents store
docstore = InMemoryStore()

# Create retrievers for different indexes
reviews_retriever = reviews_vectorstore.as_retriever(search_kwargs={"k": 3})
plots_retriever = plots_vectorstore.as_retriever(search_kwargs={"k": 2})
character_retriever = character_vectorstore.as_retriever(search_kwargs={"k": 2})

# Combine results with custom logic
def retrieve_from_multiple(query):
    review_docs = reviews_retriever.get_relevant_documents(query)
    plot_docs = plots_retriever.get_relevant_documents(query)
    character_docs = character_retriever.get_relevant_documents(query)
    
    # Custom merging logic - could use various ranking methods
    all_docs = review_docs + plot_docs + character_docs
    return all_docs[:5]  # Return top 5 combined results

# Use the multi-index retriever
results = retrieve_from_multiple("What is John Wick's motivation in the second film?")
```

## 4. Advanced Context Processing

### Document Compressors

Document compressors reduce the size of retrieved documents before sending them to the LLM, either by filtering out irrelevant sections or summarizing content.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create a document compressor that extracts only relevant information
compressor = LLMChainExtractor.from_llm(llm)

# Apply compression to documents
compressed_docs = []
for doc in documents:
    compressed = compressor.compress_documents(
        [doc],
        query="What are the action scenes like in John Wick?"
    )
    compressed_docs.extend(compressed)
```

### Token Windowing

Token windowing helps manage context limits by dynamically sliding a window through large documents to find the most relevant sections.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentWindower

# Create a document windower
windower = DocumentWindower(
    token_window_size=300,  # Size of each window
    token_overlap=50,  # Overlap between windows
    window_type="token"  # Window by tokens rather than characters
)

# Create a compression retriever with the windower
window_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    compressor=windower
)

# Get documents with dynamic windowing
windowed_docs = window_retriever.get_relevant_documents("What happens in the catacombs scene?")
```

## 5. RAG Evaluation and Optimization

### RAG Evaluation Metrics

Properly evaluating RAG systems involves multiple dimensions:

1. **Retrieval Accuracy**: How well the retriever finds relevant documents
2. **Answer Relevance**: How well the answer addresses the question
3. **Answer Groundedness**: How well the answer is supported by the retrieved context
4. **Answer Completeness**: Whether the answer includes all necessary information

As of 2025, several comprehensive evaluation frameworks have emerged:

**Implementation Using RAGBench**:

```python
from ragbench import evaluate, RAGEvaluationSet
from ragbench.metrics import (
    retrieval_precision,
    retrieval_recall,
    answer_faithfulness,
    answer_completeness,
    hallucination_score,
    context_efficiency
)

# Define evaluation data
eval_dataset = RAGEvaluationSet.from_jsonl("evaluation_data.jsonl")

# Create a comprehensive evaluation suite
evaluation_suite = RAGEvaluationSuite(
    metrics=[
        retrieval_precision,
        retrieval_recall,
        answer_faithfulness,
        answer_completeness,
        hallucination_score,
        context_efficiency
    ],
    llm_evaluator="gpt-4o"  # Using advanced LLM as judge
)

# Run evaluation
result = evaluation_suite.evaluate(
    rag_system=your_rag_pipeline,
    evaluation_set=eval_dataset,
    concurrency=4  # Parallel evaluation
)

# Get detailed performance analysis
report = result.generate_report(include_visualizations=True)
```

### Enhanced RAG Feedback Loop

A significant advancement in 2025 RAG systems is the implementation of continuous improvement through feedback loops:

```python
from langchain.evaluation import RAGFeedbackCollector
from langchain.evaluation.criteria import RetrievalCriteria

# Setup feedback collection
feedback_collector = RAGFeedbackCollector(
    criteria=[
        RetrievalCriteria.RELEVANCE,
        RetrievalCriteria.COMPLETENESS,
        RetrievalCriteria.DIVERSITY
    ],
    feedback_store="postgres://user:password@localhost:5432/feedback_db"
)

# In your application flow:
def process_query(query):
    # Get RAG response
    rag_response = rag_chain.invoke({"question": query})
    
    # Collect feedback (could be from users or automated evaluation)
    feedback_collector.collect_feedback(
        query=query,
        response=rag_response,
        retrieved_documents=rag_response["source_documents"],
        feedback_source="user"  # or "automated"
    )
    
    return rag_response

# Periodically analyze feedback and improve system
def optimize_system():
    insights = feedback_collector.analyze_feedback(timeframe="7d")
    # Use insights to retune retrievers, adjust chunking strategies, etc.
```

## 6. Multimodal RAG

Multimodal RAG represents one of the most significant advancements in RAG technology as of 2025. These systems can process, index, and retrieve information across different modalities including text, images, audio, and video.

### Multimodal Embedding and Indexing

```python
from langchain.embeddings import MultiModalEmbeddings
from langchain.vectorstores import MultiModalVectorStore
from langchain.document_processors import ImageTextExtractor, VideoFrameExtractor

# Initialize multimodal embedding model
mm_embeddings = MultiModalEmbeddings(
    text_model="text-embedding-3-large",
    image_model="clip-vit-large-patch14-336",
    fusion_method="weighted_sum"  # How to combine text and image embeddings
)

# Process different document types
text_documents = text_loader.load()
image_documents = image_loader.load()
video_documents = video_loader.load()

# Extract text from images where applicable
image_text_documents = ImageTextExtractor().process_documents(image_documents)

# Extract key frames and captions from videos
video_processed_documents = VideoFrameExtractor(
    frame_extraction_method="scene_change",
    caption_extraction=True
).process_documents(video_documents)

# Combine all documents
all_documents = text_documents + image_text_documents + video_processed_documents

# Create multimodal vector store
mm_vectorstore = MultiModalVectorStore.from_documents(
    documents=all_documents,
    embedding=mm_embeddings
)

# Create retriever that supports multimodal queries
mm_retriever = mm_vectorstore.as_retriever()
```

### Multimodal Query Processing

```python
from langchain.chains import MultiModalRAGChain

# Create a multimodal RAG chain
mm_chain = MultiModalRAGChain(
    retriever=mm_retriever,
    llm=ChatOpenAI(model="gpt-4o"),  # Using a multimodal LLM
    prompt_template="""
    Answer the question based on the following context which may include text, 
    images, or video content. Describe any visual content that's relevant to the answer.

    Context: {context}
    Question: {question}
    Answer:
    """
)

# Process a text query
text_response = mm_chain.invoke({"question": "How does John Wick's fighting style evolve through the movies?"})

# Process a query with an image
image_response = mm_chain.invoke({
    "question": "What's similar between this image and John Wick's fighting style?",
    "image": load_image("martial_arts.jpg")
})
```

### Cross-Modal Retrieval

One of the more powerful aspects of multimodal RAG is the ability to retrieve content across modalities:

```python
# Find relevant video clips based on a text description
video_results = mm_retriever.get_relevant_documents(
    "John Wick fighting with a pencil",
    filter={"modality": "video"},
    search_kwargs={"k": 3}
)

# Find images that match a concept described in text
image_results = mm_retriever.get_relevant_documents(
    "Continental Hotel architecture and design",
    filter={"modality": "image"},
    search_kwargs={"k": 5}
)
```

## 7. Privacy-Preserving RAG

As of 2025, privacy concerns have led to significant advancements in privacy-preserving RAG techniques:

```python
from langchain.retrievers import PrivacyPreservingRetriever
from langchain.privacy import DataAnonymizer, QueryObfuscator

# Set up data anonymization
anonymizer = DataAnonymizer(
    entities_to_anonymize=["PERSON", "ORG", "DATE"],
    preserve_entity_type=True
)

# Set up query obfuscation
query_obfuscator = QueryObfuscator(
    strategy="semantic_rephrasing",
    llm=ChatOpenAI(model="gpt-4-mini")
)

# Create privacy-preserving retriever
private_retriever = PrivacyPreservingRetriever(
    base_retriever=vectorstore.as_retriever(),
    data_anonymizer=anonymizer,
    query_obfuscator=query_obfuscator,
    deanonymize_results=True  # Convert back to original content before returning
)

# Use the privacy-preserving retriever
safe_results = private_retriever.get_relevant_documents("What did John Wick do at the Continental Hotel on June 14th?")
```

## References and Further Reading

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).
2. Gao, L., Ma, X., Lin, J., & Callan, J. (2023). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496). In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1776-1789.
3. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). In International Conference on Learning Representations (ICLR 2024, Oral Presentation).
4. Yoran, O., Wolfson, T., Ram, O., & Berant, J. (2023). [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://arxiv.org/abs/2310.01558).
5. Gao, L., Dai, Z., Pasupat, P., Chen, A., Chaganty, A. T., Fan, Y., Zhao, V., Lao, N., Lee, H., Juan, D., & Guu, K. (2023). [RARR: Researching and Revising What Language Models Say, Using Language Models](https://arxiv.org/abs/2210.08726). In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16838-16863.
6. Edge, S., Wang, H., Aspillaga, C., Cai, Y., Christodoulopoulos, C., Haas, S., Hsu, I., Lin, Y., Mathias, L., Noy, A., Tseng, J., Waibel, O., Xiong, D., Zelle, D., & Clarke, C. (2024). [GraphRAG: Graph-Based Retrieval Augmented Generation](https://microsoft.github.io/graphrag/). Microsoft Research.
7. Fan, T., Wang, J., Ren, X., & Huang, C. (2025). [MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation](https://arxiv.org/abs/2501.06713).
8. Xie, Y., Xu, A., Song, Y., Lee, K., & Liang, P. (2024). [Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications](https://arxiv.org/abs/2410.21943).
9. Ko, K., Joo, S., Jeon, H., Kang, U., & Seo, J. (2025). [Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval](https://arxiv.org/abs/2503.15548).
10. Friel, C., Belyi, A., Kannan, S., Kim, A., Mazumder, M., Mudgal, A., Putzky, P., Rosenbaum, R., Suglia, A., & Zelle, D. (2025). [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005).