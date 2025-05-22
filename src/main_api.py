# main_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging # Import logging
import time
from typing import Dict, List, Any, Optional

from src import settings

from src.chain_factory import (
    NAIVE_RETRIEVAL_CHAIN,
    BM25_RETRIEVAL_CHAIN,
    CONTEXTUAL_COMPRESSION_CHAIN,
    MULTI_QUERY_CHAIN,
    PARENT_DOCUMENT_CHAIN,
    ENSEMBLE_CHAIN,
    SEMANTIC_CHAIN
)

from contextlib import asynccontextmanager

# Get a logger for this module
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("FastAPI application startup...")
    logger.info("Initializing components and chains...")

    available_chains = {
        "Naive Retriever Chain": NAIVE_RETRIEVAL_CHAIN,
        "BM25 Retriever Chain": BM25_RETRIEVAL_CHAIN,
        "Contextual Compression Chain": CONTEXTUAL_COMPRESSION_CHAIN,
        "Multi-Query Chain": MULTI_QUERY_CHAIN,
        "Parent Document Chain": PARENT_DOCUMENT_CHAIN,
        "Ensemble Chain": ENSEMBLE_CHAIN,
        "Semantic Chain": SEMANTIC_CHAIN
    }

    logger.info("\n--- Chain Initialization Status ---")
    all_chains_ready = True
    for name, chain_instance in available_chains.items():
        if chain_instance is not None:
            logger.info(f"[+] {name}: Ready")
        else:
            logger.warning(f"[-] {name}: Not available. Check logs for retriever/vectorstore initialization issues.")
            all_chains_ready = False
    
    if all_chains_ready:
        logger.info("All chains initialized successfully.")
    else:
        logger.warning("One or more chains failed to initialize. API functionality may be limited.")
    logger.info("------------------------------------------------------")
    yield
    # Code to run on shutdown
    logger.info("FastAPI application shutdown.")

app = FastAPI(
    title="Advanced RAG Retriever API",
    description="API for invoking various LangChain retrieval chains for John Wick movie reviews.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan context manager
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    # context: list # You might want to define a Pydantic model for Document if returning context
    context_document_count: int

async def invoke_chain_logic(chain, question: str, chain_name: str):
    if chain is None:
        logger.error(f"Chain '{chain_name}' is not available (None). Cannot process request for question: '{question}'")
        raise HTTPException(status_code=503, detail=f"The '{chain_name}' is currently unavailable. Please check server logs.")
    try:
        logger.info(f"Invoking '{chain_name}' with question: '{question[:50]}...'")
        result = await chain.ainvoke({"question": question})
        answer = result.get("response", {}).content if hasattr(result.get("response"), "content") else "No answer content found."
        context_docs_count = len(result.get("context", []))
        logger.info(f"'{chain_name}' invocation successful. Answer: '{answer[:50]}...', Context docs: {context_docs_count}")
        return AnswerResponse(answer=answer, context_document_count=context_docs_count)
    except Exception as e:
        logger.error(f"Error invoking '{chain_name}' for question '{question[:50]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request with {chain_name}.")

@app.post("/invoke/naive_retriever", response_model=AnswerResponse)
async def invoke_naive_endpoint(request: QuestionRequest):
    """Invokes the Naive Retriever chain."""
    return await invoke_chain_logic(NAIVE_RETRIEVAL_CHAIN, request.question, "Naive Retriever Chain")

@app.post("/invoke/bm25_retriever", response_model=AnswerResponse)
async def invoke_bm25_endpoint(request: QuestionRequest):
    """Invokes the BM25 Retriever chain."""
    return await invoke_chain_logic(BM25_RETRIEVAL_CHAIN, request.question, "BM25 Retriever Chain")

@app.post("/invoke/contextual_compression_retriever", response_model=AnswerResponse)
async def invoke_contextual_compression_endpoint(request: QuestionRequest):
    """Invokes the Contextual Compression Retriever chain."""
    return await invoke_chain_logic(CONTEXTUAL_COMPRESSION_CHAIN, request.question, "Contextual Compression Chain")

@app.post("/invoke/multi_query_retriever", response_model=AnswerResponse)
async def invoke_multi_query_endpoint(request: QuestionRequest):
    """Invokes the Multi-Query Retriever chain."""
    return await invoke_chain_logic(MULTI_QUERY_CHAIN, request.question, "Multi-Query Chain")

@app.post("/invoke/parent_document_retriever", response_model=AnswerResponse)
async def invoke_parent_document_endpoint(request: QuestionRequest):
    """Invokes the Parent Document Retriever chain."""
    return await invoke_chain_logic(PARENT_DOCUMENT_CHAIN, request.question, "Parent Document Chain")

@app.post("/invoke/ensemble_retriever", response_model=AnswerResponse)
async def invoke_ensemble_endpoint(request: QuestionRequest):
    """Invokes the Ensemble Retriever chain."""
    return await invoke_chain_logic(ENSEMBLE_CHAIN, request.question, "Ensemble Chain")

@app.post("/invoke/semantic_retriever", response_model=AnswerResponse)
async def invoke_semantic_endpoint(request: QuestionRequest):
    """Invokes the Semantic Retriever chain."""
    return await invoke_chain_logic(SEMANTIC_CHAIN, request.question, "Semantic Chain")

# --- Add compatibility models and endpoint for run.py interface ---

# Compatibility models to match run.py interface
class RetrievalRequest(BaseModel):
    query: str = Field(..., description="The search query")
    retriever_type: str = Field(..., description="Type of retriever to use")
    top_k: int = Field(5, description="Number of documents to retrieve")
    model: Optional[str] = Field(None, description="Optional model name for embedding-based retrievers")

class Document(BaseModel):
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class RetrievalResponse(BaseModel):
    query: str = Field(..., description="Original query")
    retriever_type: str = Field(..., description="Type of retriever used")
    documents: List[Document] = Field(default_factory=list, description="Retrieved documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Retrieval metadata")

# Mapping between run.py retriever types and main_api.py retriever types
RETRIEVER_MAPPING = {
    "bm25": "bm25_retriever",
    "dense": "semantic_retriever",
    "hybrid": "ensemble_retriever", 
    "semantic": "semantic_retriever",
    "naive": "naive_retriever",
    "contextual_compression": "contextual_compression_retriever",
    "multi_query": "multi_query_retriever",
    "parent_document": "parent_document_retriever",
    "ensemble": "ensemble_retriever"
}

# Compatibility endpoint
@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest):
    """
    Compatibility endpoint matching the run.py interface.
    
    This endpoint bridges between the simulated API and the real LangChain implementation,
    transforming both requests and responses to maintain backward compatibility.
    """
    start_time = time.time()
    
    # Validate retriever type
    if request.retriever_type not in RETRIEVER_MAPPING:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported retriever type: {request.retriever_type}"
        )
    
    # Map to corresponding LangChain retriever
    actual_retriever = RETRIEVER_MAPPING[request.retriever_type]
    logger.info(f"Mapped retriever type '{request.retriever_type}' to '{actual_retriever}'")
    
    try:
        # Get the appropriate chain based on the retriever type
        chain = None
        if actual_retriever == "naive_retriever":
            chain = NAIVE_RETRIEVAL_CHAIN
        elif actual_retriever == "bm25_retriever":
            chain = BM25_RETRIEVAL_CHAIN
        elif actual_retriever == "contextual_compression_retriever":
            chain = CONTEXTUAL_COMPRESSION_CHAIN
        elif actual_retriever == "multi_query_retriever":
            chain = MULTI_QUERY_CHAIN
        elif actual_retriever == "parent_document_retriever":
            chain = PARENT_DOCUMENT_CHAIN
        elif actual_retriever == "ensemble_retriever":
            chain = ENSEMBLE_CHAIN
        elif actual_retriever == "semantic_retriever":
            chain = SEMANTIC_CHAIN
        
        if not chain:
            logger.error(f"Chain not found for retriever: {actual_retriever}")
            raise HTTPException(
                status_code=500,
                detail=f"Retriever chain not available: {actual_retriever}"
            )
        
        logger.info(f"Invoking chain for retriever type '{request.retriever_type}' with query: '{request.query}'")
        
        # Invoke the chain with the query
        result = await chain.ainvoke({"question": request.query})
        
        # Log the structure of the result for debugging
        logger.info(f"Chain response keys: {list(result.keys())}")
        for key in result.keys():
            value_type = type(result[key]).__name__
            logger.info(f"Key '{key}' has type: {value_type}")
            
            # For particularly important keys, log more details
            if key in ["source_documents", "context_documents", "context", "response"]:
                if isinstance(result[key], list):
                    logger.info(f"Key '{key}' is a list with {len(result[key])} items")
                    if len(result[key]) > 0:
                        sample_item_type = type(result[key][0]).__name__
                        logger.info(f"First item in '{key}' has type: {sample_item_type}")
                        # If it's a Document, log its structure
                        if hasattr(result[key][0], 'page_content'):
                            logger.info(f"First item in '{key}' has page_content and metadata attributes")
                            logger.info(f"Metadata keys: {list(result[key][0].metadata.keys()) if hasattr(result[key][0], 'metadata') else 'No metadata'}")
                elif hasattr(result[key], 'content'):
                    logger.info(f"Key '{key}' has a 'content' attribute of type: {type(result[key].content).__name__}")
        
        # Extract documents from chain response
        documents = []
        source_docs = []
        
        # Different chains might store documents in different keys - check all possible locations
        if "source_documents" in result:
            source_docs = result["source_documents"]
            logger.info(f"Found documents in 'source_documents' key ({len(source_docs)} documents)")
        elif "context_documents" in result:
            source_docs = result["context_documents"]
            logger.info(f"Found documents in 'context_documents' key ({len(source_docs)} documents)")
        elif "context" in result:
            # Context could be either a list of documents or a string
            if isinstance(result["context"], list) and len(result["context"]) > 0 and hasattr(result["context"][0], "page_content"):
                source_docs = result["context"]
                logger.info(f"Found documents in 'context' key ({len(source_docs)} documents)")
            else:
                logger.info(f"'context' key exists but does not contain Document objects")
        elif "documents" in result:
            source_docs = result["documents"]
            logger.info(f"Found documents in 'documents' key ({len(source_docs)} documents)")
        # Check if documents are in response.context (common in newer LangChain chains)
        elif "response" in result and hasattr(result["response"], "context") and result["response"].context:
            if isinstance(result["response"].context, list) and len(result["response"].context) > 0:
                source_docs = result["response"].context
                logger.info(f"Found documents in 'response.context' attribute ({len(source_docs)} documents)")
        else:
            # Check if any key contains a list of objects with page_content attribute
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 0 and hasattr(value[0], "page_content"):
                    source_docs = value
                    logger.info(f"Found documents in '{key}' key ({len(source_docs)} documents)")
                    break
            if not source_docs:
                logger.warning(f"No documents found in any expected keys. Keys available: {list(result.keys())}")
            
        # If we still don't have documents, try to extract from the result object if it has specific attributes
        if not source_docs and hasattr(result, 'source_documents'):
            source_docs = result.source_documents
            logger.info(f"Found documents as 'source_documents' attribute ({len(source_docs)} documents)")
        
        # Log the document extraction results
        logger.info(f"Extracted {len(source_docs)} documents")
        if len(source_docs) > 0:
            logger.info(f"First document type: {type(source_docs[0]).__name__}")
            if hasattr(source_docs[0], 'metadata'):
                logger.info(f"First document metadata keys: {list(source_docs[0].metadata.keys())}")
            
        # Limit to requested top_k
        for i, doc in enumerate(source_docs[:request.top_k]):
            try:
                # Extract document ID or generate one
                doc_id = str(doc.metadata.get("id", f"doc_{i}_{hash(doc.page_content) % 10000}"))
                
                # Extract score from metadata if available, or generate a fallback score
                # Try various common score keys
                score = None
                for score_key in ["score", "similarity", "relevance_score", "relevancy_score"]:
                    if score_key in doc.metadata:
                        score = doc.metadata[score_key]
                        logger.debug(f"Found score in document metadata with key '{score_key}': {score}")
                        break
                
                # If no score found, generate a fallback score based on position
                if score is None:
                    score = 1.0 - (i * 0.05)  # Simple decay function based on position
                    logger.debug(f"Using fallback position-based score for document {i}: {score}")
                
                # Ensure score is a float
                score = float(score)
                
                # Create a document in the expected format
                documents.append(Document(
                    id=doc_id,
                    content=doc.page_content,
                    score=score,
                    metadata={
                        "retriever": request.retriever_type,
                        # Include original metadata, excluding score keys which are already used
                        **{k: v for k, v in doc.metadata.items() if k not in ["score", "similarity", "relevance_score", "relevancy_score"]}
                    }
                ))
                logger.debug(f"Processed document {i+1}: ID={doc_id}, Score={score}")
            except Exception as doc_error:
                logger.error(f"Error processing document {i}: {str(doc_error)}")
                # Try a more generic approach if the standard approach fails
                try:
                    doc_id = f"doc_{i}"
                    content = str(doc) if not hasattr(doc, 'page_content') else doc.page_content
                    documents.append(Document(
                        id=doc_id,
                        content=content,
                        score=float(0.9 - (i * 0.1)),
                        metadata={"retriever": request.retriever_type}
                    ))
                    logger.debug(f"Processed document {i+1} using fallback method")
                except Exception as fallback_error:
                    logger.error(f"Fallback processing failed for document {i}: {str(fallback_error)}")
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Return response in the expected format
        logger.info(f"Returning {len(documents)} documents with latency {latency_ms:.2f}ms")
        return RetrievalResponse(
            query=request.query,
            retriever_type=request.retriever_type,
            documents=documents,
            metadata={
                "top_k": request.top_k,
                "latency_ms": latency_ms,
                "engine": f"langchain-{request.retriever_type}-retriever",
                "document_count": len(documents)
            }
        )
        
    except Exception as e:
        logger.exception(f"Error processing retrieval request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
        )

@app.get("/retrievers")
async def get_available_retrievers():
    """
    Returns a list of available retriever types.
    This endpoint is used by the evaluation pipeline to discover available retrievers.
    """
    return {
        "retrievers": list(RETRIEVER_MAPPING.keys()),
        "status": "success"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting FastAPI server using uvicorn.run() from __main__...")

    uvicorn.run(app, host="0.0.0.0", port=8000) 