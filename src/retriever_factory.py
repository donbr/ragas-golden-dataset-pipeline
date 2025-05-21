# retriever_factory.py
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import os

from src import settings
# settings.setup_env_vars() # Called by settings import

from src.data_loader import load_documents # For BM25 and ParentDocumentRetriever original docs
from src.llm_models import get_chat_model
from src.embeddings import get_openai_embeddings # For SemanticChunker if used directly here
from src.vectorstore_setup import (
    get_main_vectorstore, 
    get_parent_document_vectorstore_client_and_collection,
    get_semantic_vectorstore
)
# For SemanticChunker if ParentDocumentRetriever needs to re-chunk (it uses child_splitter)
# from langchain_experimental.text_splitter import SemanticChunker 

logger = logging.getLogger(__name__)

# Initialize base components that might be shared or needed early
logger.debug("Loading documents for retriever factory...")
DOCUMENTS = load_documents()
logger.debug("Initializing chat model for retriever factory...")
CHAT_MODEL = get_chat_model()
# EMBEDDINGS = get_openai_embeddings() # Already initialized in vectorstore_setup

logger.debug("Initializing vector stores for retriever factory...")
MAIN_VECTORSTORE = None
SEMANTIC_VECTORSTORE = None
PARENT_DOC_VS = None

if DOCUMENTS: # Only attempt to create stores if documents were loaded
    try:
        MAIN_VECTORSTORE = get_main_vectorstore()
    except Exception as e:
        logger.error(f"Failed to initialize MAIN_VECTORSTORE in retriever_factory: {e}", exc_info=True)
    try:
        SEMANTIC_VECTORSTORE = get_semantic_vectorstore()
    except Exception as e:
        logger.error(f"Failed to initialize SEMANTIC_VECTORSTORE in retriever_factory: {e}", exc_info=True)
    try:
        PARENT_DOC_VS, _ = get_parent_document_vectorstore_client_and_collection()
    except Exception as e:
        logger.error(f"Failed to initialize PARENT_DOC_VS in retriever_factory: {e}", exc_info=True)
else:
    logger.warning("No documents loaded; vector stores will not be initialized in retriever_factory.")


def get_naive_retriever():
    if not MAIN_VECTORSTORE:
        logger.warning("Main vectorstore not available for naive_retriever. Returning None.")
        return None
    logger.info("Creating naive_retriever (MAIN_VECTORSTORE.as_retriever).")
    return MAIN_VECTORSTORE.as_retriever(search_kwargs={"k": 10})

def get_bm25_retriever():
    if not DOCUMENTS:
        logger.warning("Documents not available for bm25_retriever. Returning None.")
        return None
    logger.info("Creating bm25_retriever...")
    try:
        retriever = BM25Retriever.from_documents(DOCUMENTS)
        logger.info("BM25Retriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create BM25Retriever: {e}", exc_info=True)
        return None

def get_contextual_compression_retriever():
    logger.info("Attempting to create contextual_compression_retriever...")
    naive_ret = get_naive_retriever()
    if not naive_ret:
        logger.warning("Naive retriever not available, cannot create contextual_compression_retriever. Returning None.")
        return None
    if not os.getenv("COHERE_API_KEY"): # Check for Cohere key
        logger.warning("COHERE_API_KEY not set. Cannot create CohereRerank for contextual_compression_retriever. Returning None.")
        return None
    try:
        compressor = CohereRerank(model="rerank-english-v3.0") 
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=naive_ret
        )
        logger.info("ContextualCompressionRetriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create ContextualCompressionRetriever: {e}", exc_info=True)
        return None

def get_multi_query_retriever():
    logger.info("Attempting to create multi_query_retriever...")
    naive_ret = get_naive_retriever()
    if not naive_ret:
        logger.warning("Naive retriever not available, cannot create multi_query_retriever. Returning None.")
        return None
    try:
        retriever = MultiQueryRetriever.from_llm(
            retriever=naive_ret, llm=CHAT_MODEL
        )
        logger.info("MultiQueryRetriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create MultiQueryRetriever: {e}", exc_info=True)
        return None

def get_parent_document_retriever():
    logger.info("Attempting to create parent_document_retriever...")
    if not DOCUMENTS or not PARENT_DOC_VS:
        logger.warning("Documents or Parent Document Vectorstore (PARENT_DOC_VS) not available for parent_document_retriever. Returning None.")
        return None
    
    try:
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        store = InMemoryStore() 

        retriever = ParentDocumentRetriever(
            vectorstore=PARENT_DOC_VS, 
            docstore=store,          
            child_splitter=child_splitter,
        )
        # Add documents. IDs are optional.
        if not DOCUMENTS: 
            logger.warning("No documents to add to ParentDocumentRetriever store (this check is redundant but safe).")
            # Retriever is created but store might be empty if DOCUMENTS became None unexpectedly
        else:
            logger.info(f"Adding {len(DOCUMENTS)} documents to ParentDocumentRetriever store...")
            retriever.add_documents(DOCUMENTS, ids=None)
            logger.info("Documents added to ParentDocumentRetriever store.")
        return retriever
    except Exception as e:
        logger.error(f"Error creating or adding documents to ParentDocumentRetriever: {e}", exc_info=True)
        return None


def get_semantic_retriever():
    if not SEMANTIC_VECTORSTORE:
        logger.warning("Semantic vectorstore not available for semantic_retriever. Returning None.")
        return None
    logger.info("Creating semantic_retriever (SEMANTIC_VECTORSTORE.as_retriever).")
    return SEMANTIC_VECTORSTORE.as_retriever(search_kwargs={"k": 10})


def get_ensemble_retriever():
    logger.info("Attempting to create ensemble_retriever...")
    retrievers_to_ensemble_map = {
        "bm25": get_bm25_retriever(),
        "naive": get_naive_retriever(),
        "parent_doc": get_parent_document_retriever(),
        "contextual_compression": get_contextual_compression_retriever(), # Can be slow for ensemble
        "multi_query": get_multi_query_retriever() # Can be slow for ensemble
        # "semantic": get_semantic_retriever()
    }

    active_retrievers = [r for r_name, r in retrievers_to_ensemble_map.items() if r is not None]
    active_retriever_names = [r_name for r_name, r in retrievers_to_ensemble_map.items() if r is not None]

    if not active_retrievers:
        logger.warning("No retrievers available for ensemble_retriever. Returning None.")
        return None
    if len(active_retrievers) < 2:
         logger.warning(f"Ensemble retriever requires at least 2 active retrievers, got {len(active_retrievers)} ({active_retriever_names}). Returning the first one or None.")
         return active_retrievers[0] if active_retrievers else None

    logger.info(f"Creating EnsembleRetriever with retrievers: {active_retriever_names}")
    try:
        equal_weighting = [1.0 / len(active_retrievers)] * len(active_retrievers)
        retriever = EnsembleRetriever(
            retrievers=active_retrievers, weights=equal_weighting
        )
        logger.info("EnsembleRetriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create EnsembleRetriever: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        if 'logging_config' not in globals():
            from src import logging_config
        logging_config.setup_logging()
    
    logger.info("--- Running retriever_factory.py standalone test ---")
    if not DOCUMENTS:
        logger.warning("No documents were loaded by data_loader. Retriever initialization will be limited.")
    else:
        logger.info(f"{len(DOCUMENTS)} documents loaded. Proceeding with retriever initialization tests...")

    retrievers_status = {}
    retrievers_status["Naive"] = get_naive_retriever()
    retrievers_status["BM25"] = get_bm25_retriever()
    retrievers_status["Contextual Compression"] = get_contextual_compression_retriever()
    retrievers_status["Multi-Query"] = get_multi_query_retriever()
    retrievers_status["Parent Document"] = get_parent_document_retriever()
    retrievers_status["Semantic"] = get_semantic_retriever()
    retrievers_status["Ensemble"] = get_ensemble_retriever()

    logger.info("\n--- Retriever Initialization Status ---")
    for name, r_instance in retrievers_status.items():
        logger.info(f"{name} Retriever: {'Ready' if r_instance else 'Failed/Not Available'}")

    # Example usage test for a retriever that is expected to be ready
    # test_retriever = retrievers_status.get("Naive")
    # if test_retriever:
    #     logger.info("\nTesting Naive Retriever invocation...")
    #     try:
    #         results = test_retriever.invoke("What happened in John Wick?")
    #         logger.info(f"Naive retriever sample result (first doc metadata if any): {results[0].metadata if results else 'No results'}")
    #         logger.info(f"Naive retriever returned {len(results)} documents.")
    #     except Exception as e:
    #         logger.error(f"Error testing Naive Retriever: {e}", exc_info=True)
    # else:
    #     logger.warning("Naive retriever not available for invocation test.")

    logger.info("--- Finished retriever_factory.py standalone test ---") 