# vectorstore_setup.py
from langchain_qdrant import Qdrant, QdrantVectorStore
from qdrant_client import QdrantClient, models as qdrant_models
from langchain_experimental.text_splitter import SemanticChunker
import logging

from src import settings

from src.data_loader import load_documents
from src.embeddings import get_openai_embeddings

logger = logging.getLogger(__name__)

# Initialize once
logger.debug("Loading documents for vector store setup...")
DOCUMENTS = load_documents()
logger.debug("Initializing embeddings for vector store setup...")
EMBEDDINGS = get_openai_embeddings()


def get_main_vectorstore():
    logger.info("Attempting to create main vector store 'JohnWickMain'...")
    if not DOCUMENTS:
        logger.error("No documents loaded, cannot create main vectorstore 'JohnWickMain'.")
        raise ValueError("No documents loaded, cannot create main vectorstore.")
    try:
        vs = Qdrant.from_documents(
            DOCUMENTS,
            EMBEDDINGS,
            location=":memory:", 
            collection_name="JohnWickMain"
        )
        logger.info("Main vector store 'JohnWickMain' created successfully.")
        return vs
    except Exception as e:
        logger.error(f"Failed to create main vector store 'JohnWickMain': {e}", exc_info=True)
        raise

def get_parent_document_vectorstore_client_and_collection():
    logger.info("Attempting to create/get parent document vector store 'full_documents'...")
    try:
        client = QdrantClient(location=":memory:") 
        collection_name = "full_documents"
        
        try:
            client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception: 
            logger.info(f"Collection '{collection_name}' not found. Creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(size=1536, distance=qdrant_models.Distance.COSINE)
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        
        vectorstore = QdrantVectorStore(
            collection_name=collection_name, 
            embedding=EMBEDDINGS, 
            client=client
        )
        logger.info(f"Parent document vector store for '{collection_name}' configured successfully.")
        return vectorstore, client
    except Exception as e:
        logger.error(f"Failed to create/get parent document vector store 'full_documents': {e}", exc_info=True)
        raise


def get_semantic_vectorstore():
    logger.info("Attempting to create semantic vector store 'JohnWickSemantic'...")
    if not DOCUMENTS:
        logger.error("No documents loaded, cannot create semantic vectorstore 'JohnWickSemantic'.")
        raise ValueError("No documents loaded, cannot create semantic vectorstore.")
    try:
        semantic_chunker = SemanticChunker(
            EMBEDDINGS,
            breakpoint_threshold_type="percentile"
        )
        logger.info("Splitting documents with SemanticChunker...")
        semantic_documents = semantic_chunker.split_documents(DOCUMENTS)
        if not semantic_documents:
            logger.warning("SemanticChunker produced no documents. The vector store 'JohnWickSemantic' will be empty.")
        else:
            logger.info(f"SemanticChunker produced {len(semantic_documents)} documents.")

        vs = Qdrant.from_documents(
            semantic_documents if semantic_documents else [],
            EMBEDDINGS,
            location=":memory:",
            collection_name="JohnWickSemantic"
        )
        logger.info("Semantic vector store 'JohnWickSemantic' created successfully.")
        return vs
    except Exception as e:
        logger.error(f"Failed to create semantic vector store 'JohnWickSemantic': {e}", exc_info=True)
        raise

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        if 'logging_config' not in globals():
            from src import logging_config
        logging_config.setup_logging()

    logger.info("--- Running vectorstore_setup.py standalone test ---")
    if not DOCUMENTS:
        logger.warning("No documents were loaded by data_loader. Cannot proceed with vector store creation tests.")
    else:
        logger.info("Documents loaded, proceeding with vector store creation tests...")
        try:
            main_vs = get_main_vectorstore()
            if main_vs:
                logger.info(f"Main vector store '{main_vs.collection_name}' test instance created.")
                client = QdrantClient(location=":memory:")
                count = client.count(collection_name=main_vs.collection_name).count
                logger.info(f"-> Points in '{main_vs.collection_name}': {count}")
        except Exception as e:
            logger.error(f"Error during Main vector store test: {e}", exc_info=True)

        try:
            parent_vs, parent_client = get_parent_document_vectorstore_client_and_collection()
            if parent_vs and parent_client:
                logger.info(f"Parent document vector store '{parent_vs.collection_name}' test instance configured.")
                try:
                    count_result = parent_client.count(collection_name=parent_vs.collection_name)
                    logger.info(f"-> Points in '{parent_vs.collection_name}': {count_result.count}")
                except Exception as e:
                    logger.error(f"--> Error counting points in '{parent_vs.collection_name}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error during Parent Document vector store test: {e}", exc_info=True)
        
        try:
            semantic_vs = get_semantic_vectorstore()
            if semantic_vs:
                logger.info(f"Semantic vector store '{semantic_vs.collection_name}' test instance created.")
                client = QdrantClient(location=":memory:")
                count = client.count(collection_name=semantic_vs.collection_name).count
                logger.info(f"-> Points in '{semantic_vs.collection_name}': {count}")
        except Exception as e:
            logger.error(f"Error during Semantic vector store test: {e}", exc_info=True)

    logger.info("--- Finished vectorstore_setup.py standalone test ---") 