"""
# Capturing LangChain Document Loader Metadata in a Pandas DataFrame

This module provides utilities for standardizing metadata from various LangChain document loaders 
into consistent pandas DataFrames. It focuses particularly on scientific publication loaders 
like ArXiv and PubMed, but also supports other document loaders.

## Purpose
- Normalize metadata fields across different document types
- Capture both standard and loader-specific metadata
- Provide a consistent DataFrame structure for analysis and evaluation

## Usage
```python
# Initialize the manager
metadata_manager = DocumentMetadataManager()

# Load and standardize documents
loader = ArxivLoader(query="agentic AI", max_results=10)
docs = loader.load()
docs_dataframe = metadata_manager.documents_to_dataframe(docs, loader_type="ArxivLoader")

# Handle multiple document types
pubmed_loader = PubmedLoader(query="GPT-4", max_results=5)
pubmed_docs = pubmed_loader.load()
all_docs = docs + pubmed_docs
combined_df = metadata_manager.documents_to_dataframe(all_docs, include_all_metadata=True)
```

## References
- LangChain DocumentLoaders: https://python.langchain.com/docs/integrations/document_loaders/
- ArXiv Loader: https://python.langchain.com/docs/integrations/document_loaders/arxiv
- PubMed Loader: https://python.langchain.com/docs/integrations/document_loaders/pubmed
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Set
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables with defaults
RAW_DIR = os.environ.get("RAW_DIR", "data/raw")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")

class DocumentMetadataManager:
    """
    Utility class for standardizing metadata from LangChain documents
    and converting document collections to pandas DataFrames.
    """
    
    def __init__(self, standard_metadata_keys: Optional[List[str]] = None):
        """
        Initialize with standard metadata keys to extract.
        
        Args:
            standard_metadata_keys: List of keys to extract as standard columns
        """
        self.standard_metadata_keys = standard_metadata_keys or [
            "source", "title", "author", "created_at", "publication_date", 
            "document_type", "url", "doi", "page", "total_pages", "file_path", 
            "file_name", "loader_type", "abstract", "published", "journal", 
            "publication", "entry_id"
        ]
    
    def standardize_metadata(self, documents: List[Document], loader_type: Optional[str] = None) -> List[Document]:
        """
        Standardize metadata across a list of documents.
        
        Args:
            documents: List of LangChain Document objects
            loader_type: Optional string indicating the loader type
            
        Returns:
            List of Document objects with standardized metadata
        """
        standardized_docs = []
        
        for i, doc in enumerate(documents):
            new_metadata = doc.metadata.copy()
            
            # Add inferred fields
            if loader_type and "loader_type" not in new_metadata:
                new_metadata["loader_type"] = loader_type
                
            # Handle source field
            if "source" not in new_metadata:
                if "file_path" in new_metadata:
                    new_metadata["source"] = new_metadata["file_path"]
                elif "url" in new_metadata:
                    new_metadata["source"] = new_metadata["url"]
                elif "entry_id" in new_metadata:
                    new_metadata["source"] = new_metadata["entry_id"]
            
            # Handle document type field
            if "document_type" not in new_metadata:
                # Try to infer from loader type
                if loader_type:
                    if "arxiv" in loader_type.lower():
                        new_metadata["document_type"] = "scientific_paper"
                    elif "pubmed" in loader_type.lower():
                        new_metadata["document_type"] = "medical_paper"
                    elif "pdf" in loader_type.lower():
                        new_metadata["document_type"] = "pdf"
                    elif "web" in loader_type.lower():
                        new_metadata["document_type"] = "web_page"
                    elif "csv" in loader_type.lower():
                        new_metadata["document_type"] = "tabular"
                # Try to infer from file extension
                elif "source" in new_metadata:
                    source = str(new_metadata["source"]).lower()
                    if source.endswith(".pdf"):
                        new_metadata["document_type"] = "pdf"
                    elif source.endswith((".docx", ".doc")):
                        new_metadata["document_type"] = "word"
                    elif source.endswith((".csv", ".tsv")):
                        new_metadata["document_type"] = "tabular"
                    elif "arxiv.org" in source:
                        new_metadata["document_type"] = "scientific_paper"
                    elif "pubmed" in source:
                        new_metadata["document_type"] = "medical_paper"
            
            # Add index or chunk ID if not present
            if "chunk_id" not in new_metadata:
                new_metadata["chunk_id"] = i
            
            # Normalize ArXiv metadata
            if loader_type and "arxiv" in loader_type.lower():
                if "Published" in new_metadata:
                    new_metadata["publication_date"] = new_metadata["Published"]
                if "Title" in new_metadata:
                    new_metadata["title"] = new_metadata["Title"]
                if "Authors" in new_metadata:
                    new_metadata["author"] = new_metadata["Authors"]
                if "Summary" in new_metadata:
                    new_metadata["abstract"] = new_metadata["Summary"]
                if "entry_id" in new_metadata:
                    arxiv_id = new_metadata["entry_id"]
                    if isinstance(arxiv_id, str) and "arxiv.org" in arxiv_id:
                        new_metadata["url"] = arxiv_id
                        id_parts = arxiv_id.split("/")
                        if len(id_parts) > 0:
                            new_metadata["arxiv_id"] = id_parts[-1]
            
            # Normalize PubMed metadata
            if loader_type and "pubmed" in loader_type.lower():
                if "publication_date" not in new_metadata and "pubdate" in new_metadata:
                    new_metadata["publication_date"] = new_metadata["pubdate"]
                if "title" not in new_metadata and "article_title" in new_metadata:
                    new_metadata["title"] = new_metadata["article_title"]
                if "journal" not in new_metadata and "journal_title" in new_metadata:
                    new_metadata["journal"] = new_metadata["journal_title"]
                if "abstract" not in new_metadata and "abstract_text" in new_metadata:
                    new_metadata["abstract"] = new_metadata["abstract_text"]
                if "pmid" in new_metadata:
                    new_metadata["entry_id"] = new_metadata["pmid"]
                    new_metadata["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{new_metadata['pmid']}/"
            
            standardized_docs.append(Document(
                page_content=doc.page_content,
                metadata=new_metadata
            ))
        
        return standardized_docs
    
    def documents_to_dataframe(
        self, 
        documents: List[Document], 
        loader_type: Optional[str] = None, 
        include_content: bool = True, 
        include_all_metadata: bool = False,
        max_content_length: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert documents to a pandas DataFrame with standardized columns.
        
        Args:
            documents: List of LangChain Document objects
            loader_type: Optional string indicating the loader type
            include_content: Whether to include the page_content in the DataFrame
            include_all_metadata: Whether to include all metadata fields
            max_content_length: Optional limit for content length in the DataFrame
            
        Returns:
            pandas DataFrame with standardized columns
        """
        # First standardize the metadata
        if loader_type:
            documents = self.standardize_metadata(documents, loader_type)
        
        rows = []
        all_metadata_keys = set()
        
        # First pass to collect all metadata keys if needed
        if include_all_metadata:
            for doc in documents:
                all_metadata_keys.update(doc.metadata.keys())
        
        # Determine columns to include
        columns_to_extract = self.standard_metadata_keys.copy()
        if include_all_metadata:
            columns_to_extract.extend([k for k in all_metadata_keys 
                                      if k not in self.standard_metadata_keys])
        
        # Second pass to create rows
        for doc in documents:
            row = {}
            
            if include_content:
                content = doc.page_content
                if max_content_length and len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                row["page_content"] = content
                
            # Extract standard and additional metadata
            for key in columns_to_extract:
                row[key] = doc.metadata.get(key, None)
                
            # Store any remaining metadata as JSON if needed
            if not include_all_metadata:
                custom_keys = set(doc.metadata.keys()) - set(columns_to_extract)
                if custom_keys:
                    row["additional_metadata"] = {k: doc.metadata[k] for k in custom_keys}
            
            rows.append(row)
        
        return pd.DataFrame(rows)

    def ragas_testset_to_dataframe(self, testset, include_source_docs: bool = True):
        """
        Convert a RAGAS testset to a pandas DataFrame.
        
        Args:
            testset: RAGAS testset object
            include_source_docs: Whether to include source document info
            
        Returns:
            pandas DataFrame with testset data
        """
        rows = []
        
        # Extract data from RAGAS testset
        for item in testset:
            row = {
                "question": item.question,
                "answer": item.answer
            }
            
            # Include context information if available
            if hasattr(item, "contexts") and item.contexts:
                row["contexts"] = item.contexts
                
                # Extract metadata from context documents if needed
                if include_source_docs and isinstance(item.contexts[0], Document):
                    source_docs = []
                    for ctx_doc in item.contexts:
                        source_docs.append({
                            "content": ctx_doc.page_content[:100] + "...",  # Truncate for brevity
                            "source": ctx_doc.metadata.get("source", "Unknown"),
                            "title": ctx_doc.metadata.get("title", None),
                            "page": ctx_doc.metadata.get("page", None)
                        })
                    row["source_documents"] = source_docs
            
            rows.append(row)
        
        return pd.DataFrame(rows)


class ArxivLoader:
    """
    Example wrapper around the LangChain ArxivLoader with metadata enhancements.
    This extends the standard LangChain ArxivLoader to provide more consistent metadata.
    
    For an actual implementation, see:
    https://python.langchain.com/docs/integrations/document_loaders/arxiv
    """
    
    def __init__(self, query: str, max_results: int = 10, load_all_available_meta: bool = True):
        """
        Initialize the ArxivLoader.
        
        Args:
            query: The query to search for papers on arXiv
            max_results: Maximum number of results to return
            load_all_available_meta: Whether to load all available metadata
        """
        from langchain_community.document_loaders import ArxivLoader as LangChainArxivLoader
        self.loader = LangChainArxivLoader(
            query=query,
            load_all_available_meta=load_all_available_meta,
            max_results=max_results
        )
    
    def load(self) -> List[Document]:
        """
        Load documents from arXiv with enhanced metadata.
        
        Returns:
            List of Documents with enhanced metadata
        """
        docs = self.loader.load()
        metadata_manager = DocumentMetadataManager()
        return metadata_manager.standardize_metadata(docs, loader_type="ArxivLoader")


class PubMedLoader:
    """
    Example wrapper around the LangChain PubMedLoader with metadata enhancements.
    This extends the standard LangChain PubMedLoader to provide more consistent metadata.
    
    For an actual implementation, see:
    https://python.langchain.com/docs/integrations/document_loaders/pubmed
    """
    
    def __init__(self, query: str, max_results: int = 10):
        """
        Initialize the PubMedLoader.
        
        Args:
            query: The query to search for papers on PubMed
            max_results: Maximum number of results to return
        """
        from langchain_community.document_loaders import PubMedLoader as LangChainPubMedLoader
        self.loader = LangChainPubMedLoader(
            query=query,
            load_max_docs=max_results
        )
    
    def load(self) -> List[Document]:
        """
        Load documents from PubMed with enhanced metadata.
        
        Returns:
            List of Documents with enhanced metadata
        """
        docs = self.loader.load()
        metadata_manager = DocumentMetadataManager()
        return metadata_manager.standardize_metadata(docs, loader_type="PubMedLoader")


def example_arxiv_pubmed_metadata_capture():
    """
    Example function demonstrating how to load and standardize metadata
    from both ArXiv and PubMed documents.
    """
    
    # Load papers from ArXiv
    arxiv_loader = ArxivLoader(query="agentic AI", max_results=5)
    arxiv_docs = arxiv_loader.load()
    
    # Load papers from PubMed
    pubmed_loader = PubMedLoader(query="large language models", max_results=5)
    pubmed_docs = pubmed_loader.load()
    
    # Combine documents and convert to DataFrame
    all_docs = arxiv_docs + pubmed_docs
    metadata_manager = DocumentMetadataManager()
    
    # Convert to DataFrame with all metadata
    scientific_papers_df = metadata_manager.documents_to_dataframe(
        all_docs, 
        include_content=True,
        include_all_metadata=True,
        max_content_length=500
    )
    
    # Save to CSV for analysis
    scientific_papers_df.to_csv("scientific_papers_metadata.csv", index=False)
    
    # Create a summary DataFrame with just the key columns
    summary_df = scientific_papers_df[['title', 'author', 'publication_date', 
                                       'journal', 'url', 'loader_type']]
    
    return scientific_papers_df, summary_df


def example_ragas_testset_from_scientific_papers():
    """
    Example function demonstrating how to use the RAGAS testset generator
    with scientific papers and then convert the results to a DataFrame.
    """
    # Import necessary components
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Load documents
    arxiv_loader = ArxivLoader(query="agentic AI", max_results=10)
    docs = arxiv_loader.load()
    
    # Initialize RAGAS components
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Generate testset
    generator = TestsetGenerator(llm=llm, embedding_model=emb)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=5)
    
    # Convert to DataFrame
    metadata_manager = DocumentMetadataManager()
    testset_df = metadata_manager.ragas_testset_to_dataframe(dataset)
    
    # Save to CSV
    testset_df.to_csv("ragas_testset_scientific_papers.csv", index=False)
    
    return testset_df


def main():
    """
    Main function demonstrating the usage of the DocumentMetadataManager.
    """
    # Load PDF documents with PyPDFDirectoryLoader
    from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
    
    # Setup loader
    loader = PyPDFDirectoryLoader(
        RAW_DIR,              # directory path from environment variable
        glob="*.pdf",         # file pattern
        silent_errors=True,   # skip unreadable files
    )
    
    # Load documents
    docs = loader.load()
    print(f"Loaded {len(docs)} pages across all PDFs")
    
    # Create metadata manager
    metadata_manager = DocumentMetadataManager()
    
    # Convert to DataFrame
    docs_df = metadata_manager.documents_to_dataframe(
        docs, 
        loader_type="PyPDFDirectoryLoader",
        include_content=True,
        include_all_metadata=True
    )
    
    # Save DataFrame to CSV
    output_path = os.path.join(PROCESSED_DIR, "pdf_documents_metadata.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    docs_df.to_csv(output_path, index=False)
    
    # Display summary
    print(f"Saved metadata for {len(docs_df)} documents to CSV")
    print(f"Output path: {output_path}")
    print(f"Columns in DataFrame: {', '.join(docs_df.columns)}")
    
    return docs_df


if __name__ == "__main__":
    main()
