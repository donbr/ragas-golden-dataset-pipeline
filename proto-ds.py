"""
Final Enhanced RAG Evaluation Pipeline with Multiple Retrievers

This production-ready pipeline evaluates RAG systems using Ragas metrics across different retriever strategies
with comprehensive configuration, error handling, and reporting capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import tempfile
import os

import prefect
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.filesystems import LocalFileSystem
from prefect.cache_policies import NO_CACHE

# Ragas and LangChain imports
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from datasets import load_dataset
from ragas import EvaluationDataset
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers import (
    MultiQueryRetriever,
    ParentDocumentRetriever,
    EnsembleRetriever
)
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

class RetrieverEvaluationConfig:
    """Centralized configuration for RAG evaluation pipeline
    
    Attributes:
        evaluator_llm: LLM wrapper for evaluation (default: gpt-4)
        generator_llm: LLM for answer generation (default: gpt-3.5-turbo)
        run_config: Ragas RunConfig for evaluation
        metrics: List of Ragas metrics to compute
        chunk_size: Document chunk size in characters
        chunk_overlap: Document chunk overlap in characters
        retrieval_k: Number of documents to retrieve
        golden_dataset_name: Name of HuggingFace dataset to use
    """
    def __init__(
        self,
        evaluator_llm: Any = None,
        generator_llm: Any = None,
        run_config: RunConfig = None,
        metrics: List[Any] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 3,
        golden_dataset_name: str = "dwb2023/ragas-golden-dataset-v2"
    ):
        # LLM configurations
        self.evaluator_llm = evaluator_llm or LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4", temperature=0)
        )
        self.generator_llm = generator_llm or ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0
        )
        
        # Evaluation configuration
        # self.run_config = run_config or RunConfig(
        self.run_config = RunConfig(            
            timeout=300,
            max_retries=15,
            max_wait=90,
            max_workers=8,
            log_tenacity=True
        )
        
        # Metrics to compute
        self.metrics = metrics or [
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall(),
            NoiseSensitivity()
        ]
        
        # Document processing config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.golden_dataset_name = golden_dataset_name

@task(
    name="load-golden-dataset",
    description="Load golden dataset for RAG evaluation",
    retries=3,
    retry_delay_seconds=10,
    tags=["data-loading"]
)
def load_golden_dataset(dataset_name: str) -> Dict[str, Any]:
    """Load the golden dataset for evaluation"""
    logger = get_run_logger()
    logger.info(f"Loading golden dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
        return {
            "questions": dataset["user_input"],
            "contexts": dataset["reference_contexts"],
            "answers": dataset["reference"],
            "dataset": dataset
        }
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

@task(
    name="load-documents",
    description="Load and process source documents",
    retries=2,
    retry_delay_seconds=5,
    tags=["data-processing"]
)
def load_and_process_documents(
    document_path: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[Any]:
    """Load and split documents into chunks"""
    logger = get_run_logger()
    logger.info(f"Loading documents from {document_path}")
    
    try:
        # Use DirectoryLoader if path is a directory
        if Path(document_path).is_dir():
            loader = DirectoryLoader(document_path)
        else:
            loader = TextLoader(document_path)
            
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        logger.info(f"Split documents into {len(splits)} chunks")
        return splits
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise

def batch_documents(documents, batch_size=500):
    """Split a list of documents into batches of specified size"""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

@task(
    name="create-retrievers",
    description="Create different retriever types for evaluation",
    tags=["retriever-setup"]
)
def create_retrievers(
    document_splits: List[Any],
    retrieval_k: int,
    llm: Any = None
) -> Dict[str, Any]:
    """Create various retriever types for comparative evaluation"""
    logger = get_run_logger()
    
    # Create separate temp directories for each embedding model
    openai_dir = tempfile.mkdtemp(prefix="chroma_openai_")
    logger.info(f"Created OpenAI embeddings Chroma directory: {openai_dir}")
    
    # Create vector store with OpenAI embeddings in its own directory
    embedding_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=document_splits,
        embedding=embedding_model,
        persist_directory=openai_dir
    )
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(
        document_splits,
        k=retrieval_k
    )
    
    # Create various retrievers
    retrievers = {
        "naive": vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}
        ),
        "bm25": bm25_retriever,
        # "multi_query": MultiQueryRetriever.from_llm(
        #     retriever=vectorstore.as_retriever(search_kwargs={"k": retrieval_k}),
        #     llm=llm
        # ),
        # "ensemble": EnsembleRetriever(
        #     retrievers=[
        #         vectorstore.as_retriever(search_kwargs={"k": retrieval_k}),
        #         bm25_retriever
        #     ],
        #     weights=[0.5, 0.5]
        #),
    }
    
    # Add semantic retriever with completely separate Chroma instance
    # try:
    #     # Create a separate directory for HuggingFace embeddings
    #     hf_dir = tempfile.mkdtemp(prefix="chroma_huggingface_")
    #     logger.info(f"Created HuggingFace embeddings Chroma directory: {hf_dir}")
        
    #     hf_embeddings = HuggingFaceEmbeddings(
    #         model_name="sentence-transformers/all-mpnet-base-v2"
    #     )
    #     semantic_vectorstore = Chroma.from_documents(
    #         documents=document_splits,
    #         embedding=hf_embeddings,
    #         persist_directory=hf_dir
    #     )
    #     retrievers["semantic"] = semantic_vectorstore.as_retriever(
    #         search_kwargs={"k": retrieval_k}
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to create semantic retriever: {str(e)}")
    
    # Add contextual compression if Cohere available
    # try:
    #     if CohereRerank is not None:
    #         compressor = CohereRerank(model="rerank-english-v3.0")
    #         retrievers["contextual_compression"] = ContextualCompressionRetriever(
    #             base_compressor=compressor,
    #             base_retriever=vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
    #         )
    #     else:
    #         logger.warning("CohereRerank not available - skipping contextual compression retriever")
    # except ImportError:
    #     logger.warning("Cohere not available - skipping contextual compression retriever")
    
    # # Add parent document retriever with explicit document addition
    # try:
    #     parent_vs_dir = tempfile.mkdtemp(prefix="chroma_parent_")
    #     parent_vs = Chroma(
    #         collection_name="parent_documents",
    #         embedding_function=embedding_model,
    #         persist_directory=parent_vs_dir
    #     )
        
    #     doc_store = InMemoryStore()
    #     parent_retriever = ParentDocumentRetriever(
    #         vectorstore=parent_vs,
    #         docstore=doc_store,
    #         child_splitter=RecursiveCharacterTextSplitter(
    #             chunk_size=200
    #         )
    #     )
        
    #     # Add documents in smaller batches to avoid "batch size too large" error
    #     logger.info(f"Adding {len(document_splits)} documents to parent document retriever in batches")
    #     for i, batch in enumerate(batch_documents(document_splits, batch_size=100)):
    #         logger.info(f"Adding batch {i+1} to parent document retriever")
    #         parent_retriever.add_documents(batch)
        
    #     retrievers["parent_document"] = parent_retriever
    # except Exception as e:
    #     logger.error(f"Error creating parent document retriever: {str(e)}")
    
    logger.info(f"Created {len(retrievers)} retriever types")
    return retrievers

@task(
    name="generate-responses",
    description="Generate responses for evaluation",
    retries=2,
    retry_delay_seconds=10,
    # disable caching,
    cache_policy=NO_CACHE,
    tags=["response-generation"]
)
def generate_responses(
    retriever: Any,
    retriever_name: str,
    questions: List[str],
    llm: Any
) -> Dict[str, Any]:
    """Generate responses using a specific retriever"""
    logger = get_run_logger()
    
    prompt_template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answers = []
    contexts = []
    
    for question in questions:
        try:
            answer = chain.invoke(question)
            # Updated to use invoke() instead of get_relevant_documents()
            retrieved_docs = retriever.invoke(question)
            context_texts = [doc.page_content for doc in retrieved_docs]
            
            answers.append(answer)
            contexts.append(context_texts)
        except Exception as e:
            logger.warning(f"Error processing question '{question[:50]}...': {str(e)}")
            answers.append("")
            contexts.append([])
    
    return {
        "retriever_name": retriever_name,
        "questions": questions,
        "answers": answers,
        "contexts": contexts
    }

@task(
    name="create-evaluation-dataset",
    description="Create Ragas evaluation dataset",
    tags=["data-preparation"]
)
def create_evaluation_dataset(
    response_data: Dict[str, Any],
    golden_answers: List[str]
) -> EvaluationDataset:
    """Create Ragas EvaluationDataset from response data"""
    evaluation_samples = []
    
    for i in range(len(response_data["questions"])):
        sample = {
            "user_input": response_data["questions"][i],
            "response": response_data["answers"][i],
            "retrieved_contexts": response_data["contexts"][i],
            "reference": golden_answers[i]
        }
        evaluation_samples.append(sample)
    
    return EvaluationDataset.from_list(evaluation_samples)

@task(
    name="evaluate-retriever",
    description="Evaluate retriever using Ragas metrics",
    retries=3,
    retry_delay_seconds=30,
    cache_policy=NO_CACHE,
    timeout_seconds=600,
    tags=["evaluation"]
)
def evaluate_retriever(
    dataset: EvaluationDataset,
    config: RetrieverEvaluationConfig
) -> Dict[str, float]:
    """Evaluate a retriever using Ragas metrics"""
    logger = get_run_logger()
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=config.metrics,
            llm=config.evaluator_llm,
            run_config=config.run_config
        )
        
        scores = {metric.name: result[metric.name] for metric in config.metrics}
        logger.info(f"Evaluation completed with scores: {scores}")
        return scores
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

@task(
    name="generate-comparison-report",
    description="Generate comparative evaluation report",
    tags=["reporting"]
)
def generate_comparison_report(
    evaluation_results: Dict[str, Dict[str, float]],
    output_path: str = None
) -> Dict[str, Any]:
    """Generate comparative report of all retriever evaluations"""
    logger = get_run_logger()
    
    # Create DataFrame for analysis
    df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    
    # Calculate overall score (average of all metrics)
    df['overall_score'] = df.mean(axis=1)
    df = df.sort_values('overall_score', ascending=False)
    
    # Generate markdown report
    report_md = "# RAG Retriever Evaluation Report\n\n"
    report_md += "## Overall Performance\n"
    report_md += df[['overall_score']].to_markdown() + "\n\n"
    
    report_md += "## Detailed Metrics\n"
    report_md += df.drop(columns=['overall_score']).to_markdown() + "\n\n"
    
    # Create artifacts
    create_markdown_artifact(
        key="retriever-comparison-report",
        markdown=report_md,
        description="Comparative evaluation of different retriever types"
    )
    
    create_table_artifact(
        key="retriever-metrics-table",
        table=df.reset_index().rename(columns={'index': 'retriever'}).to_dict('records'),
        description="Detailed metrics by retriever type"
    )
    
    # Save to file if requested
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        df.to_csv(output_dir / "retriever_comparison.csv")
        
        # Save full report
        with open(output_dir / "evaluation_report.md", "w") as f:
            f.write(report_md)
        
        logger.info(f"Saved report to {output_path}")
    
    return {
        "dataframe": df,
        "markdown": report_md,
        "output_path": output_path
    }

@flow(
    name="retriever-evaluation-flow",
    description="Evaluate a single retriever type",
    task_runner=ConcurrentTaskRunner(),
    timeout_seconds=1800
)
def evaluate_retriever_flow(
    retriever: Any,
    retriever_name: str,
    golden_data: Dict[str, Any],
    config: RetrieverEvaluationConfig
) -> Dict[str, Any]:
    """Complete evaluation flow for a single retriever type"""
    logger = get_run_logger()
    
    # Generate responses
    response_data = generate_responses(
        retriever=retriever,
        retriever_name=retriever_name,
        questions=golden_data["questions"],
        llm=config.generator_llm
    )
    
    # Create evaluation dataset
    eval_dataset = create_evaluation_dataset(
        response_data=response_data,
        golden_answers=golden_data["answers"]
    )
    
    # Evaluate with Ragas
    scores = evaluate_retriever(
        dataset=eval_dataset,
        config=config
    )
    
    return {
        "retriever_name": retriever_name,
        "scores": scores,
        "sample_size": len(golden_data["questions"])
    }

@flow(
    name="rag-retriever-comparison-flow",
    description="Compare multiple RAG retriever strategies",
    task_runner=ConcurrentTaskRunner(),
    timeout_seconds=7200,
    log_prints=True
)
def rag_retriever_comparison_flow(
    document_path: str,
    output_dir: str = "output/retriever_comparison",
    config: RetrieverEvaluationConfig = None
) -> Dict[str, Any]:
    """Main flow to compare multiple RAG retriever strategies"""
    logger = get_run_logger()
    config = config or RetrieverEvaluationConfig()
    
    # Load golden dataset
    golden_data = load_golden_dataset(config.golden_dataset_name)
    
    # Process documents
    document_splits = load_and_process_documents(
        document_path=document_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Create retrievers
    retrievers = create_retrievers(
        document_splits=document_splits,
        retrieval_k=config.retrieval_k,
        llm=config.generator_llm
    )
    
    # Evaluate all retrievers
    evaluation_results = {}
    for retriever_name, retriever in retrievers.items():
        logger.info(f"Evaluating {retriever_name} retriever")
        try:
            result = evaluate_retriever_flow(
                retriever=retriever,
                retriever_name=retriever_name,
                golden_data=golden_data,
                config=config
            )
            evaluation_results[retriever_name] = result["scores"]
        except Exception as e:
            logger.error(f"Failed to evaluate {retriever_name}: {str(e)}")
            evaluation_results[retriever_name] = {"error": str(e)}
    
    # Generate comparison report
    report = generate_comparison_report(
        evaluation_results=evaluation_results,
        output_path=output_dir
    )
    
    # Final summary
    create_markdown_artifact(
        key="evaluation-summary",
        markdown=f"""
        # RAG Retriever Evaluation Complete
        
        Evaluated {len(retrievers)} retriever types using {len(golden_data['questions'])} questions.
        
        Results saved to: {output_dir}
        """,
        description="Final evaluation summary"
    )
    
    return {
        "evaluation_results": evaluation_results,
        "report": report,
        "config": vars(config)
    }

if __name__ == "__main__":
    # Example configuration with custom settings
    config = RetrieverEvaluationConfig(
        evaluator_llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0)),
        generator_llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0),
        run_config=RunConfig(
            timeout=400,
            max_retries=20,
            max_wait=120,
            max_workers=6
        ),
        chunk_size=600,
        chunk_overlap=100,
        retrieval_k=4
    )
    
    # Run the comparison flow
    results = rag_retriever_comparison_flow(
        document_path="data/raw",
        output_dir="results/retriever_comparison",
        config=config
    )