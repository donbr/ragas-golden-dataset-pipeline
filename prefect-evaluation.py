from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import os
import json
import pandas as pd
import prefect
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import NO_CACHE
import argparse
from datetime import timedelta
from io import StringIO
import inspect
import types
import sys

# RAGAS and LangChain imports
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall, Faithfulness, FactualCorrectness,
    ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
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

# Fix for RAGAS StringIO validation error
class EnhancedStringIO(StringIO):
    """A StringIO wrapper that provides the 'text' field required by RAGAS."""
    def __init__(self, content, classifications=None):
        super().__init__(content)
        self.text = content  # Add the required 'text' field
        if classifications:
            self.classifications = classifications

def patch_ragas_output_parser():
    """
    Monkey patch the RAGAS output parser to handle StringIO validation errors.
    This fixes the bug described in https://github.com/explodinggradients/ragas/issues/1831
    """
    logger = get_run_logger()
    logger.info("Applying monkey patch to fix RAGAS StringIO validation error")
    
    try:
        # Import the module containing the problematic parser
        from ragas.prompt.pydantic_prompt import RagasOutputParser
        
        # Store the original method
        original_parse_output_string = RagasOutputParser.parse_output_string
        
        # Define the patched method
        async def patched_parse_output_string(self, output_string, prompt_value, llm, callbacks, retries_left=1):
            try:
                # Try the original method first
                result = await original_parse_output_string(
                    self, output_string, prompt_value, llm, callbacks, retries_left
                )
                
                # If the result is a StringIO object, enhance it
                if isinstance(result, StringIO):
                    logger.info("Detected StringIO result, enhancing with required fields")
                    # Extract classifications if they exist in the output
                    classifications = None
                    try:
                        import json
                        # Try to extract classifications from the output string
                        start_idx = output_string.find('{"classifications":')
                        if start_idx != -1:
                            # Find the matching closing brace
                            brace_count = 0
                            for i in range(start_idx, len(output_string)):
                                if output_string[i] == '{':
                                    brace_count += 1
                                elif output_string[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_str = output_string[start_idx:i+1]
                                        classifications = json.loads(json_str).get("classifications")
                                        break
                    except Exception as e:
                        logger.warning(f"Error extracting classifications: {str(e)}")
                    
                    # Create an enhanced StringIO with the required fields
                    return EnhancedStringIO(output_string, classifications)
                
                return result
            except Exception as e:
                logger.error(f"Error in patched_parse_output_string: {str(e)}")
                # If all else fails, return the original result
                return await original_parse_output_string(
                    self, output_string, prompt_value, llm, callbacks, retries_left
                )
        
        # Apply the patch
        RagasOutputParser.parse_output_string = patched_parse_output_string
        logger.info("Successfully patched RAGAS output parser")
    except Exception as e:
        logger.error(f"Failed to apply RAGAS patch: {str(e)}")

class RetrieverEvaluationConfig:
    """
    Configuration for retriever evaluation.
    """
    def __init__(
        self,
        golden_dataset: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 3,
        evaluator_model: str = "gpt-4.1-mini",
        generator_model: str = "gpt-4.1-mini",
        run_config: Optional[RunConfig] = None,
        metrics: Optional[List[Any]] = None
    ):
        self.golden_dataset = golden_dataset
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=evaluator_model, temperature=0)
        )
        self.generator_llm = ChatOpenAI(model=generator_model, temperature=0)
        self.run_config = run_config or RunConfig(
            timeout=300, max_retries=5, max_wait=60, max_workers=8, log_tenacity=True
        )
        self.metrics = metrics or [
            LLMContextRecall(), Faithfulness(), FactualCorrectness(),
            ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()
        ]

@task(name="load-golden-dataset", retries=3, retry_delay_seconds=5, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_golden_dataset(name: str) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"Loading golden dataset: {name}")
    ds = load_dataset(name, split="train")
    # Safely extract contexts
    contexts = ds["reference_contexts"] if "reference_contexts" in ds.column_names else [[] for _ in range(len(ds))]
    questions = ds["user_input"] if "user_input" in ds.column_names else []
    answers = ds["reference"] if "reference" in ds.column_names else []
    if not questions or not answers:
        logger.error("Golden dataset missing required columns 'user_input' or 'reference'.")
        raise ValueError("Invalid golden dataset format.")
    return {"questions": questions, "contexts": contexts, "answers": answers}

@task(name="load-documents", retries=2, retry_delay_seconds=5, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7))
def load_and_split_docs(path: str, chunk_size: int, chunk_overlap: int) -> List[Any]:
    logger = get_run_logger()
    if not Path(path).exists():
        logger.error(f"Document path not found: {path}")
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    # Use TextLoader explicitly for text files to avoid JSON parsing issues
    if Path(path).is_dir():
        loader = DirectoryLoader(
            path, 
            glob="*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
    else:
        loader = TextLoader(path, autodetect_encoding=True)
    
    docs = loader.load()
    if not docs:
        logger.error("No documents loaded from the provided path.")
        raise ValueError("Document loading failed or empty source.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    logger.info(f"Loaded and split into {len(chunks)} chunks")
    return chunks

@task(name="create-retrievers", retries=1, cache_policy=NO_CACHE)
def create_retrievers(splits: List[Any], k: int) -> Dict[str, Any]:
    logger = get_run_logger()
    emb = OpenAIEmbeddings()
    vs_dir = tempfile.mkdtemp(prefix="chroma_")
    vectorstore = Chroma.from_documents(documents=splits, embedding=emb, persist_directory=vs_dir)
    retrievers = {
        "similarity": vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}),
        "bm25": BM25Retriever.from_documents(splits, k=k)
    }
    logger.info(f"Created retrievers: {list(retrievers.keys())}")
    return retrievers

@task(name="generate-responses", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def generate_responses(retriever: Any, questions: List[str], llm: Any) -> Dict[str, Any]:
    logger = get_run_logger()
    prompt = ChatPromptTemplate.from_template("Answer based on context:\n{context}\nQuestion: {question}")
    chain = RunnableParallel({"context": retriever, "question": RunnablePassthrough()}) | prompt | llm | StrOutputParser()
    answers, contexts = [], []
    for q in questions:
        try:
            # Pass just the question to the chain - retriever is already part of the chain
            ans = chain.invoke(q)
            # Use invoke() instead of deprecated get_relevant_documents()
            docs = retriever.invoke(q)
            answers.append(ans)
            contexts.append([d.page_content for d in docs])
        except Exception as e:
            logger.warning(f"Query error: {e}")
            answers.append("")
            contexts.append([])
    return {"answers": answers, "contexts": contexts}

@task(name="prepare-eval-dataset", cache_policy=NO_CACHE)
def prepare_eval_dataset(res: Dict[str, Any], golden: Dict[str, Any]) -> EvaluationDataset:
    if len(res["answers"]) != len(golden["questions"]):
        raise ValueError("Mismatch between generated answers and golden questions count")
    samples = []
    for i, q in enumerate(golden["questions"]):
        samples.append({
            "user_input": q,
            "response": res["answers"][i],
            "retrieved_contexts": res["contexts"][i],
            "reference": golden["answers"][i]
        })
    return EvaluationDataset.from_list(samples)

@task(name="evaluate-retriever", retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def evaluate_retriever(dataset: EvaluationDataset, config: RetrieverEvaluationConfig) -> Dict[str, float]:
    logger = get_run_logger()
    
    # Apply the monkey patch to fix RAGAS StringIO validation error
    patch_ragas_output_parser()
    
    try:
        # Run the evaluation with our patched RAGAS
        result = evaluate(
            dataset=dataset, 
            metrics=config.metrics, 
            llm=config.evaluator_llm, 
            run_config=config.run_config
        )
        
        # Extract scores from the result - RAGAS returns a DataFrame directly
        # Convert to dict with metric names as keys
        scores = {}
        if hasattr(result, 'to_dict'):
            # If result is a DataFrame or has to_dict method
            result_dict = result.to_dict()
            # Handle first row of results
            if isinstance(result_dict, dict) and result_dict:
                # Get the first record from each metric column
                for metric in config.metrics:
                    metric_name = metric.__class__.__name__
                    if metric_name in result_dict:
                        # Get the first (and typically only) value
                        values = list(result_dict[metric_name].values())
                        if values:
                            scores[metric_name] = float(values[0])
                        else:
                            scores[metric_name] = 0.0
                    else:
                        scores[metric_name] = 0.0
        else:
            # If result has a different structure, try to extract scores directly
            logger.info(f"Result type: {type(result)}")
            for metric in config.metrics:
                metric_name = metric.__class__.__name__
                if hasattr(result, metric_name):
                    scores[metric_name] = float(getattr(result, metric_name))
                else:
                    scores[metric_name] = 0.0
        
        return scores
    except Exception as e:
        logger.error(f"Evaluation failed despite patch: {str(e)}")
        
        # Fallback to individual metric evaluation
        scores = {}
        for metric in config.metrics:
            metric_name = metric.__class__.__name__
            try:
                # Try to evaluate each metric separately
                single_result = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    llm=config.evaluator_llm,
                    run_config=config.run_config
                )
                
                # Handle the result the same way as above
                if hasattr(single_result, 'to_dict'):
                    result_dict = single_result.to_dict()
                    if isinstance(result_dict, dict) and result_dict and metric_name in result_dict:
                        values = list(result_dict[metric_name].values())
                        if values:
                            scores[metric_name] = float(values[0])
                        else:
                            scores[metric_name] = 0.0
                    else:
                        scores[metric_name] = 0.0
                elif hasattr(single_result, metric_name):
                    scores[metric_name] = float(getattr(single_result, metric_name))
                else:
                    scores[metric_name] = 0.0
            except Exception as metric_error:
                logger.warning(f"Error evaluating {metric_name}: {str(metric_error)}")
                scores[metric_name] = 0.0
        
        return scores

@task(name="report-results")
def report_results(results: Dict[str, Dict[str, float]], output_dir: str) -> str:
    df = pd.DataFrame(results).T
    df["overall"] = df.mean(axis=1)
    df = df.sort_values("overall", ascending=False)
    md = df.reset_index().rename(columns={"index":"retriever"}).to_markdown(index=False)
    create_markdown_artifact(key="comparison_report", markdown="# Comparison Report\n" + md)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "comparison.csv"
    df.to_csv(csv_path, index=False)
    create_table_artifact(key="comparison_table", table={"columns": df.reset_index().columns.tolist(), "data": df.reset_index().values.tolist()})
    return str(csv_path)

@flow(name="ragas-retriever-comparison", task_runner=ConcurrentTaskRunner(), timeout_seconds=3600)
def ragas_retriever_comparison(document_path: str, golden_dataset: str, output_dir: str = "output") -> Dict[str, Any]:
    cfg = RetrieverEvaluationConfig(golden_dataset)
    golden = load_golden_dataset(cfg.golden_dataset)
    splits = load_and_split_docs(document_path, cfg.chunk_size, cfg.chunk_overlap)
    retrievers = create_retrievers(splits, cfg.retrieval_k)
    results = {}
    for name, ret in retrievers.items():
        res = generate_responses(ret, golden["questions"], cfg.generator_llm)
        ds = prepare_eval_dataset(res, golden)
        scores = evaluate_retriever(ds, cfg)
        results[name] = scores
    report_csv = report_results(results, output_dir)
    return {"results": results, "report_csv": report_csv}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS retriever comparison flow.")
    parser.add_argument("--document-path", required=False, help="Path to documents or omit to use dataset contexts.")
    parser.add_argument("--golden-dataset", required=True, help="HuggingFace dataset name for golden Q&A")
    parser.add_argument("--output-dir", default="output", help="Directory for results")
    args = parser.parse_args()
    document_path = args.document_path
    if not document_path:
        golden = load_golden_dataset(args.golden_dataset)
        flat_contexts = []
        for lst in golden["contexts"]:
            flat_contexts.extend(lst if isinstance(lst, list) else [lst])
        document_path = tempfile.mkdtemp(prefix="ragas_ctx_")
        for i, txt in enumerate(flat_contexts):
            # Ensure txt is a string to avoid JSON parsing issues
            if not isinstance(txt, str):
                txt = str(txt)
            with open(Path(document_path)/f"doc_{i}.txt", "w", encoding="utf-8") as f:
                f.write(txt)
    result = ragas_retriever_comparison(document_path=document_path, golden_dataset=args.golden_dataset, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
