from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import os
import json
import pandas as pd
import prefect
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.artifacts import create_markdown_artifact, create_table_artifact
# Removed CacheHint import due to Prefect version compatibility
import argparse

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
        evaluator_model: str = "gpt-4",
        generator_model: str = "gpt-3.5-turbo",
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

@task(name="load-golden-dataset", retries=3, retry_delay_seconds=5,
      cache_key_fn=lambda *args, **kwargs: args[0], cache_expiration=dict(days=1))
def load_golden_dataset(name: str) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"Loading golden dataset: {name}")
    ds = load_dataset(name, split="train")
    return {"questions": ds["user_input"], "contexts": ds.get("reference_contexts", []), "answers": ds["reference"]}

@task(name="load-documents", retries=2, retry_delay_seconds=5,
      cache_key_fn=lambda *args, **kwargs: args[0], cache_expiration=dict(days=7))
def load_and_split_docs(path: str, chunk_size: int, chunk_overlap: int) -> List[Any]:
    logger = get_run_logger()
    if not Path(path).exists():
        logger.error(f"Document path not found: {path}")
        raise FileNotFoundError(f"Path does not exist: {path}")
    loader = DirectoryLoader(path) if Path(path).is_dir() else TextLoader(path)
    docs = loader.load()
    split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = split.split_documents(docs)
    logger.info(f"Loaded and split into {len(chunks)} chunks")
    return chunks

@task(name="create-retrievers", retries=1)
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

@task(name="generate-responses", retries=2, retry_delay_seconds=5)
def generate_responses(retriever: Any, questions: List[str], llm: Any) -> Dict[str, Any]:
    logger = get_run_logger()
    prompt = ChatPromptTemplate.from_template(
        """Answer based on context:\n{context}\nQuestion: {question}"""
    )
    chain = (RunnableParallel({"context": retriever, "question": RunnablePassthrough()}) |
             prompt | llm | StrOutputParser())
    answers, contexts = [], []
    for q in questions:
        try:
            ans = chain.invoke({'context': retriever, 'question': q})
            docs = retriever.get_relevant_documents(q)
            answers.append(ans)
            contexts.append([d.page_content for d in docs])
        except Exception as e:
            logger.warning(f"Query error: {e}")
            answers.append("")
            contexts.append([])
    return {"answers": answers, "contexts": contexts}

@task(name="prepare-eval-dataset")
def prepare_eval_dataset(res: Dict[str, Any], golden: Dict[str, Any]) -> EvaluationDataset:
    if len(res['answers']) != len(golden['questions']):
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

@task(name="evaluate-retriever", retries=2, retry_delay_seconds=10)
def evaluate_retriever(dataset: EvaluationDataset, config: RetrieverEvaluationConfig) -> Dict[str, float]:
    logger = get_run_logger()
    result = evaluate(dataset=dataset, metrics=config.metrics, llm=config.evaluator_llm, run_config=config.run_config)
    scores = {}
    for metric in config.metrics:
        key = metric.__class__.__name__
        scores[key] = float(result.get(key, 0.0))
    return scores

@task(name="report-results")
def report_results(results: Dict[str, Dict[str, float]], output_dir: str) -> str:
    df = pd.DataFrame(results).T
    df['overall'] = df.mean(axis=1)
    df = df.sort_values('overall', ascending=False)
    md = df.reset_index().rename(columns={'index':'retriever'}).to_markdown(index=False)
    create_markdown_artifact(key="comparison_report", markdown="# Comparison Report\n" + md)
    Path(output_dir).mkdir(exist_ok=True)
    csv_path = Path(output_dir) / "comparison.csv"
    df.to_csv(csv_path)
    create_table_artifact(
        key="comparison_table",
        table={"columns": list(df.reset_index().columns), "data": df.reset_index().values.tolist()}
    )
    return str(csv_path)

@flow(name="ragas-retriever-comparison", task_runner=ConcurrentTaskRunner(), timeout_seconds=3600)
def ragas_retriever_comparison(
    document_path: str,
    golden_dataset: str,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """Main flow entrypoint for RAGAS retriever comparison."""
    cfg = RetrieverEvaluationConfig(golden_dataset)
    golden = load_golden_dataset(cfg.golden_dataset)
    splits = load_and_split_docs(document_path, cfg.chunk_size, cfg.chunk_overlap)
    retrievers = create_retrievers(splits, cfg.retrieval_k)

    results = {}
    for name, ret in retrievers.items():
        res = generate_responses(ret, golden['questions'], cfg.generator_llm)
        ds = prepare_eval_dataset(res, golden)
        scores = evaluate_retriever(ds, cfg)
        results[name] = scores

    report_csv = report_results(results, output_dir)
    return {"results": results, "report_csv": report_csv}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS retriever comparison flow.")
    parser.add_argument("--document-path", required=True, help="Path to documents (file or directory)")
    parser.add_argument("--golden-dataset", required=True, help="Hugging Face dataset name for golden Q&A")
    parser.add_argument("--output-dir", default="output", help="Directory to write comparison results")
    args = parser.parse_args()

    result = ragas_retriever_comparison(
        document_path=args.document_path,
        golden_dataset=args.golden_dataset,
        output_dir=args.output_dir
    )
    print(json.dumps(result, indent=2))
