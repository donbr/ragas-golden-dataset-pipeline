"""
API server management and retriever evaluation.

This module provides tasks for managing the API server and
evaluating different retriever implementations.
"""

import os
import time
import subprocess
import requests
from typing import Dict, List, Any, Optional
from datetime import timedelta
from pathlib import Path

from prefect import task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.events import emit_event


@task(
    name="prepare-api-server",
    description="Starts the API server in a subprocess and ensures it's ready",
    retries=3,
    retry_delay_seconds=10,
    tags=["setup", "api"],
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def prepare_api_server(port: int = 8000) -> Dict[str, Any]:
    """
    Starts the API server in a subprocess and verifies it's healthy.
    
    Args:
        port: Port for the API server
        
    Returns:
        Dictionary with API server information (without process object)
    """
    logger = get_run_logger()
    logger.info(f"Preparing API server on port {port}")
    
    # Get path to run.py
    script_dir = Path(__file__).parent.parent.parent  # Go up from libs/evaluation_retrieval
    api_script = script_dir / "run.py"
    
    if not api_script.exists():
        # Try alternative locations
        if Path("run.py").exists():
            api_script = Path("run.py")
        else:
            raise FileNotFoundError("Could not find run.py in expected locations")
    
    # Log the resolved path
    logger.info(f"Using API script at {api_script.resolve()}")
    
    # Start server as subprocess - don't store the process object
    process = subprocess.Popen(
        ["python", str(api_script), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to be ready
    server_url = f"http://localhost:{port}/health"
    max_attempts = 10
    wait_seconds = 2
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Checking if server is ready (attempt {attempt+1}/{max_attempts})")
            response = requests.get(server_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Server is ready at {server_url}")
                emit_event(
                    event="retriever-evaluation/server/started",
                    resource={"prefect.resource.id": f"api-server:{port}"},
                    payload={"port": port, "status": "running", "pid": process.pid}
                )
                # Return JSON-serializable data without the process object
                return {
                    "port": port,
                    "url": f"http://localhost:{port}",
                    "pid": process.pid,
                    "status": "running"
                }
        except Exception as e:
            logger.warning(f"Server not ready yet: {str(e)}")
        
        time.sleep(wait_seconds)
    
    # If we get here, the server failed to start
    try:
        process.terminate()
    except:
        pass
    
    emit_event(
        event="retriever-evaluation/server/failed",
        resource={"prefect.resource.id": f"api-server:{port}"},
        payload={"port": port, "status": "failed"}
    )
    
    raise RuntimeError(f"Failed to start API server after {max_attempts} attempts")


@task(
    name="shutdown-api-server",
    description="Stops the API server subprocess",
    retries=1,
    retry_delay_seconds=5,
    tags=["cleanup", "api"],
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def shutdown_api_server(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stops the API server.
    
    Args:
        server_info: Server information returned by prepare_api_server
        
    Returns:
        Dictionary with shutdown status
    """
    logger = get_run_logger()
    
    port = server_info.get("port", "unknown")
    url = server_info.get("url", f"http://localhost:{port}")
    
    logger.info(f"Shutting down API server on port {port}")
    
    try:
        # Try to use the shutdown endpoint if available
        shutdown_url = f"{url}/shutdown"
        try:
            logger.info(f"Attempting to shut down server via API endpoint: {shutdown_url}")
            response = requests.post(shutdown_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"API server on port {port} shutdown via API endpoint")
                
                emit_event(
                    event="retriever-evaluation/server/stopped",
                    resource={"prefect.resource.id": f"api-server:{port}"},
                    payload={"port": port, "status": "stopped", "method": "api"}
                )
                
                return {"status": "success", "port": port, "method": "api"}
        except Exception as e:
            logger.warning(f"Could not shut down via API endpoint: {str(e)}")
        
        # Try to kill the process by PID or port
        pid = server_info.get("pid")
        if pid:
            try:
                logger.info(f"Attempting to terminate process with PID {pid}")
                if os.name == 'nt':  # Windows
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
                else:  # Unix/Linux
                    subprocess.run(["kill", "-9", str(pid)], check=False)
                
                logger.info(f"Terminated process with PID {pid}")
                
                emit_event(
                    event="retriever-evaluation/server/stopped",
                    resource={"prefect.resource.id": f"api-server:{port}"},
                    payload={"port": port, "status": "stopped", "method": "pid", "pid": pid}
                )
                
                return {"status": "success", "port": port, "method": "pid"}
            except Exception as e:
                logger.warning(f"Could not terminate by PID: {str(e)}")
        
        # Last resort: try to kill by port
        try:
            logger.info(f"Attempting to kill process on port {port}")
            if os.name == 'nt':  # Windows
                # Find PID listening on the port and kill it
                subprocess.run(f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port} ^| findstr LISTENING\') do taskkill /F /PID %a', shell=True)
            else:  # Unix/Linux
                subprocess.run(f"fuser -k {port}/tcp", shell=True)
            
            logger.info(f"Killed process on port {port}")
            
            emit_event(
                event="retriever-evaluation/server/stopped",
                resource={"prefect.resource.id": f"api-server:{port}"},
                payload={"port": port, "status": "stopped", "method": "port"}
            )
            
            return {"status": "success", "port": port, "method": "port"}
        except Exception as e:
            logger.warning(f"Could not kill by port: {str(e)}")
        
        # If we get here, we couldn't shut down the server
        logger.error(f"Failed to shut down API server on port {port}")
        return {"status": "failed", "port": port}
        
    except Exception as e:
        logger.error(f"Error shutting down API server: {str(e)}")
        return {"status": "error", "port": port, "error": str(e)}


@task(
    name="evaluate-retriever",
    description="Evaluates a specific retriever against test questions",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=2,
    retry_delay_seconds=30,
    tags=["evaluation", "retriever", "api"],
    result_serializer="json"  # Use JSON serializer to avoid pickle issues
)
def evaluate_retriever(
    retriever_type: str,
    dataset: Dict[str, Any],
    api_url: str = "http://localhost:8000",
    top_k: int = 5,
    batch_size: int = 10,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates a specific retriever against test questions.
    
    Args:
        retriever_type: Type of retriever to evaluate
        dataset: Test dataset with questions
        api_url: URL of the retriever API server
        top_k: Number of documents to retrieve
        batch_size: Size of evaluation batches
        config: Optional evaluation configuration
        
    Returns:
        Dictionary with evaluation results
    """
    logger = get_run_logger()
    logger.info(f"Starting evaluation of {retriever_type} retriever")
    
    # Validate the dataset structure 
    if not dataset or "dataset" not in dataset or "examples" not in dataset["dataset"]:
        raise ValueError("Invalid dataset format. Expected {'dataset': {'examples': [...]}, ...}")

    # Get the questions from the dataset structure
    questions = dataset["dataset"]["examples"]
    
    # Store retriever-specific information
    retriever_info = {
        "type": retriever_type,
        "top_k": top_k,
        "api_url": api_url
    }
    
    # Add retriever-specific characteristics based on the type
    if retriever_type == "bm25":
        retriever_info["description"] = "Traditional lexical search using BM25 algorithm"
        retriever_info["strength"] = "Good for exact keyword matching, fast, no training required"
        retriever_info["weakness"] = "Doesn't understand semantics, sensitive to vocabulary mismatch"
        retriever_info["best_for"] = "Precise keyword queries, straightforward information retrieval"
    elif retriever_type == "dense":
        retriever_info["description"] = "Neural embedding-based semantic search using dense vectors"
        retriever_info["strength"] = "Good semantic understanding, can find relevant content without exact matches"
        retriever_info["weakness"] = "More computationally expensive, requires training or pre-trained models"
        retriever_info["best_for"] = "Complex questions, conceptual similarity, handling paraphrases"
    elif retriever_type == "hybrid":
        retriever_info["description"] = "Combined approach using both lexical and semantic matching"
        retriever_info["strength"] = "Leverages benefits of both BM25 and dense retrieval"
        retriever_info["weakness"] = "More complex, potentially higher latency"
        retriever_info["best_for"] = "General-purpose retrieval with both keyword and semantic components"
    elif retriever_type == "semantic":
        retriever_info["description"] = "Pure semantic search optimized for meaning rather than keywords"
        retriever_info["strength"] = "High recall for semantically related content"
        retriever_info["weakness"] = "May miss exact matches, slower than lexical search"
        retriever_info["best_for"] = "Open-ended questions, conceptual exploration"
    
    logger.info(f"Evaluating {retriever_type} retriever against {len(questions)} questions")
    
    # Log more detailed information about the retriever
    for key, value in retriever_info.items():
        if key != "type":  # We already logged the type
            logger.info(f"  {key}: {value}")
    
    # Initialize results
    results = {
        "retriever_type": retriever_type,
        "retriever_info": retriever_info,
        "total_questions": len(questions),
        "total_successful": 0,
        "total_failed": 0,
        "latencies_ms": [],
        "document_counts": [],
        "document_scores": [],
        "token_counts": [],
        "detailed_results": []
    }
    
    # Process questions in batches
    ground_truth = dataset.get("ground_truth", {})
    
    # Calculate number of batches
    num_batches = (len(questions) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx}-{end_idx-1})")
        
        # Process each question in the batch
        for idx in range(start_idx, end_idx):
            question = questions[idx]
            question_id = question.get("id", str(idx))
            question_text = question.get("text", question.get("question", ""))
            
            if not question_text:
                logger.warning(f"Question {question_id} has no text, skipping")
                results["total_failed"] += 1
                continue
            
            try:
                # Call retriever API
                response = requests.post(
                    f"{api_url}/retrieve",
                    json={
                        "query": question_text,
                        "retriever_type": retriever_type,
                        "top_k": top_k
                    },
                    timeout=60
                )
                
                # Parse response
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract documents and metadata
                    documents = data.get("documents", [])
                    metadata = data.get("metadata", {})
                    
                    # Extract relevant metrics
                    latency_ms = metadata.get("latency_ms", 0)
                    
                    # Update results
                    results["total_successful"] += 1
                    results["latencies_ms"].append(latency_ms)
                    results["document_counts"].append(len(documents))
                    
                    # Extract document scores
                    doc_scores = [doc.get("score", 0) for doc in documents]
                    results["document_scores"].extend(doc_scores)
                    
                    # Estimate token count (simplified)
                    doc_contents = [doc.get("content", "") for doc in documents]
                    token_count = sum(len(content.split()) for content in doc_contents)
                    results["token_counts"].append(token_count)
                    
                    # Store detailed result
                    results["detailed_results"].append({
                        "question_id": question_id,
                        "question": question_text,
                        "latency_ms": latency_ms,
                        "document_count": len(documents),
                        "avg_score": sum(doc_scores) / len(doc_scores) if doc_scores else 0,
                        "documents": documents
                    })
                else:
                    logger.warning(f"Error for question {question_id}: {response.status_code} - {response.text}")
                    results["total_failed"] += 1
                    
            except Exception as e:
                logger.error(f"Exception for question {question_id}: {str(e)}")
                results["total_failed"] += 1
        
        # Progress report
        completed = min(end_idx, len(questions))
        percentage = (completed / len(questions)) * 100
        logger.info(f"Progress: {completed}/{len(questions)} ({percentage:.1f}%)")
    
    # Calculate aggregate metrics
    results["success_rate"] = (results["total_successful"] / results["total_questions"]) * 100 if results["total_questions"] > 0 else 0
    results["avg_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"]) if results["latencies_ms"] else 0
    results["avg_document_count"] = sum(results["document_counts"]) / len(results["document_counts"]) if results["document_counts"] else 0
    results["avg_document_score"] = sum(results["document_scores"]) / len(results["document_scores"]) if results["document_scores"] else 0
    results["avg_token_count"] = sum(results["token_counts"]) / len(results["token_counts"]) if results["token_counts"] else 0
    
    # Only keep a limited number of detailed results to avoid serialization issues
    if len(results["detailed_results"]) > 50:
        results["detailed_results"] = results["detailed_results"][:50]
        results["detailed_results_truncated"] = True
    
    logger.info(f"Completed evaluation of {retriever_type} retriever")
    logger.info(f"Success rate: {results['success_rate']:.1f}% ({results['total_successful']}/{results['total_questions']})")
    logger.info(f"Average latency: {results['avg_latency_ms']:.2f}ms")
    
    # Emit event
    emit_event(
        event="retriever-evaluation/evaluation/completed",
        resource={"prefect.resource.id": f"retriever:{retriever_type}"},
        payload={
            "retriever_type": retriever_type,
            "success_rate": results["success_rate"],
            "avg_latency_ms": results["avg_latency_ms"],
            "total_questions": results["total_questions"]
        }
    )
    
    return results 