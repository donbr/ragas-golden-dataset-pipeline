#!/usr/bin/env python3
"""
Test script for the /retrieve compatibility endpoint.
This script tests all retriever types and analyzes the response structure in detail.
"""

import json
import sys
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration
BASE_URL = "http://127.0.0.1:8000/retrieve"
RETRIEVER_TYPES = [
    "bm25",
    "dense",
    "hybrid",
    "semantic",
    "naive",
    "contextual_compression",
    "multi_query",
    "parent_document",
    "ensemble"
]
TEST_QUESTIONS = [
    "Did people generally like John Wick?",
    "Do any reviews have a rating of 10? If so - can I have the URLs to those reviews?",
    "What happened in John Wick?"
]

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m"
}

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Filename for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, f"retrieve_endpoint_test_{timestamp}.json")


def color_print(text: str, color: str = "RESET") -> None:
    """Print colored text to the terminal."""
    if sys.platform == "win32" and os.environ.get("TERM") != "xterm":
        # Windows without proper terminal, don't use colors
        print(text)
    else:
        print(f"{COLORS.get(color, COLORS['RESET'])}{text}{COLORS['RESET']}")


def test_retriever(retriever_type: str, question: str, top_k: int = 5) -> Dict[str, Any]:
    """Test a specific retriever with a question and return the results."""
    
    color_print(f"\n{'=' * 40}", "CYAN")
    color_print(f"Testing: {retriever_type}", "MAGENTA")
    color_print(f"Question: {question}", "CYAN")
    
    result = {
        "retriever_type": retriever_type,
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "status": "ERROR",
        "error": None,
        "metadata": None,
        "document_count": 0,
        "documents": [],
        "execution_time_ms": 0
    }
    
    start_time = time.time()
    
    try:
        # Prepare the request payload
        payload = {
            "query": question,
            "retriever_type": retriever_type,
            "top_k": top_k
        }
        
        # Send the request
        response = requests.post(
            BASE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # 60 second timeout
        )
        
        # Record execution time
        result["execution_time_ms"] = round((time.time() - start_time) * 1000)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Process successful response
            data = response.json()
            
            # Store response data in result
            result["status"] = "SUCCESS"
            result["metadata"] = data.get("metadata", {})
            result["document_count"] = len(data.get("documents", []))
            result["documents"] = data.get("documents", [])
            
            # Print success information
            color_print("STATUS: SUCCESS", "GREEN")
            color_print(f"Document Count: {result['document_count']}", "GREEN")
            color_print(f"Metadata: {json.dumps(result['metadata'], indent=2)}", "CYAN")
            
            # Check for empty documents
            if result["document_count"] == 0:
                color_print("WARNING: No documents returned", "YELLOW")
            else:
                # Print first document details
                first_doc = result["documents"][0]
                color_print("First Document:", "CYAN")
                color_print(f"  ID: {first_doc.get('id', 'N/A')}", "CYAN")
                color_print(f"  Score: {first_doc.get('score', 'N/A')}", "CYAN")
                content_preview = first_doc.get('content', '')[:100] + "..." if len(first_doc.get('content', '')) > 100 else first_doc.get('content', '')
                color_print(f"  Content (preview): {content_preview}", "CYAN")
                
                # Check if scores are populated correctly
                if first_doc.get('score', 0) == 0:
                    color_print("WARNING: Document score is 0", "YELLOW")
                
                # Additional analysis of metadata
                if "metadata" in first_doc and first_doc["metadata"]:
                    color_print(f"  Metadata: {json.dumps(first_doc['metadata'], indent=2)}", "CYAN")
        else:
            # Process error response
            result["status"] = "ERROR"
            result["error"] = f"HTTP Status: {response.status_code}, Response: {response.text}"
            
            color_print(f"STATUS: ERROR (HTTP {response.status_code})", "RED")
            color_print(f"Error details: {response.text}", "RED")
    except Exception as e:
        # Process exception
        result["status"] = "ERROR"
        result["error"] = f"Exception: {str(e)}"
        result["execution_time_ms"] = round((time.time() - start_time) * 1000)
        
        color_print("STATUS: ERROR (Exception)", "RED")
        color_print(f"Error details: {str(e)}", "RED")
    
    return result


def main():
    """Main function to run the tests."""
    
    color_print(f"\n=== Starting Retriever API Tests ===", "MAGENTA")
    print(f"Test run started at: {datetime.now().isoformat()}")
    
    # Store all results
    all_results = []
    successful_retrievers = []
    failed_retrievers = []
    empty_document_retrievers = []
    
    # Test each retriever with the first question only for faster testing
    question = TEST_QUESTIONS[0]
    
    for retriever_type in RETRIEVER_TYPES:
        result = test_retriever(retriever_type, question)
        all_results.append(result)
        
        # Categorize results
        if result["status"] == "SUCCESS":
            successful_retrievers.append(retriever_type)
            if result["document_count"] == 0:
                empty_document_retrievers.append(retriever_type)
        else:
            failed_retrievers.append(retriever_type)
    
    # Write results to log file
    with open(LOG_FILE, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "total_retrievers": len(RETRIEVER_TYPES),
                "successful_retrievers": successful_retrievers,
                "failed_retrievers": failed_retrievers,
                "empty_document_retrievers": empty_document_retrievers
            }
        }, f, indent=2)
    
    # Print summary
    color_print("\n=== Test Summary ===", "MAGENTA")
    color_print(f"Tests ran at: {datetime.now().isoformat()}", "CYAN")
    color_print(f"Log file: {LOG_FILE}", "CYAN")
    
    color_print(f"\nSuccessful Retrievers ({len(successful_retrievers)}):", "GREEN")
    for retriever in successful_retrievers:
        color_print(f"- {retriever}", "GREEN")
    
    if empty_document_retrievers:
        color_print(f"\nRetrievers with Empty Results ({len(empty_document_retrievers)}):", "YELLOW")
        for retriever in empty_document_retrievers:
            color_print(f"- {retriever}", "YELLOW")
    
    if failed_retrievers:
        color_print(f"\nFailed Retrievers ({len(failed_retrievers)}):", "RED")
        for retriever in failed_retrievers:
            color_print(f"- {retriever}", "RED")
    
    color_print(f"\nTests completed. See log file for details: {LOG_FILE}", "CYAN")


if __name__ == "__main__":
    main() 