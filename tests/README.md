# Testing the Retriever API Compatibility Layer

This directory contains scripts and tools for testing the compatibility layer between the simulated retriever API and the actual LangChain retriever implementation.

## Available Tests

1. **PowerShell Test Script**: `test_retrieve_endpoint.ps1`
2. **Python Test Script**: `test_retrieve_endpoint.py`
3. **Original API Test Script**: `test_api_endpoints.ps1` (tests the LangChain `/invoke` endpoints)

## Prerequisites

- The FastAPI server must be running
- For Python tests: Python 3.7+ with the `requests` package installed

## Running the Tests

### PowerShell Test

```powershell
cd tests
.\test_retrieve_endpoint.ps1
```

This script will:
- Test all retriever types with a sample question
- Display colored output showing success/failure status
- Generate a detailed log file in the `logs` directory
- Provide a summary of successful and failed retrievers

### Python Test

```bash
cd tests
python test_retrieve_endpoint.py
```

This script provides more detailed analysis including:
- Testing all retriever types with a sample question
- Detailed inspection of document structure and scores
- JSON log file with complete test results
- Color-coded terminal output

## Test Result Interpretation

### Success Criteria

A successful test meets the following criteria:
- HTTP status code 200
- Non-empty documents array
- Each document contains:
  - A unique ID
  - Non-empty content
  - A non-zero score
  - Appropriate metadata

### Common Issues

1. **Empty Document Lists**: 
   - Check that the chain is returning documents
   - Verify that document extraction logic is finding documents in the response

2. **Missing Scores**:
   - Check if the underlying retriever provides score information
   - Verify the score extraction logic is checking all possible metadata keys

3. **Retriever Not Found**:
   - Ensure the retriever type is correctly mapped in the `RETRIEVER_MAPPING` dictionary

## Log Files

Test logs are stored in the `logs` directory with timestamps in their filenames:
- PowerShell logs: `retriever_test_YYYYMMDD_HHMMSS.log`
- Python logs: `retrieve_endpoint_test_YYYYMMDD_HHMMSS.json`

These logs contain detailed information about each test run and can be used for troubleshooting and performance analysis.

## Comparing with Simulated API

To compare the compatibility layer with the original simulated API:

1. Run tests against the original simulated API
2. Run the same tests against our compatibility layer
3. Compare the response formats, document counts, and scores
4. Verify that downstream applications function correctly with our compatibility layer 