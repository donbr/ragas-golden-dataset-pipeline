# Migration Assessment: Pipeline Retriever API to LangChain Implementation

## Current System Analysis

The current system uses a simulation-based approach with:

**pipeline_retriever_api.py**:
- FastAPI server simulating different retriever types
- Single `/retrieve` endpoint that accepts a `retriever_type` parameter
- Returns document objects with scores and detailed metadata
- Supports 9 retriever types including BM25, dense, hybrid, and specialized variations

**libs/evaluation_retrieval/api.py**:
- Contains Prefect tasks to manage the API server lifecycle
- Launches pipeline_retriever_api.py as a subprocess
- Sends evaluation requests to the API server
- Processes responses for evaluation metrics

## Proposed System Analysis

The alternative implementation uses real LangChain retrievers:

**main_api.py**:
- FastAPI server with actual LangChain retrieval chains
- Multiple endpoints (`/invoke/{retriever_type}`)
- Returns answers rather than raw documents
- Contains lifecycle management for retriever initialization

**run.py**:
- Entry point script that starts the FastAPI server
- Configures logging and handles shutdown

## Interface Compatibility Challenges

| Challenge | Details | Difficulty |
|-----------|---------|------------|
| Endpoint Pattern | Need to bridge single vs. multiple endpoint patterns | Medium |
| Response Structure | Current: document objects with scores<br>Proposed: answers with minimal document information | High |
| Request Format | Current: single JSON body with retriever_type<br>Proposed: Endpoint-specific parameters | Medium |
| Retriever Mapping | Need to map between 9 simulated and 7 actual retrievers | Low |

## Implementation Strategy

1. **Create a Compatibility Layer**
   - Add a `/retrieve` endpoint to main_api.py that:
     - Maps the `retriever_type` parameter to the appropriate endpoint
     - Formats the response to match the current expected structure
   - Include document scores and metadata in responses

2. **Server Management Updates**
   - Modify `prepare_api_server()` to launch run.py instead
   - Update health check logic to work with new API

3. **Response Transformation**
   - Extract context documents from LangChain responses
   - Add scoring information (potentially using the real relevance calculations)
   - Maintain backward compatibility in metadata format

4. **Retriever Mapping Table**
   ```
   pipeline_retriever_api.py  →  main_api.py
   -----------------------------------------
   bm25                      →  bm25_retriever
   dense                     →  semantic_retriever
   hybrid                    →  ensemble_retriever
   semantic                  →  semantic_retriever
   naive                     →  naive_retriever
   contextual_compression    →  contextual_compression_retriever
   multi_query               →  multi_query_retriever
   parent_document           →  parent_document_retriever
   ensemble                  →  ensemble_retriever
   ```

## Technical Feasibility Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| API Compatibility | Medium | Requires new endpoint that bridges the pattern difference |
| Response Format | High | Significant transformation needed for maintaining backward compatibility |
| Launch Mechanism | Low | Simple update to subprocess command |
| Retriever Mapping | Low | Direct mapping exists for all retrievers |
| Testing Complexity | Medium | Need to validate real vs simulated retriever behavior |
| Overall Complexity | Medium | Main challenge is in response format compatibility |

## Migration Plan

1. **Phase 1: Preparation**
   - Add compatibility endpoint to main_api.py
   - Create response transformation logic

2. **Phase 2: Server Management**
   - Update prepare_api_server to support both implementations
   - Add configuration option to choose which implementation to use

3. **Phase 3: Testing**
   - Run parallel evaluations with both implementations
   - Compare metrics and validate equivalence

4. **Phase 4: Switchover**
   - Move entirely to the LangChain implementation
   - Remove compatibility layer once metrics are updated

## Implementation Details

The compatibility layer has been successfully implemented in the `main_api.py` file. Below are the key components of the implementation:

### 1. Compatibility Models

Pydantic models have been created to match the interface of the simulated retriever API:

- `RetrievalRequest`: Defines the expected request format with query, retriever_type, and top_k parameters
- `Document`: Represents retrieved documents with id, content, score, and metadata fields
- `RetrievalResponse`: Defines the expected response format with query, retriever_type, documents, and metadata fields

### 2. Retriever Type Mapping

A mapping dictionary (`RETRIEVER_MAPPING`) has been created to translate between the simulated retriever types and the actual LangChain retriever endpoints:

```python
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
```

### 3. Compatibility Endpoint

A new `/retrieve` endpoint has been added that:

1. Accepts the simulated API request format
2. Maps the retriever type to the appropriate LangChain chain
3. Invokes the appropriate chain with the query
4. Extracts documents from various possible response structures
5. Formats the response to match the expected API format, including:
   - Adding document IDs
   - Transferring or generating relevance scores
   - Preserving metadata
   - Adding execution metadata (latency, document count, etc.)

### 4. Document Extraction Logic

The implementation handles the variability in how different chains store retrieved documents by checking multiple possible locations:

- `source_documents`
- `context_documents`
- `context` (as a list of documents)
- `documents`
- `response.context`
- Any other list-type field containing objects with `page_content` attribute

### 5. Score Handling

The implementation addresses the scoring differences by:

1. Looking for scores in various metadata keys (`score`, `similarity`, `relevance_score`, `relevancy_score`)
2. Preserving the original scores when available
3. Generating fallback position-based scores when no scores are available

### 6. Testing

Two testing scripts have been created to validate the compatibility layer:

1. `tests/test_retrieve_endpoint.ps1`: A PowerShell script for testing all retriever types
2. `tests/test_retrieve_endpoint.py`: A Python script providing more detailed analysis of the response structure

The tests verify that all retriever types can be invoked through the compatibility layer and that the responses match the expected format.

## Conclusion

The migration from simulated retrievers to actual LangChain retrievers is feasible with medium complexity. The primary challenges are in request/response format compatibility rather than core functionality. The benefit of using real retrievers outweighs the migration effort, as it will provide more accurate and meaningful evaluation metrics.

## REVIEWERS ANALYSIS

The migration plan provides a solid foundation for transitioning from simulated retrievers to actual LangChain implementations. However, several areas require further consideration:

1. **Response Transformation Complexity**:
   - The plan underestimates the complexity of transforming LangChain responses to match the expected format.
   - LangChain retrievers don't natively provide the same scoring metrics as the simulation.
   - Based on LangChain documentation, adding scores to retriever results requires custom implementation through a wrapper function or subclassing.
   - Consider implementing a scoring adapter to generate comparable relevance scores.

2. **Testing Strategy Enhancement**:
   - The parallel evaluation phase should include specific metrics to determine equivalence.
   - Define acceptance criteria for when the migration can be considered successful.
   - Add regression tests that compare responses from both implementations for identical queries.
   - Implement automated testing to verify response format compatibility.

3. **Performance Considerations**:
   - The current plan doesn't address potential performance differences between simulated and real retrievers.
   - LangChain retrievers may have different latency characteristics, especially for complex types.
   - Add performance benchmarking before, during, and after migration.
   - Define acceptable performance thresholds for the new implementation.

4. **Error Handling Alignment**:
   - Ensure error cases and edge conditions are handled consistently between implementations.
   - Document how error responses should be standardized across both systems.
   - Implement error mapping in the compatibility layer to preserve error semantics.

5. **Configuration Management**:
   - Add details about how retriever-specific configurations will be maintained and migrated.
   - Consider implementing a configuration validation step during initialization.
   - Document configuration differences between simulated and real retrievers.

6. **Documentation Requirements**:
   - Include API documentation updates in the migration plan.
   - Ensure client implementations are informed about any subtle behavioral differences.
   - Document new capabilities enabled by the LangChain implementation.

7. **Canary Release and Rollback Strategy**:
   - Implement a canary release approach to gradually route traffic to the new implementation.
   - Define clear criteria and process for rolling back if issues are encountered.
   - Consider implementing a feature flag system to toggle between implementations.
   - Set up monitoring alerts to detect issues quickly during the migration.

This migration represents a significant architectural change that affects core evaluation functionality. A phased approach with comprehensive testing is essential to ensure smooth transition with minimal disruption to downstream systems.

