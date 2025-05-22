# API Base URL (no /invoke prefix for the compatibility endpoint)
$BaseUrl = "http://127.0.0.1:8000/retrieve"

# Define the retriever types (matching the run.py values)
$RetrieverTypes = @(
    "bm25"
    "dense"
    "hybrid"
    "semantic"
    "naive" 
    "contextual_compression"
    "multi_query"
    "parent_document"
    "ensemble"
)

# Define the questions
$Questions = @(
    "Did people generally like John Wick?"
    "Do any reviews have a rating of 10? If so - can I have the URLs to those reviews?"
    "What happened in John Wick?"
)

# Create logs directory if it doesn't exist
$LogsDir = "logs"
if (-not (Test-Path -Path $LogsDir)) {
    New-Item -Path $LogsDir -ItemType Directory | Out-Null
}

# Define colors for better readability
$colorSuccess = "Green"
$colorError = "Red"
$colorWarning = "Yellow"
$colorInfo = "Cyan"
$colorHighlight = "Magenta"

# Function to output colored text
function Write-ColorOutput {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [string]$ForegroundColor = "White"
    )
    
    Write-Host $Message -ForegroundColor $ForegroundColor
}

# Log file for this test run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path -Path $LogsDir -ChildPath "retriever_test_$timestamp.log"

# Create a summary of successful and failed retrievers
$successfulRetrievers = @()
$failedRetrievers = @()
$emptyDocumentRetrievers = @()

# Function to test a retriever with a question
function Test-Retriever {
    param (
        [string]$RetrieverType,
        [string]$Question
    )

    Write-ColorOutput "`n==============================" $colorInfo
    Write-ColorOutput "Testing: $RetrieverType" $colorHighlight
    Write-ColorOutput "Question: $Question" $colorInfo
    
    # Log to file
    Add-Content -Path $logFile -Value "`n=============================="
    Add-Content -Path $logFile -Value "Testing: $RetrieverType"
    Add-Content -Path $logFile -Value "Question: $Question"

    try {
        # Prepare request body
        $body = @{
            query = $Question
            retriever_type = $RetrieverType
            top_k = 5
        } | ConvertTo-Json

        # Make the request
        $response = Invoke-RestMethod -Method Post -Uri $BaseUrl -Body $body -ContentType "application/json" -ErrorAction Stop
        
        # Get response details
        $docCount = $response.documents.Count
        $metadata = $response.metadata | ConvertTo-Json
        
        # Success - we got a response
        Write-ColorOutput "STATUS: SUCCESS" $colorSuccess
        Write-ColorOutput "Document Count: $docCount" $colorSuccess
        Write-ColorOutput "Metadata: $metadata" $colorInfo
        
        # Log success to file
        Add-Content -Path $logFile -Value "STATUS: SUCCESS"
        Add-Content -Path $logFile -Value "Document Count: $docCount"
        Add-Content -Path $logFile -Value "Metadata: $metadata"

        # Check for zero documents
        if ($docCount -eq 0) {
            Write-ColorOutput "WARNING: No documents returned" $colorWarning
            Add-Content -Path $logFile -Value "WARNING: No documents returned"
            $script:emptyDocumentRetrievers += $RetrieverType
        }
        else {
            # Display first document details
            $firstDoc = $response.documents[0]
            Write-ColorOutput "First Document:" $colorInfo
            Write-ColorOutput "  ID: $($firstDoc.id)" $colorInfo
            Write-ColorOutput "  Score: $($firstDoc.score)" $colorInfo
            Write-ColorOutput "  Content (preview): $($firstDoc.content.Substring(0, [Math]::Min(100, $firstDoc.content.Length)))..." $colorInfo
            
            # Log first document details
            Add-Content -Path $logFile -Value "First Document:"
            Add-Content -Path $logFile -Value "  ID: $($firstDoc.id)"
            Add-Content -Path $logFile -Value "  Score: $($firstDoc.score)"
            Add-Content -Path $logFile -Value "  Content (preview): $($firstDoc.content.Substring(0, [Math]::Min(100, $firstDoc.content.Length)))..."
            
            # Check if scores are populated correctly (should not be 0)
            if ($firstDoc.score -eq 0) {
                Write-ColorOutput "WARNING: Document score is 0" $colorWarning
                Add-Content -Path $logFile -Value "WARNING: Document score is 0"
            }
        }
        
        # Add to successful retrievers list
        $script:successfulRetrievers += $RetrieverType
        
        return $true
    }
    catch {
        # Error handling
        $errorMessage = $_.Exception.Message
        
        Write-ColorOutput "STATUS: ERROR" $colorError
        Write-ColorOutput "Error details: $errorMessage" $colorError
        
        # Try to get more details from the response
        if ($_.Exception.Response) {
            try {
                $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $responseBody = $reader.ReadToEnd()
                $reader.Close()
                
                Write-ColorOutput "Response Body: $responseBody" $colorError
                Add-Content -Path $logFile -Value "Response Body: $responseBody"
            }
            catch {
                Write-ColorOutput "Could not read response body" $colorError
                Add-Content -Path $logFile -Value "Could not read response body"
            }
        }
        
        # Log error to file
        Add-Content -Path $logFile -Value "STATUS: ERROR"
        Add-Content -Path $logFile -Value "Error details: $errorMessage"
        
        # Add to failed retrievers list
        $script:failedRetrievers += $RetrieverType
        
        return $false
    }
}

# Run tests for each retriever and question
Write-ColorOutput "`n=== Starting Retriever API Tests ===" $colorHighlight
Add-Content -Path $logFile -Value "=== Starting Retriever API Tests ==="
Add-Content -Path $logFile -Value "Test run started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Test each retriever with the first question only for faster testing
foreach ($retriever in $RetrieverTypes) {
    Test-Retriever -RetrieverType $retriever -Question $Questions[0]
}

# Print summary
Write-ColorOutput "`n=== Test Summary ===" $colorHighlight
Write-ColorOutput "Tests ran at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" $colorInfo
Write-ColorOutput "Log file: $logFile" $colorInfo

Write-ColorOutput "`nSuccessful Retrievers ($($successfulRetrievers.Count)):" $colorSuccess
foreach ($retriever in $successfulRetrievers) {
    Write-ColorOutput "- $retriever" $colorSuccess
}

if ($emptyDocumentRetrievers.Count -gt 0) {
    Write-ColorOutput "`nRetrievers with Empty Results ($($emptyDocumentRetrievers.Count)):" $colorWarning
    foreach ($retriever in $emptyDocumentRetrievers) {
        Write-ColorOutput "- $retriever" $colorWarning
    }
}

if ($failedRetrievers.Count -gt 0) {
    Write-ColorOutput "`nFailed Retrievers ($($failedRetrievers.Count)):" $colorError
    foreach ($retriever in $failedRetrievers) {
        Write-ColorOutput "- $retriever" $colorError
    }
}

# Log summary to file
Add-Content -Path $logFile -Value "`n=== Test Summary ==="
Add-Content -Path $logFile -Value "Successful Retrievers ($($successfulRetrievers.Count)): $($successfulRetrievers -join ', ')"
Add-Content -Path $logFile -Value "Retrievers with Empty Results ($($emptyDocumentRetrievers.Count)): $($emptyDocumentRetrievers -join ', ')"
Add-Content -Path $logFile -Value "Failed Retrievers ($($failedRetrievers.Count)): $($failedRetrievers -join ', ')"
Add-Content -Path $logFile -Value "Test run completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-ColorOutput "`nTests completed. See log file for details: $logFile" $colorInfo 