# API Base URL
$BaseUrl = "http://127.0.0.1:8000/invoke"

# Define the retriever endpoint paths (without the base URL and /invoke/)
$Endpoints = @(
    "naive_retriever"
    "bm25_retriever"
    "contextual_compression_retriever"
    "multi_query_retriever"
    "parent_document_retriever"
    "ensemble_retriever"
    "semantic_retriever"
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
    Write-Host "Creating directory: $LogsDir"
    New-Item -Path $LogsDir -ItemType Directory | Out-Null
}

# Create a timestamped log file in the logs directory
$LogFile = "$LogsDir\api_test_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "Starting API endpoint tests. Results will be logged to: $LogFile"

# Function to make a web request and log it
function Invoke-ApiCall {
    param(
        [string]$EndpointName,
        [string]$QuestionText
    )
    
    $FullUrl = "$BaseUrl/$EndpointName"
    $SeparatorLine = "------------------------------------------------------------"
    
    Write-Host $SeparatorLine
    Add-Content -Path $LogFile -Value $SeparatorLine
    
    $EndpointInfo = "Testing Endpoint: $FullUrl"
    Write-Host $EndpointInfo
    Add-Content -Path $LogFile -Value $EndpointInfo
    
    $QuestionInfo = "Question: $QuestionText"
    Write-Host $QuestionInfo
    Add-Content -Path $LogFile -Value $QuestionInfo
    
    $TimestampInfo = "Timestamp: $(Get-Date)"
    Write-Host $TimestampInfo
    Add-Content -Path $LogFile -Value $TimestampInfo
    
    $CurlInfo = "PowerShell command:"
    Write-Host $CurlInfo
    Add-Content -Path $LogFile -Value $CurlInfo
    
    $CommandInfo = "Invoke-RestMethod -Method POST -Uri '$FullUrl' -Headers @{'Content-Type'='application/json'} -Body '{`"question`":`"$QuestionText`"}'"
    Write-Host $CommandInfo
    Add-Content -Path $LogFile -Value $CommandInfo
    
    Write-Host "Response:"
    Add-Content -Path $LogFile -Value "Response:"
    
    try {
        $Body = @{
            question = $QuestionText
        } | ConvertTo-Json
        
        $Response = Invoke-RestMethod -Method POST -Uri $FullUrl -Headers @{'Content-Type'='application/json'} -Body $Body -ErrorAction Stop
        
        $StatusInfo = "Status Code: 200"
        Write-Host $StatusInfo
        Add-Content -Path $LogFile -Value $StatusInfo
        
        # Format and display response
        $FormattedResponse = $Response | ConvertTo-Json -Depth 10
        Write-Host $FormattedResponse
        Add-Content -Path $LogFile -Value $FormattedResponse
    }
    catch {
        $StatusCode = $_.Exception.Response.StatusCode.value__
        $StatusInfo = "Status Code: $StatusCode"
        Write-Host $StatusInfo -ForegroundColor Red
        Add-Content -Path $LogFile -Value $StatusInfo
        
        $ErrorInfo = "Error: $_"
        Write-Host $ErrorInfo -ForegroundColor Red
        Add-Content -Path $LogFile -Value $ErrorInfo
    }
    
    Write-Host $SeparatorLine
    Add-Content -Path $LogFile -Value $SeparatorLine
    Write-Host ""
    Add-Content -Path $LogFile -Value ""
}

# Iterate over each endpoint and each question
foreach ($Endpoint in $Endpoints) {
    foreach ($Question in $Questions) {
        Invoke-ApiCall -EndpointName $Endpoint -QuestionText $Question
        # Optional: Add a small delay between requests
        Start-Sleep -Seconds 10
    }
}

Write-Host "API endpoint tests completed. Full results logged to: $LogFile" 