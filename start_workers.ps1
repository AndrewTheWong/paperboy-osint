#!/usr/bin/env powershell
<#
.SYNOPSIS
    Start all Celery workers for the Paperboy pipeline

.DESCRIPTION
    This script starts dedicated Celery workers for each pipeline step:
    - Scraper worker
    - Translation worker  
    - Tagging worker
    - Embedding worker
    - Clustering worker
    - Orchestrator worker

.PARAMETER Workers
    Specific workers to start (default: all)
    Options: scraper, translator, tagger, embedder, cluster, orchestrator, all

.PARAMETER Concurrency
    Number of worker processes per worker type (default: 1)

.EXAMPLE
    .\start_workers.ps1
    Start all workers with default settings

.EXAMPLE
    .\start_workers.ps1 -Workers "scraper,translator" -Concurrency 2
    Start only scraper and translator workers with 2 processes each
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$Workers = "all",
    
    [Parameter(Mandatory=$false)]
    [int]$Concurrency = 1
)

Write-Host "🚀 Starting Paperboy Celery Workers" -ForegroundColor Green
Write-Host "Workers: $Workers" -ForegroundColor Yellow
Write-Host "Concurrency: $Concurrency" -ForegroundColor Yellow
Write-Host ""

# Function to start a worker
function Start-Worker {
    param(
        [string]$WorkerName,
        [string]$QueueName,
        [string]$WorkerLabel
    )
    
    Write-Host "Starting $WorkerLabel worker..." -ForegroundColor Cyan
    
    $workerCmd = "celery -A celery_worker worker --loglevel=info --concurrency=$Concurrency -Q $QueueName -n $WorkerLabel@%h"
    
    Write-Host "Command: $workerCmd" -ForegroundColor Gray
    
    # Start the worker in a new PowerShell window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; $workerCmd" -WindowStyle Normal
    
    Write-Host "✅ $WorkerLabel worker started" -ForegroundColor Green
    Start-Sleep -Seconds 2
}

# Define worker configurations
$workerConfigs = @{
    "scraper" = @{
        QueueName = "scrape"
        WorkerLabel = "Scraper"
    }
    "translator" = @{
        QueueName = "translate" 
        WorkerLabel = "Translator"
    }
    "tagger" = @{
        QueueName = "tag"
        WorkerLabel = "Tagger"
    }
    "embedder" = @{
        QueueName = "embed"
        WorkerLabel = "Embedder"
    }
    "cluster" = @{
        QueueName = "cluster"
        WorkerLabel = "Cluster"
    }
    "orchestrator" = @{
        QueueName = "orchestrate"
        WorkerLabel = "Orchestrator"
    }
}

# Parse workers parameter
$workersToStart = @()
if ($Workers -eq "all") {
    $workersToStart = $workerConfigs.Keys
} else {
    $workersToStart = $Workers.Split(",") | ForEach-Object { $_.Trim() }
}

# Validate worker names
foreach ($worker in $workersToStart) {
    if (-not $workerConfigs.ContainsKey($worker)) {
        Write-Host "❌ Unknown worker: $worker" -ForegroundColor Red
        Write-Host "Available workers: $($workerConfigs.Keys -join ', ')" -ForegroundColor Yellow
        exit 1
    }
}

# Start Redis if not running
Write-Host "🔍 Checking Redis..." -ForegroundColor Yellow
try {
    $redisTest = redis-cli ping 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️ Redis not running. Please start Redis first:" -ForegroundColor Yellow
        Write-Host "redis-server" -ForegroundColor Gray
        exit 1
    } else {
        Write-Host "✅ Redis is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Redis not available. Please install and start Redis." -ForegroundColor Red
    exit 1
}

# Start workers
Write-Host ""
Write-Host "Starting workers..." -ForegroundColor Green
Write-Host ""

foreach ($worker in $workersToStart) {
    $config = $workerConfigs[$worker]
    Start-Worker -WorkerName $worker -QueueName $config.QueueName -WorkerLabel $config.WorkerLabel
}

Write-Host ""
Write-Host "🎉 All workers started!" -ForegroundColor Green
Write-Host ""
Write-Host "Worker Status:" -ForegroundColor Yellow
foreach ($worker in $workersToStart) {
    $config = $workerConfigs[$worker]
    Write-Host "  ✅ $($config.WorkerLabel) - Queue: $($config.QueueName)" -ForegroundColor Green
}

Write-Host ""
Write-Host "To monitor workers:" -ForegroundColor Cyan
Write-Host "  celery -A celery_worker flower" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop all workers:" -ForegroundColor Cyan  
Write-Host "  Get-Process | Where-Object {`$_.ProcessName -eq 'python'} | Stop-Process" -ForegroundColor Gray
Write-Host ""
Write-Host "To run the full pipeline:" -ForegroundColor Cyan
Write-Host "  python -c 'from workers.orchestrator import run_full_pipeline; result = run_full_pipeline.delay(); print(result.get())'" -ForegroundColor Gray 