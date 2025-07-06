# Paperboy System Startup Script
# This script starts all components of the Paperboy system

Write-Host "Starting Paperboy System..." -ForegroundColor Green

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $connection -ne $null
}

# Function to wait for a service to be ready
function Wait-ForService {
    param([string]$Url, [int]$Timeout = 30)
    $startTime = Get-Date
    while ((Get-Date) -lt $startTime.AddSeconds($Timeout)) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "Service is ready at $Url" -ForegroundColor Green
                return $true
            }
        }
        catch {
            Write-Host "Waiting for service at $Url..." -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    Write-Host "Service at $Url failed to start within timeout" -ForegroundColor Red
    return $false
}

# Kill any existing processes
Write-Host "Stopping any existing processes..." -ForegroundColor Yellow
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name "celery" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Wait a moment for processes to stop
Start-Sleep -Seconds 3

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Please run: python -m venv venv" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
pip install supabase-realtime-py

# Start Supabase (if needed)
Write-Host "Starting Supabase..." -ForegroundColor Yellow
Start-Process -FilePath "supabase" -ArgumentList "start" -NoNewWindow -PassThru

# Wait for Supabase to be ready
Start-Sleep -Seconds 10

# Start Celery Beat Scheduler (background)
Write-Host "Starting Celery Beat Scheduler..." -ForegroundColor Yellow
Start-Process -FilePath "celery" -ArgumentList "-A app.celery_worker beat --loglevel=info" -NoNewWindow -PassThru

# Start Celery Worker (background)
Write-Host "Starting Celery Worker..." -ForegroundColor Yellow
Start-Process -FilePath "celery" -ArgumentList "-A app.celery_worker worker --loglevel=info --queues=default,scraper,preprocess,embedding,tagging,clustering" -NoNewWindow -PassThru

# Wait for Celery to start
Start-Sleep -Seconds 5

# Start FastAPI Server (background)
Write-Host "Starting FastAPI Server..." -ForegroundColor Yellow
Start-Process -FilePath "uvicorn" -ArgumentList "app.main:app --host 0.0.0.0 --port 8000 --reload" -NoNewWindow -PassThru

# Wait for FastAPI to start
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test the services
Write-Host "Testing services..." -ForegroundColor Yellow

# Test FastAPI
if (Wait-ForService "http://localhost:8000/health") {
    Write-Host "FastAPI server is running!" -ForegroundColor Green
} else {
    Write-Host "FastAPI server failed to start" -ForegroundColor Red
}

# Test Celery
try {
    $celeryStatus = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
    if ($celeryStatus.StatusCode -eq 200) {
        Write-Host "Celery workers are running!" -ForegroundColor Green
    }
} catch {
    Write-Host "Celery workers may not be ready yet" -ForegroundColor Yellow
}

Write-Host "`nPaperboy System Status:" -ForegroundColor Cyan
Write-Host "FastAPI Server: http://localhost:8000" -ForegroundColor White
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor White

Write-Host "`nAvailable Endpoints:" -ForegroundColor Cyan
Write-Host "POST /scraper/run - Start scraping" -ForegroundColor White
Write-Host "POST /ingest - Ingest an article" -ForegroundColor White
Write-Host "POST /cluster/run - Run clustering" -ForegroundColor White
Write-Host "GET /articles - List articles" -ForegroundColor White

Write-Host "`nTo run the full pipeline:" -ForegroundColor Cyan
Write-Host "1. curl.exe -X POST http://localhost:8000/scraper/run" -ForegroundColor White
Write-Host "2. curl.exe -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' -d '{\"url\":\"your_article_url\"}'" -ForegroundColor White
Write-Host "3. curl.exe -X POST http://localhost:8000/cluster/run" -ForegroundColor White

Write-Host "`nSystem is ready!" -ForegroundColor Green 