# Paperboy Local Development Startup Script
# This script sets up and runs the entire Paperboy pipeline locally

Write-Host "🚀 Starting Paperboy Local Development Environment..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ Error: Please run this script from the Paperboy project root directory" -ForegroundColor Red
    exit 1
}

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Command "python")) {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "pip")) {
    Write-Host "❌ Error: pip is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
Write-Host "🐍 Setting up Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check if Redis is running, start if not
Write-Host "🔴 Checking Redis..." -ForegroundColor Yellow
try {
    $redisTest = redis-cli ping 2>$null
    if ($redisTest -eq "PONG") {
        Write-Host "✅ Redis is already running" -ForegroundColor Green
    } else {
        throw "Redis not responding"
    }
} catch {
    Write-Host "🔄 Starting Redis..." -ForegroundColor Yellow
    
    # Try to start Redis using Docker if available
    if (Test-Command "docker") {
        Write-Host "Using Docker to start Redis..." -ForegroundColor Cyan
        Start-Process -FilePath "docker" -ArgumentList "run", "-d", "--name", "paperboy-redis", "-p", "6379:6379", "redis:alpine" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    } else {
        # Try Windows Redis if available
        if (Test-Path "redis-windows\redis-server.exe") {
            Write-Host "Starting Windows Redis..." -ForegroundColor Cyan
            Start-Process -FilePath ".\redis-windows\redis-server.exe" -WindowStyle Hidden
            Start-Sleep -Seconds 3
            } else {
        Write-Host "Error: Redis not found. Please install Redis or Docker" -ForegroundColor Red
        Write-Host "   - Install Docker Desktop and run: docker run -d --name paperboy-redis -p 6379:6379 redis:alpine" -ForegroundColor Yellow
        Write-Host "   - Or download Redis for Windows from: https://github.com/microsoftarchive/redis/releases" -ForegroundColor Yellow
        exit 1
    }
    }
}

# Verify Redis is working
try {
    $redisTest = redis-cli ping 2>$null
    if ($redisTest -eq "PONG") {
        Write-Host "✅ Redis is running successfully" -ForegroundColor Green
    } else {
        throw "Redis still not responding"
    }
} catch {
    Write-Host "Error: Failed to start Redis" -ForegroundColor Red
    exit 1
}

# Set environment variables
Write-Host "🔧 Setting environment variables..." -ForegroundColor Yellow
$env:CELERY_BROKER_URL = "redis://localhost:6379/0"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/0"
$env:REDIS_URL = "redis://localhost:6379/0"

# Function to start a worker in a new window
function Start-Worker {
    param(
        [string]$WorkerName,
        [string]$QueueName,
        [string]$Command
    )
    
    Write-Host "🚀 Starting $WorkerName worker..." -ForegroundColor Cyan
    
    $workerScript = @"
# Activate virtual environment and start worker
cd '$PWD'
& '.\venv\Scripts\Activate.ps1'
`$env:CELERY_BROKER_URL = 'redis://localhost:6379/0'
`$env:CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
`$env:REDIS_URL = 'redis://localhost:6379/0'
Write-Host 'Starting $WorkerName worker...' -ForegroundColor Green
$Command
pause
"@
    
    $tempScript = "temp_$WorkerName.ps1"
    $workerScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $tempScript -WindowStyle Normal
    
    # Clean up temp script after a delay
    Start-Job -ScriptBlock {
        param($scriptPath)
        Start-Sleep -Seconds 5
        if (Test-Path $scriptPath) { Remove-Item $scriptPath -Force }
    } -ArgumentList $tempScript
}

# Start all workers
Write-Host "👥 Starting Celery workers..." -ForegroundColor Yellow

Start-Worker -WorkerName "Scraper" -QueueName "scraping_queue" -Command "celery -A workers.scraper worker -Q scraping_queue -n Scraper@%h -l info"
Start-Sleep -Seconds 2

Start-Worker -WorkerName "Preprocess" -QueueName "preprocess_queue" -Command "celery -A workers.preprocess worker -Q preprocess_queue -n Preprocess@%h -l info"
Start-Sleep -Seconds 2

Start-Worker -WorkerName "Embedder" -QueueName "embedding_queue" -Command "celery -A workers.embedder worker -Q embedding_queue -n Embedder@%h -l info"
Start-Sleep -Seconds 2

Start-Worker -WorkerName "Clusterer" -QueueName "clustering_queue" -Command "celery -A workers.cluster worker -Q clustering_queue -n Clusterer@%h -l info"
Start-Sleep -Seconds 2

Start-Worker -WorkerName "Tagger" -QueueName "tagging_queue" -Command "celery -A workers.tagger worker -Q tagging_queue -n Tagger@%h -l info"
Start-Sleep -Seconds 2

Start-Worker -WorkerName "Translator" -QueueName "translation_queue" -Command "celery -A workers.translator worker -Q translation_queue -n Translator@%h -l info"
Start-Sleep -Seconds 2

Start-Worker -WorkerName "Orchestrator" -QueueName "orchestrator_queue" -Command "celery -A workers.orchestrator worker -Q orchestrator_queue -n Orchestrator@%h -l info"
Start-Sleep -Seconds 2

# Start the API
Write-Host "🌐 Starting API server..." -ForegroundColor Yellow
Start-Worker -WorkerName "API" -QueueName "" -Command "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

# Wait a moment for everything to start
Start-Sleep -Seconds 5

Write-Host "`n🎉 Paperboy is now running!" -ForegroundColor Green
Write-Host "`n📋 Services running:" -ForegroundColor Cyan
Write-Host "   • Redis (Message Broker)" -ForegroundColor White
Write-Host "   • Scraper Worker" -ForegroundColor White
Write-Host "   • Preprocess Worker" -ForegroundColor White
Write-Host "   • Embedder Worker" -ForegroundColor White
Write-Host "   • Clusterer Worker" -ForegroundColor White
Write-Host "   • Tagger Worker" -ForegroundColor White
Write-Host "   • Translator Worker" -ForegroundColor White
Write-Host "   • Orchestrator Worker" -ForegroundColor White
Write-Host "   • API Server (http://localhost:8000)" -ForegroundColor White

Write-Host "`n🔗 Useful URLs:" -ForegroundColor Cyan
Write-Host "   • API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "   • API Health Check: http://localhost:8000/health" -ForegroundColor White

Write-Host "`n💡 To stop all services:" -ForegroundColor Yellow
Write-Host "   • Close the worker windows" -ForegroundColor White
Write-Host "   • Stop Redis: redis-cli shutdown" -ForegroundColor White
Write-Host "   • Or if using Docker: docker stop paperboy-redis" -ForegroundColor White

Write-Host "`n✨ All services are now running in separate windows!" -ForegroundColor Green 