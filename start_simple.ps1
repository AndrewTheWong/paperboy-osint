# Simple Paperboy Startup Script
Write-Host "Starting Paperboy System..." -ForegroundColor Green

# Start Supabase
Write-Host "Starting Supabase..." -ForegroundColor Yellow
supabase start

# Start Redis (if not running)
Write-Host "Starting Redis..." -ForegroundColor Yellow
Start-Process -FilePath "redis-server" -WindowStyle Minimized

# Start Celery Worker
Write-Host "Starting Celery Worker..." -ForegroundColor Yellow
Start-Process -FilePath "celery" -ArgumentList "-A app.celery_worker worker --loglevel=info --queues=scraper,preprocess,embedding,tagging,clustering" -WindowStyle Minimized

# Start Celery Beat
Write-Host "Starting Celery Beat..." -ForegroundColor Yellow
Start-Process -FilePath "celery" -ArgumentList "-A app.celery_worker beat --loglevel=info" -WindowStyle Minimized

# Start FastAPI Server
Write-Host "Starting FastAPI Server..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Minimized

Write-Host "All services started!" -ForegroundColor Green
Write-Host "FastAPI: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Health: http://localhost:8000/health" -ForegroundColor Cyan 