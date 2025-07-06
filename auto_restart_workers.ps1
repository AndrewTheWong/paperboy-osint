# Auto-restart Celery workers on file changes

Write-Host "Starting Auto-Restart Monitor..." -ForegroundColor Green
Write-Host "Monitoring for file changes in app/ directory..." -ForegroundColor Cyan

function Stop-AllWorkers {
    Write-Host "Stopping all Celery workers..." -ForegroundColor Yellow
    Get-Process | Where-Object {$_.ProcessName -like "*python*" -and $_.MainWindowTitle -like "*celery*"} | Stop-Process -Force
    Start-Sleep -Seconds 3
    Write-Host "All workers stopped" -ForegroundColor Green
}

function Start-AllWorkers {
    Write-Host "Starting all workers..." -ForegroundColor Yellow
    
    # Start FastAPI
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Normal
    
    # Start Celery Beat
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; celery -A app.celery_worker beat --loglevel=info" -WindowStyle Normal
    
    # Start dedicated workers
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; celery -A app.celery_worker worker --loglevel=info --pool=solo -Q scrape -n Scraper" -WindowStyle Normal
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; celery -A app.celery_worker worker --loglevel=info --pool=solo -Q preprocess -n Preprocess" -WindowStyle Normal
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; celery -A app.celery_worker worker --loglevel=info --pool=solo -Q clustering -n Clusterer" -WindowStyle Normal
    
    Write-Host "All workers restarted" -ForegroundColor Green
}

# Initial start
Start-AllWorkers

# Monitor for changes
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = "app"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    $timeStamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    
    Write-Host "[$timeStamp] $changeType detected: $path" -ForegroundColor Yellow
    
    # Only restart for Python files
    if ($path -match "\.py$") {
        Write-Host "Python file changed - restarting workers..." -ForegroundColor Red
        Stop-AllWorkers
        Start-Sleep -Seconds 2
        Start-AllWorkers
        Write-Host "Workers restarted after file change" -ForegroundColor Green
    }
}

# Register event handlers
Register-ObjectEvent $watcher "Changed" -Action $action
Register-ObjectEvent $watcher "Created" -Action $action
Register-ObjectEvent $watcher "Deleted" -Action $action
Register-ObjectEvent $watcher "Renamed" -Action $action

Write-Host "Auto-restart monitor is running..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Cyan

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Cleanup
    Unregister-Event -SourceIdentifier $watcher.Changed
    Unregister-Event -SourceIdentifier $watcher.Created
    Unregister-Event -SourceIdentifier $watcher.Deleted
    Unregister-Event -SourceIdentifier $watcher.Renamed
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "Auto-restart monitor stopped" -ForegroundColor Yellow
} 