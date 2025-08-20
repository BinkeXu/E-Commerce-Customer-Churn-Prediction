# E-Commerce Customer Churn Prediction Deployment Script (PowerShell)

Write-Host "🚀 Starting deployment..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker and try again." -ForegroundColor Red
    exit 1
}

# Build the Docker image
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
docker build -t churn-prediction:latest .

# Stop existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose down

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check API health
Write-Host "🔍 Checking API health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API is healthy!" -ForegroundColor Green
    } else {
        throw "API returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Host "❌ API health check failed" -ForegroundColor Red
    docker-compose logs churn-api
    exit 1
}

# Check Streamlit
Write-Host "🔍 Checking Streamlit..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Streamlit is running!" -ForegroundColor Green
    } else {
        throw "Streamlit returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Host "❌ Streamlit health check failed" -ForegroundColor Red
    docker-compose logs churn-streamlit
    exit 1
}

Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Services running:" -ForegroundColor Cyan
Write-Host "   - FastAPI: http://localhost:8000" -ForegroundColor White
Write-Host "   - API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "   - Streamlit: http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "📝 Useful commands:" -ForegroundColor Cyan
Write-Host "   - View logs: docker-compose logs -f" -ForegroundColor White
Write-Host "   - Stop services: docker-compose down" -ForegroundColor White
Write-Host "   - Restart: docker-compose restart" -ForegroundColor White
