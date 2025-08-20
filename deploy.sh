#!/bin/bash

# E-Commerce Customer Churn Prediction Deployment Script

set -e

echo "🚀 Starting deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t churn-prediction:latest .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check API health
echo "🔍 Checking API health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed"
    docker-compose logs churn-api
    exit 1
fi

# Check Streamlit
echo "🔍 Checking Streamlit..."
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Streamlit is running!"
else
    echo "❌ Streamlit health check failed"
    docker-compose logs churn-streamlit
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Services running:"
echo "   - FastAPI: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Streamlit: http://localhost:8501"
echo ""
echo "📝 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart: docker-compose restart"
