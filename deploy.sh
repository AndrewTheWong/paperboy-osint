#!/bin/bash

# Paperboy Pipeline Deployment Script

set -e

echo "🚀 Deploying Paperboy Pipeline..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check environment variables
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo "❌ Environment variables SUPABASE_URL and SUPABASE_KEY must be set."
    echo "Please set them or create a .env file:"
    echo "  export SUPABASE_URL=your_supabase_url"
    echo "  export SUPABASE_KEY=your_supabase_key"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t paperboy:latest .

# Start services with docker-compose
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Test the API
echo "🧪 Testing API..."
if curl -f http://localhost:8000/health; then
    echo "✅ API is running successfully!"
else
    echo "❌ API is not responding"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Services running:"
echo "  - API: http://localhost:8000"
echo "  - API docs: http://localhost:8000/docs"
echo "  - Flower (monitoring): http://localhost:5555"
echo ""
echo "📝 To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "🛑 To stop services:"
echo "  docker-compose down" 