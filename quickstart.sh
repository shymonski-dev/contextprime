#!/bin/bash

# QuickStart Script for DocTags RAG Dual Indexing Infrastructure
# This script sets up and validates the complete system

set -e

echo "======================================"
echo "DocTags RAG - QuickStart Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "--------------------------------"

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker from https://www.docker.com/"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

if ! command_exists docker-compose; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found${NC}"

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

echo ""

# Start databases
echo "Step 2: Starting databases..."
echo "--------------------------------"

echo "Starting Neo4j and Qdrant with Docker Compose..."
docker-compose up -d neo4j qdrant

echo "Waiting for databases to be ready..."
sleep 10

# Check if Neo4j is ready
echo -n "Checking Neo4j... "
for i in {1..30}; do
    if curl -s http://localhost:7474 >/dev/null; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Timeout${NC}"
        exit 1
    fi
    sleep 2
done

# Check if Qdrant is ready
echo -n "Checking Qdrant... "
for i in {1..30}; do
    if curl -s http://localhost:6333/health >/dev/null; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Timeout${NC}"
        exit 1
    fi
    sleep 2
done

echo ""

# Install Python dependencies
echo "Step 3: Installing Python dependencies..."
echo "--------------------------------"

cd doctags_rag
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Setup databases
echo "Step 4: Initializing databases..."
echo "--------------------------------"

python scripts/setup_databases.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Databases initialized successfully${NC}"
else
    echo -e "${RED}✗ Database initialization failed${NC}"
    exit 1
fi

echo ""

# Run tests
echo "Step 5: Running tests (optional)..."
echo "--------------------------------"
read -p "Do you want to run the test suite? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    pytest tests/test_indexing.py -v --tb=short

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${YELLOW}⚠ Some tests failed (this is normal if databases are empty)${NC}"
    fi
fi

echo ""

# Summary
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Services running:"
echo "  - Neo4j:  http://localhost:7474 (Browser)"
echo "            bolt://localhost:7687 (Bolt)"
echo "            Username: neo4j"
echo "            Password: password"
echo ""
echo "  - Qdrant: http://localhost:6333 (API)"
echo "            http://localhost:6333/dashboard (Dashboard)"
echo ""
echo "Next steps:"
echo "  1. Review the documentation: /Users/simonkelly/SUPER_RAG/DUAL_INDEXING_SETUP.md"
echo "  2. Run example usage: python scripts/example_usage.py"
echo "  3. Start building your RAG application!"
echo ""
echo "To stop the services:"
echo "  docker-compose down"
echo ""
echo "To stop and remove all data:"
echo "  docker-compose down -v"
echo ""
echo -e "${GREEN}Happy coding!${NC}"
