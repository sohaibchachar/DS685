#!/bin/bash
set -e

echo "🔧 Setting up PostgreSQL + pgvector inside Docker container..."

# Update package lists
sudo apt update

# Install PostgreSQL
echo "📦 Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-16

# Start PostgreSQL service
echo "🚀 Starting PostgreSQL..."
sudo service postgresql start

# Wait for PostgreSQL to be ready
sleep 3

# Create database and user
echo "🗄️  Creating database and user..."
sudo -u postgres psql -c "CREATE DATABASE agents;" 2>/dev/null || echo "Database 'agents' already exists"
sudo -u postgres psql -c "ALTER ROLE postgres WITH PASSWORD 'postgres';" 2>/dev/null || echo "Password already set"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agents TO postgres;"

# Enable pgvector extension
echo "🧮 Setting up pgvector extension..."
cd /tmp
git clone https://github.com/pgvector/pgvector.git || echo "Repository already cloned"
cd pgvector

# Build and install pgvector
echo "🔨 Building pgvector..."
sudo make clean || true
make clean || true
make
sudo make install

# Enable extension in database
echo "✅ Enabling pgvector extension..."
psql -U postgres -d agents -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "Extension already enabled"

# Test connection
echo "🧪 Testing connection..."
psql -U postgres -d agents -c "SELECT version();"

echo "✅ PostgreSQL + pgvector setup complete!"
echo ""
echo "🍽️  Database info:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: agents"
echo "   User: postgres"
echo "   Password: postgres"
echo ""
echo "🔗 Test connection with:"
echo "   PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d agents"
