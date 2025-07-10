#!/bin/bash

# Setup script to create .env file for spectral compositional generalization tutorial

echo "Setting up environment configuration..."

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    echo "Current contents:"
    cat .env
    echo ""
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Existing .env file preserved."
        exit 0
    fi
fi

# Get current directory as project root
PROJECT_ROOT="$(pwd)"

# Create .env file with default values
cat > .env << EOF
# Environment configuration

# Project root directory
PROJECT_ROOT=$PROJECT_ROOT
EOF

echo "✅ Created .env file with default settings:"
echo ""
cat .env
echo ""
echo "You can edit .env to add custom environment variables."
echo ""
echo "Now you can run: ./demo.sh"