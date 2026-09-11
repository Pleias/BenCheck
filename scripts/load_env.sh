#!/bin/bash
# Load environment variables from .env file
# Usage: source scripts/load_env.sh

if [ -f .env ]; then
    echo "📝 Loading environment variables from .env..."

    # Export variables from .env file
    # This handles the format: KEY=value
    set -a
    source <(cat .env | grep -v '^#' | grep -v '^$' | sed 's/^/export /')
    set +a

    echo "✅ Environment variables loaded"

    # Check if GEMINI_API_KEY is set
    if [ -n "$GEMINI_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ]; then
        echo "✅ Gemini API key detected"
    else
        echo "⚠️  No Gemini API key found in .env"
    fi
else
    echo "❌ .env file not found"
    echo ""
    echo "To create .env file:"
    echo "  1. Copy template: cp .env.example .env"
    echo "  2. Edit .env and add your API key"
    echo "  3. Load it: source scripts/load_env.sh"
    echo ""
    echo "Or set directly:"
    echo "  export GEMINI_API_KEY='your-key-here'"
fi
