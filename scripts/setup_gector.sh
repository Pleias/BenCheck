#!/bin/bash
# Setup script for downloading GECToR vocabulary files
# These files are required for the Grammar Quality Check

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GECTOR_DATA_DIR="$PROJECT_ROOT/models/gector/data"
TEMP_DIR=$(mktemp -d)

echo "Setting up GECToR data files..."
echo "Target directory: $GECTOR_DATA_DIR"

# Create the data directory if it doesn't exist
mkdir -p "$GECTOR_DATA_DIR/output_vocabulary"

# Clone the gector repository to temporary directory
echo "Cloning gotutiyan/gector repository..."
git clone --depth 1 https://github.com/gotutiyan/gector.git "$TEMP_DIR/gector"

# Copy the required files
echo "Copying vocabulary files..."
cp "$TEMP_DIR/gector/data/verb-form-vocab.txt" "$GECTOR_DATA_DIR/verb-form-vocab.txt"
cp "$TEMP_DIR/gector/data/output_vocabulary/labels.txt" "$GECTOR_DATA_DIR/output_vocabulary/labels.txt"

# Clean up
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "✓ GECToR data files successfully downloaded!"
echo ""
echo "Downloaded files:"
echo "  - $GECTOR_DATA_DIR/verb-form-vocab.txt ($(du -h "$GECTOR_DATA_DIR/verb-form-vocab.txt" | cut -f1))"
echo "  - $GECTOR_DATA_DIR/output_vocabulary/labels.txt ($(du -h "$GECTOR_DATA_DIR/output_vocabulary/labels.txt" | cut -f1))"
echo ""
echo "You can now use GrammarQualityCheck with GECToR models."
