#!/bin/bash
# Deploy TurboMOQ to Mac mini and run benchmarks
# Usage: ./scripts/deploy_mac_mini.sh

set -e

REMOTE="cyber 02@192.168.23.25"
REMOTE_DIR="/tmp/turboMOQ"

echo "=== Deploying TurboMOQ to Mac mini ==="

# Sync repo (excluding .venv, __pycache__, .git)
echo "Syncing files..."
rsync -avz --delete \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude '*.egg-info' \
    --exclude '*.dylib' \
    --exclude '*.so' \
    /tmp/turboMOQ/ "${REMOTE}:${REMOTE_DIR}/"

echo "Setting up environment on Mac mini..."
ssh "${REMOTE}" bash -s << 'EOF'
cd /tmp/turboMOQ

# Create venv if not exists
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

# Install dependencies
.venv/bin/pip install -e ".[dev]" --quiet

# Try MLX install (Apple Silicon only)
.venv/bin/pip install mlx --quiet 2>/dev/null || echo "MLX install skipped (may need ARM64)"

# Compile C extension
cd turbomoq/llamacpp_ext
make clean && make
cd /tmp/turboMOQ

echo ""
echo "=== Running tests ==="
.venv/bin/pytest tests/ -q --tb=line 2>&1

echo ""
echo "=== Running benchmark ==="
.venv/bin/python benchmarks/benchmark_mac_mini_16gb.py 2>&1
EOF

echo ""
echo "=== Fetching results ==="
scp "${REMOTE}:${REMOTE_DIR}/benchmark-results-raw/mac_mini_16gb_results.json" \
    /tmp/turboMOQ/benchmark-results-raw/mac_mini_16gb_results.json 2>/dev/null || echo "No results file to fetch"

echo "Done!"
