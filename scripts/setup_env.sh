#!/usr/bin/env bash
# GovBench-Med environment setup
# Run this ONCE from the project root.
# Works in git-bash / WSL on Windows.

set -e
echo "=== GovBench-Med Setup ==="

# 1. Python virtual environment
echo "[1/5] Creating Python virtual environment..."
python -m venv .venv
source .venv/Scripts/activate || source .venv/bin/activate

# 2. Python dependencies
echo "[2/5] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet \
    requests \
    datasets \
    huggingface_hub \
    transformers \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    tqdm \
    jupyter

echo "  Python packages installed."

# 3. Ollama (local LLM inference server — no API key needed)
echo "[3/5] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Ollama not found. Installing..."
    # Windows: download installer
    echo "  Please download Ollama for Windows from: https://ollama.com/download"
    echo "  Then re-run this script."
    exit 1
else
    echo "  Ollama found: $(ollama --version)"
fi

# 4. Pull the three models (this downloads the weights — ~14 GB total)
echo "[4/5] Pulling LLM models (this may take a while — ~14 GB total)..."
echo "  Pulling llama3.1:8b ..."
ollama pull llama3.1:8b
echo "  Pulling mistral:7b ..."
ollama pull mistral:7b
echo "  Pulling qwen2.5:7b ..."
ollama pull qwen2.5:7b

# 5. __init__ files
echo "[5/5] Creating package __init__ files..."
touch src/__init__.py
touch src/agents/__init__.py
touch src/governance/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Activate venv:          source .venv/Scripts/activate"
echo "  2. Prepare datasets:       python scripts/prepare_data.py"
echo "  3. Run pilot experiment:   python scripts/run_experiments.py --pilot"
echo "  4. Run full experiment:    python scripts/run_experiments.py --full"
echo ""
echo "Ollama server must be running before experiments:"
echo "  Start it with:             ollama serve"
