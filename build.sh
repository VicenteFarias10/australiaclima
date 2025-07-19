#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Install dependencies
echo "--- Installing dependencies ---"
pip install -r requirements.txt

# 2. Run all training scripts
echo "--- Running training scripts ---"
python entrenamiento_clustering.py
python entrenamiento_asociacion.py
python entrenamiento_confort.py
python entrenamiento_lluvia.py
python entrenamiento_evaporacion.py

echo "--- Build completed successfully ---" 