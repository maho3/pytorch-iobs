#!/bin/bash

cd ..

source /opt/homebrew/anaconda3/etc/profile.d/conda.sh

conda activate iob
pip3 install coverage pytest tqdm numpy torch
echo "Testing layers"
COVERAGE_FILE=.coverage_layers python3 -m coverage run --source=iobs -m pytest tests/test_layers.py

echo "Testing simulators"
COVERAGE_FILE=.coverage_sims python3 -m coverage run --source=iobs -m pytest tests/test_simulators.py
conda deactivate

echo "Testing models"
COVERAGE_FILE=.coverage_models python3 -m coverage run --source=iobs -m pytest tests/test_models.py
conda deactivate

coverage combine .coverage_layers .coverage_sims .coverage_models
coverage xml
coverage report -m