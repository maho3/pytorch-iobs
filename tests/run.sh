#!/bin/bash

cd ..

conda activate iobs
pip3 install coverage pytest
echo "Running tests"
COVERAGE_FILE=.coverage_iobs python3 -m coverage run --source=iobs -m pytest tests/test_layers.py

coverage xml
coverage report -m