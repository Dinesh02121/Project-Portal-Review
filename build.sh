#!/usr/bin/env bash
set -e

echo "Python version:"
python --version

pip install --upgrade pip
pip install -r requirements.txt
