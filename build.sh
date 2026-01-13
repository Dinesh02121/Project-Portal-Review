#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install dependencies with prefer-binary flag
pip install --no-cache-dir --prefer-binary -r requirements.txt
