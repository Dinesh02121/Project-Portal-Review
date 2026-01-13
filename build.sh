#!/bin/bash
pip install --only-binary=:all: pydantic-core pydantic

pip install --upgrade pip
pip install -r requirements.txt
