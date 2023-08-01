#!/bin/bash

# this scripts runs the first time setup for the AI module repository

# setup pre-commit
pre-commit install --install-hooks
pre-commit run --all-files

# add poetry environment alias
echo "alias poetry_activate='source \"\$(poetry env list --full-path | grep Activated | cut -d\" \" -f1 )/bin/activate\"'" >> ~/.bashrc