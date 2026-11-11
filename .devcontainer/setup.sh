#!/usr/bin/env bash
# Runs once when the codespace is created: installs FTCI Studio and its Julia engine.
set -euo pipefail
cd "$(dirname "$0")/../studio"
python -m pip install --quiet -e .
julia --startup-file=no --project=ftci/julia -e 'using Pkg; Pkg.instantiate(); using TensorCrossInterpolation'
echo "FTCI Studio is installed."
