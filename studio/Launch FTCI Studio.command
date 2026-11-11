#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print "Please complete the installation in README.md first."
  exit 1
fi
exec .venv/bin/python -m ftci.server
