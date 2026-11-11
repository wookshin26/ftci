#!/usr/bin/env bash
# Runs each time an editor attaches to the codespace: starts FTCI Studio unless it is running.
set -euo pipefail
cd "$(dirname "$0")/../studio"
if curl --silent --fail --max-time 2 http://127.0.0.1:8765/api/health > /dev/null 2>&1; then
  echo "FTCI Studio is already running. Open port 8765 from the Ports tab."
  exit 0
fi
echo "Starting FTCI Studio. Open port 8765 from the Ports tab if no browser tab opens."
echo "Keep this terminal open: closing it stops the server."
exec python -m ftci.server
