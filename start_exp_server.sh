#!/bin/sh
# Start the experiment tracking server (replaces start_aim_server.sh).
# Reads EXP_DIR and PORT env vars, or uses defaults.
#
# Usage:
#   ./start_exp_server.sh
#   EXP_DIR=/path/to/sweeps PORT=53800 ./start_exp_server.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Generate runner wrappers in /tmp/eco/ so shebangs can use a fixed path
mkdir -p /tmp/eco

cat > /tmp/eco/run_config.sh <<EOF
#!/bin/bash
exec "$REPO_DIR/run_config.sh" "\$@"
EOF
chmod +x /tmp/eco/run_config.sh

cat > /tmp/eco/run_sweep.py <<EOF
#!/usr/bin/env python3
import os, sys
os.execv(sys.executable, [sys.executable, "$REPO_DIR/run_sweep.py"] + sys.argv[1:])
EOF
chmod +x /tmp/eco/run_sweep.py

echo "Installed runner wrappers in /tmp/eco/"
echo "Starting experiment server..."
echo "  Metrics API: http://0.0.0.0:${PORT:-53800}"
echo "  Visualize:   python visualize_experiment.py --exp-source http://localhost:${PORT:-53800}"
echo "Press Ctrl+C to stop"
echo ""

exec python3 "$REPO_DIR/exp_server.py" \
    --exp-dir "${EXP_DIR:-$REPO_DIR/outputs/sweeps}" \
    --port "${PORT:-53800}" \
    --host 0.0.0.0
