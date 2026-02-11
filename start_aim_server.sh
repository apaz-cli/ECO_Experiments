#!/bin/sh

# Generate runner wrappers in /tmp/eco/ so shebangs can use a fixed path
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
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

# Set the Aim repository path (default to .aim at repo root)
AIM_REPO="${AIM_REPO:-$REPO_DIR/.aim}"

# Create Aim repository directory if it doesn't exist
if [ ! -d "$AIM_REPO" ]; then
    echo "Creating directory: $AIM_REPO"
    mkdir -p "$AIM_REPO"
fi

# Check if Aim repository needs initialization
# An initialized Aim repo has a meta directory inside it
if [ ! -d "$AIM_REPO/meta" ]; then
    echo "Initializing Aim repository at $AIM_REPO"
    aim init --repo "$AIM_REPO" --yes
    if [ $? -ne 0 ]; then
        echo "Error: Failed to initialize Aim repository"
        exit 1
    fi
fi

echo "Starting Aim server..."
echo "Visit http://localhost:43800 to see the Aim UI"
echo "Press Ctrl+C to stop the server"
echo ""

# Start Aim server (blocking command)
aim up --host 0.0.0.0 --port 43800 --repo "$AIM_REPO"
