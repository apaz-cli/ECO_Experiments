#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <from-path> <to-path>"
    exit 1
fi

FROM="$1"
TO="$2"

# Escape for use in sed (handles /, ., etc.)
FROM_ESC=$(printf '%s' "$FROM" | sed 's|[\\/.^$*[]|\\&|g')
TO_ESC=$(printf '%s' "$TO" | sed 's|[\\/.^$*[]|\\&|g; s|&|\\&|g')

echo "Replacing: $FROM -> $TO"

# Find all text files in the repo that contain the path, excluding .git
mapfile -t FILES < <(grep -rl --exclude-dir=.git "$FROM" .)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files contain '$FROM'"
    exit 0
fi

for FILE in "${FILES[@]}"; do
    echo "  Updating: $FILE"
    sed -i "s|${FROM_ESC}|${TO}|g" "$FILE"
done

echo "Done. Updated ${#FILES[@]} file(s)."
