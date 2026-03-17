#!/bin/bash
# Export a diff of ECO_Experiments changes relative to pytorch/torchtitan
# (at the commit just before the config system refactor, PR #2386)

BASE_COMMIT="4aebdd2c"
OUTPUT="eco_changes.diff"

# Paths to exclude
EXCLUDE=(
    tests/
    sweeps/
    docs/
    paper/
    ECO_Paper.tex
    download_c4.py
    download_fineweb.py
    export_diff.sh
    eco_changes.diff
)

# Build exclude args for git diff
EXCLUDE_ARGS=()
for path in "${EXCLUDE[@]}"; do
    EXCLUDE_ARGS+=(":(exclude)${path}")
done

git diff "${BASE_COMMIT}" -- . "${EXCLUDE_ARGS[@]}" > "${OUTPUT}"

echo "Wrote ${OUTPUT} ($(wc -l < "${OUTPUT}") lines, $(git diff "${BASE_COMMIT}" --name-only -- . "${EXCLUDE_ARGS[@]}" | wc -l) files)"
