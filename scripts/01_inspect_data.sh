#!/bin/bash
# 01_inspect_data.sh
# Verify raw dataset download integrity: file counts, sizes, and annotation presence.
# Run from project root: bash scripts/01_inspect_data.sh

set -e

RAW="data/raw"

echo "========================================"
echo "Step 1: Inspecting raw dataset contents"
echo "========================================"

for DATASET in cbvd5 cvb; do
    DIR="$RAW/$DATASET"
    echo ""
    echo "--- $DATASET ---"
    if [ ! -d "$DIR" ]; then
        echo "  [MISSING] $DIR — run download steps in docs/datasets.md"
        continue
    fi

    echo "  Top-level:"
    ls "$DIR"

    echo "  Disk usage:"
    du -sh "$DIR"

    echo "  File counts by type:"
    find "$DIR" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10
done

echo ""
echo "========================================"
echo "Step 1 complete. Verify both datasets are present before running Step 2."
echo "========================================"
