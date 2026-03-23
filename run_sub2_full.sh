#!/bin/bash
# Sequential overnight runner for sub2 and full resolution pairs
# using the legacy BruteForce RIFT2 pipeline.
# Run with: nohup bash run_sub2_full.sh > run_sub2_full.log 2>&1 &

PYTHON=/home/ubuntu/miniconda3/envs/rift2/bin/python
SCRIPT=test_rift2_single.py
DIR=/home/ubuntu/rse_radar/Rift/RIFT2-multimodal-matching-rotation-python

cd "$DIR"

echo "=========================================="
echo "Starting run at $(date)"
echo "=========================================="

echo ""
echo "--- Running sub2 ---"
$PYTHON -u $SCRIPT --res sub2
echo "sub2 finished at $(date)"

echo ""
echo "--- Running full ---"
$PYTHON -u $SCRIPT --res full
echo "full finished at $(date)"

echo ""
echo "=========================================="
echo "All runs complete at $(date)"
echo "=========================================="
