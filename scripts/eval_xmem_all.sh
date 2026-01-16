#!/bin/bash

# ============================================================================
# CONFIGURATION
# ============================================================================
CKPT_DIR="log/ckpt"
CKPT_PATTERN="phase0_only_seq8_epoch_*.pth"
# ============================================================================

echo "======================================================================"
echo "CHECKPOINT EVALUATION SCRIPT"
echo "======================================================================"

# Find all checkpoint files matching the pattern and sort them
CKPTS=$(find "$CKPT_DIR" -name "$CKPT_PATTERN" | sort -V)

# Check if any checkpoints were found
if [ -z "$CKPTS" ]; then
    echo ""
    echo "Error: No checkpoints found matching pattern: $CKPT_PATTERN"
    echo "in directory: $CKPT_DIR"
    exit 1
fi

# Count checkpoints
NUM_CKPTS=$(echo "$CKPTS" | wc -l)
echo ""
echo "Found $NUM_CKPTS checkpoint(s):"
echo "$CKPTS" | while read ckpt; do
    epoch=$(basename "$ckpt" | sed 's/.*epoch_\([0-9]*\)\.pth/\1/')
    echo "  Epoch $epoch: $(basename $ckpt)"
done

echo ""
echo "Starting evaluation loop..."
echo "======================================================================"

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0

# Loop through each checkpoint
for ckpt in $CKPTS; do
    # Extract epoch number from filename
    epoch=$(basename "$ckpt" | sed 's/.*epoch_\([0-9]*\)\.pth/\1/')
    
    echo ""
    echo "======================================================================"
    echo "Evaluating: $(basename $ckpt)"
    echo "Epoch: $epoch"
    echo "======================================================================"
    
    # Run evaluation with all arguments hardcoded
    python eval_xmem.py \
        --cfg_file xmem_det/configs/temporal_pp_xmem_nuscenes.yaml \
        --xmem_cfg xmem_det/configs/xmem.yaml \
        --ckpt "$ckpt" \
        --split val \
        --seq_len 9 \
        --stride 8 \
        --keep_last_partial \
        --thr 0.5 \
        --eval_tag xmem_miou_len8_stride8_thr05_ep${epoch}
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully evaluated epoch $epoch"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "✗ Failed to evaluate epoch $epoch"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo "======================================================================"
done

# Print summary
echo ""
echo "======================================================================"
echo "EVALUATION SUMMARY"
echo "======================================================================"
echo "Total checkpoints: $NUM_CKPTS"
echo "Successful:        $SUCCESS_COUNT"
echo "Failed:            $FAIL_COUNT"
echo "======================================================================"

# Exit with error if any failures
if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Warning: $FAIL_COUNT evaluation(s) failed!"
    exit 1
else
    echo ""
    echo "All evaluations completed successfully!"
    exit 0
fi