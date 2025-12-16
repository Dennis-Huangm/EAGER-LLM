#!/bin/bash

# DeepSpeed checkpoint to FP32 converter
# Usage: ./convert_checkpoint.sh <checkpoint_path> [output_suffix]
# Example: ./convert_checkpoint.sh /path/to/checkpoint-354
#          ./convert_checkpoint.sh /path/to/checkpoint-354 fp32

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [output_suffix]"
    echo "Example: $0 /root/autodl-tmp/huangminrui/EAGER-LLM/checkpoints/multievent/v0/checkpoint-354"
    exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_SUFFIX="${2:-fp32}"

# Remove trailing slash if present
CHECKPOINT_PATH="${CHECKPOINT_PATH%/}"

# Output path
OUTPUT_PATH="${CHECKPOINT_PATH}-${OUTPUT_SUFFIX}"

# zero_to_fp32.py location
SCRIPT_PATH="${CHECKPOINT_PATH}/zero_to_fp32.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: zero_to_fp32.py not found at $SCRIPT_PATH"
    exit 1
fi

echo "Converting checkpoint..."
echo "  Input:  $CHECKPOINT_PATH"
echo "  Output: $OUTPUT_PATH"

python "$SCRIPT_PATH" "$CHECKPOINT_PATH" "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
    
    # Copy all json files from input to output
    echo "Copying json files..."
    cp "$CHECKPOINT_PATH"/*.json "$OUTPUT_PATH"/ 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "JSON files copied successfully!"
    else
        echo "No JSON files found to copy."
    fi
else
    echo "Conversion failed!"
    exit 1
fi
