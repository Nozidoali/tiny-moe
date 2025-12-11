#!/bin/bash

set -e

# Use relative path from TINYMOE_DIR
TINYMOE_DIR="${TINYMOE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$TINYMOE_DIR/external/llama.cpp}"
CHECKPOINT_PATH=""
NAME=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --llama-cpp-dir)
            LLAMA_CPP_DIR="$2"
            shift 2
            ;;
        --tinymoe-dir)
            TINYMOE_DIR="$2"
            LLAMA_CPP_DIR="$TINYMOE_DIR/external/llama.cpp"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

[ -z "$CHECKPOINT_PATH" ] && { echo "Error: --checkpoint required"; exit 1; }
[ -z "$NAME" ] && { echo "Error: --name required"; exit 1; }
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="${TINYMOE_DIR:-$HOME/Documents/tinymoe}/gguf"

OUTPUT_FILE_F32="$OUTPUT_DIR/$NAME-f32.gguf"
OUTPUT_FILE_Q4_0="$OUTPUT_DIR/$NAME-q4_0.gguf"

mkdir -p "$OUTPUT_DIR"

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
[ ! -f "$CONVERT_SCRIPT" ] && CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert.py"

QUANTIZE_TOOL="$LLAMA_CPP_DIR/build/bin/llama-quantize"
[ ! -f "$QUANTIZE_TOOL" ] && QUANTIZE_TOOL="$LLAMA_CPP_DIR/llama-quantize"

python3 "$CONVERT_SCRIPT" "$CHECKPOINT_PATH" --outfile "$OUTPUT_FILE_F32" --outtype f32
"$QUANTIZE_TOOL" "$OUTPUT_FILE_F32" "$OUTPUT_FILE_Q4_0" Q4_0
