#!/bin/bash

# Usage: ./convert_model.sh --model_name biodatlab/whisper-th-medium-combined --output_name whisper_th_medium_combined

# Default values
INFERENCE_PRECISION="float16"
WEIGHT_ONLY_PRECISION="int8"
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --output_name) OUTPUT_NAME="$2"; shift ;;
        --inference_precision) INFERENCE_PRECISION="$2"; shift ;;
        --weight_only_precision) WEIGHT_ONLY_PRECISION="$2"; shift ;;
        --max_beam_width) MAX_BEAM_WIDTH="$2"; shift ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if mandatory arguments are provided
if [ -z "$MODEL_NAME" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: ./convert_model.sh --model_name <model_name> --output_name <output_name> [--inference_precision <precision>] [--weight_only_precision <precision>] [--max_beam_width <width>] [--max_batch_size <size>]"
    exit 1
fi

# Set directories
CHECKPOINT_DIR=$OUTPUT_NAME
OUTPUT_DIR=$OUTPUT_NAME

# Step 1: Convert Hugging Face model to OpenAI format
python convert_hf_to_openai.py --model_name "$MODEL_NAME" --output_name "$OUTPUT_NAME"

# Step 2: Convert checkpoint
python convert_checkpoint.py --model_name "$OUTPUT_NAME" --use_weight_only --weight_only_precision "$WEIGHT_ONLY_PRECISION" --output_dir "$CHECKPOINT_DIR"

# Step 3: Build encoder with TensorRT
trtllm-build --checkpoint_dir ${CHECKPOINT_DIR}/encoder \
             --output_dir ${OUTPUT_DIR}/encoder \
             --paged_kv_cache disable \
             --moe_plugin disable \
             --enable_xqa disable \
             --use_custom_all_reduce disable \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --gemm_plugin disable \
             --bert_attention_plugin ${INFERENCE_PRECISION} \
             --remove_input_padding disable

# Step 4: Build decoder with TensorRT
trtllm-build --checkpoint_dir ${CHECKPOINT_DIR}/decoder \
             --output_dir ${OUTPUT_DIR}/decoder \
             --paged_kv_cache disable \
             --moe_plugin disable \
             --enable_xqa disable \
             --use_custom_all_reduce disable \
             --max_beam_width ${MAX_BEAM_WIDTH} \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --max_output_len 114 \
             --max_input_len 14 \
             --max_encoder_input_len 1500 \
             --gemm_plugin ${INFERENCE_PRECISION} \
             --bert_attention_plugin ${INFERENCE_PRECISION} \
             --gpt_attention_plugin ${INFERENCE_PRECISION} \
             --remove_input_padding disable
