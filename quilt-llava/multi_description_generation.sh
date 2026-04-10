#!/bin/bash
echo "Launching 4 parallel tasks..."

PROJECT_ROOT="/home/yuhaowang/project/wsicap/PathAgent/quilt-llava" # change this to your project root directory
DATASET_ROOT="/data/yuhaowang/processed_wsi/TCGA-BRCA" # change this to your dataset root directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

GPUS=(4 5 6 7) # change this to your GPU IDs for each task


for i in {1..4}
do
    GPU_ID=${GPUS[$((i-1))]}
    echo "Starting Task $i on GPU $GPU_ID..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/description_generation.py" \
        --model-path "/data/yuhaowang/cache/qllava_ckpt" \
        --image-dir "$DATASET_ROOT/patches_output" \
        --output-json "$DATASET_ROOT/desc/patches_descriptions${i}.json" \
        --slide-list "$DATASET_ROOT/split_name/slides_part${i}.txt" \
        --load-4bit > "$LOG_DIR/task${i}.log" 2>&1 &
done

echo "✅ All tasks started successfully. Please check $LOG_DIR for logs."
