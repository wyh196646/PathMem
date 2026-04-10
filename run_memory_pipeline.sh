#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-all}"

PROJECT_DIR="/home/yuhaowang/project/wsicap/PathAgent"
PLIP_LIB_PATH="./plip"
QWEN_CKPT="/data/yuhaowang/cache/qwen3b/"
PLIP_CKPT="/data/yuhaowang/cache/plip"
PATHO_R1_CKPT="/data/yuhaowang/cache/Patho-R1-7B"

DESCRIPTIONS_FILE="/data/yuhaowang/processed_wsi/TCGA-BRCA/desc/patches_descriptions.json"
TRAIN_QUESTIONS_FILE="/data/yuhaowang/wsi-vqa/WsiVQA_train.json"
TEST_QUESTIONS_FILE="/data/yuhaowang/wsi-vqa/WsiVQA_test.json"
FEATURE_DIR="/data/yuhaowang/processed_wsi/TCGA-BRCA/img_features"
PATCH_ROOT="/data/yuhaowang/processed_wsi/TCGA-BRCA/patches_output"

TRAIN_SAVE_DIR="/data/yuhaowang/processed_wsi/TCGA-BRCA/results/memory_build_train"
TEST_SAVE_DIR="/data/yuhaowang/processed_wsi/TCGA-BRCA/results/memory_inference_test"
MEMORY_BANK_PATH="/data/yuhaowang/processed_wsi/TCGA-BRCA/results/diagnostic_memory.json"

DATASET_NAME="wsi_vqa"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
TRAIN_MAX_CASES="${TRAIN_MAX_CASES:-1000}"
TEST_MAX_CASES="${TEST_MAX_CASES:-}"

mkdir -p "${TRAIN_SAVE_DIR}" "${TEST_SAVE_DIR}"
cd "${PROJECT_DIR}"

run_build() {
  python pathagent.py \
    --plip_lib_path "${PLIP_LIB_PATH}" \
    --qwen_ckpt "${QWEN_CKPT}" \
    --plip_ckpt "${PLIP_CKPT}" \
    --patho_r1_ckpt "${PATHO_R1_CKPT}" \
    --descriptions_file "${DESCRIPTIONS_FILE}" \
    --questions_file "${TRAIN_QUESTIONS_FILE}" \
    --feature_dir "${FEATURE_DIR}" \
    --patch_root "${PATCH_ROOT}" \
    --save_dir "${TRAIN_SAVE_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --memory_mode build \
    --memory_bank_path "${MEMORY_BANK_PATH}" \
    --gpu_ids "${GPU_IDS}" \
    --workers_per_gpu "${WORKERS_PER_GPU}" \
    ${TRAIN_MAX_CASES:+--max_cases "${TRAIN_MAX_CASES}"}
}

run_test() {
  python pathagent.py \
    --plip_lib_path "${PLIP_LIB_PATH}" \
    --qwen_ckpt "${QWEN_CKPT}" \
    --plip_ckpt "${PLIP_CKPT}" \
    --patho_r1_ckpt "${PATHO_R1_CKPT}" \
    --descriptions_file "${DESCRIPTIONS_FILE}" \
    --questions_file "${TEST_QUESTIONS_FILE}" \
    --feature_dir "${FEATURE_DIR}" \
    --patch_root "${PATCH_ROOT}" \
    --save_dir "${TEST_SAVE_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --memory_mode inference \
    --memory_bank_path "${MEMORY_BANK_PATH}" \
    --memory_top_k 3 \
    --memory_lambda_slide 0.5 \
    --memory_lambda_question 0.5 \
    --expand_alpha 0.6 \
    --expand_beta 0.3 \
    --expand_gamma 0.1 \
    --zoom_eta 0.7 \
    --zoom_mu 0.3 \
    --gpu_ids "${GPU_IDS}" \
    --workers_per_gpu "${WORKERS_PER_GPU}" \
    ${TEST_MAX_CASES:+--max_cases "${TEST_MAX_CASES}"}
}

run_all() {
  python test.py \
    --plip_lib_path "${PLIP_LIB_PATH}" \
    --qwen_ckpt "${QWEN_CKPT}" \
    --plip_ckpt "${PLIP_CKPT}" \
    --patho_r1_ckpt "${PATHO_R1_CKPT}" \
    --descriptions_file "${DESCRIPTIONS_FILE}" \
    --train_questions_file "${TRAIN_QUESTIONS_FILE}" \
    --test_questions_file "${TEST_QUESTIONS_FILE}" \
    --feature_dir "${FEATURE_DIR}" \
    --patch_root "${PATCH_ROOT}" \
    --train_save_dir "${TRAIN_SAVE_DIR}" \
    --test_save_dir "${TEST_SAVE_DIR}" \
    --memory_bank_path "${MEMORY_BANK_PATH}" \
    --gpu_ids "${GPU_IDS}" \
    --workers_per_gpu "${WORKERS_PER_GPU}" \
    ${TRAIN_MAX_CASES:+--train_max_cases "${TRAIN_MAX_CASES}"} \
    ${TEST_MAX_CASES:+--test_max_cases "${TEST_MAX_CASES}"}
}

case "${MODE}" in
  build)
    run_build
    ;;
  test)
    run_test
    ;;
  all)
    run_all
    ;;
  *)
    echo "Usage: bash run_memory_pipeline.sh [build|test|all]"
    exit 1
    ;;
esac
