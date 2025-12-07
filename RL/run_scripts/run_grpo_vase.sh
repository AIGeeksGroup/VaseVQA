#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# Data path configuration - Vase task
data_paths="${REPO_HOME}/../Data/data_train_single_llava_vasevl_v9.json"
image_folders="${REPO_HOME}/../Data"
model_path="${REPO_HOME}/../Models/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=False
reward_method="vase"  # Use vase reward method, supports JSON format parsing

echo "data_paths: $data_paths"
echo "image_folders: $image_folders"
echo "reward_method: $reward_method"

export EXP_NAME="Vase-RL"
TASK_TYPE="vase"

cd ${REPO_HOME}/src/open-r1-multimodal
export PYTHONPATH="${REPO_HOME}/src/open-r1-multimodal/src:${PYTHONPATH}"
export DEBUG_MODE="true"
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

export WANDB_DISABLED=False
export WANDB_PROJECT="vase-rl-and-sft"
# export WANDB_API_KEY="......"

CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12353" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --reward_method $reward_method \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --max_completion_length 1024 \
    --reward_funcs vase_action vase_format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    # --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \

echo "Training completed for ${EXP_NAME}"
