from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from PIL import Image

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def setup_distributed():
    """Initialize distributed environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, world_size, rank


def extract_bbox_answer(content):
    """Extract bounding box coordinates from model output"""
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)),
                    int(bbox_match.group(4))]
            return bbox
    return [0, 0, 0, 0]


def iou(box1, box2):
    """Calculate IoU of two bounding boxes"""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    """Scale model output coordinates back to original image size"""
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox


# --- Distributed environment setup ---
local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")
main_rank = 0

# --- Global parameters and path settings ---
REPO_HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if rank == main_rank:
    print(f"Project root directory (REPO_HOME) set to: {REPO_HOME}")

RUN_NAME = "Qwen2.5-VL-3B-Instruct-rec"
DATA_ROOT = os.path.abspath(os.path.join(REPO_HOME, "..", "Dataset/REC-R1/rec_jsons_processed"))
IMAGE_ROOT = os.path.abspath(os.path.join(REPO_HOME, "..", "Dataset/REC-R1"))
OUTPUT_PATH_TEMPLATE = "./logs/rec_results_{DATASET}_{RUN_NAME}_{STEPS}.json"
BSZ = 2
TEST_DATASETS = ['lisa_test']
num_samples = 2000
QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

# --- Define list of steps to run in batch ---
step_values = list(range(200, 1001, 100))  # [200, 300, ..., 1000]

# --- Main loop, run evaluations for different steps serially ---
for steps in step_values:
    if rank == main_rank:
        print("\n" + "=" * 100)
        print(f"Starting evaluation: steps = {steps}")
        print("=" * 100)

    # --- Build model path based on current steps ---
    MODEL_PATH = os.path.join(REPO_HOME, 'checkpoints', 'rl', RUN_NAME, f'checkpoint-{steps}')
    if rank == main_rank:
        print(f"Loading model from path: {MODEL_PATH}")

    # --- Load model and processor for current steps ---
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # --- Perform evaluation for each dataset ---
    for ds in TEST_DATASETS:
        if rank == main_rank:
            print(f"\n--- Processing dataset: {ds} (steps={steps}) ---")

        ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        random.seed(42)
        random.shuffle(data)
        data = data[:num_samples]

        # Split data by rank for distributed evaluation
        per_rank_data = len(data) // world_size
        start_idx = rank * per_rank_data
        end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
        rank_data = data[start_idx:end_idx]

        messages = []
        for x in rank_data:
            image_path = os.path.join(IMAGE_ROOT, x['image'])
            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x['problem'])}
                ]}
            ]
            messages.append(message)

        rank_outputs = []

        # Data processing and inference
        for i in tqdm(range(0, len(messages), BSZ), disable=rank != main_rank, desc=f"Rank {rank} Inference"):
            batch_messages = messages[i:i + BSZ]
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                    batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text, images=image_inputs, videos=video_inputs, padding=True,
                padding_side="left", return_tensors="pt"
            ).to(device)

            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)

            for j, output_text in enumerate(batch_output_text):
                input_height = int(inputs['image_grid_thw'][j][1] * 14)
                input_width = int(inputs['image_grid_thw'][j][2] * 14)
                image = Image.open(batch_messages[j][0]['content'][0]['image'].split("file://")[1])
                image_width, image_height = image.size
                rank_outputs.append((output_text, input_height, input_width, image_height, image_width))

        # print(f"Rank {rank} completed processing {len(rank_outputs)} samples")

        # Gather results from all ranks
        all_outputs = [None] * len(data)
        rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, rank_results)

        # Main process collects and processes all results
        if rank == main_rank:
            for results in gathered_results:
                for idx, output in results:
                    all_outputs[idx] = output

            final_output = []
            correct_number = 0
            for input_example, model_output in zip(data, all_outputs):
                original_output, input_height, input_width, image_height, image_width = model_output
                ground_truth = input_example['solution']
                model_answer = extract_bbox_answer(original_output)
                resized_model_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)

                correct = 1 if iou(resized_model_answer, ground_truth) > 0.5 else 0
                correct_number += correct

                result = {
                    'image': input_example['image'],
                    'question': input_example['problem'],
                    'ground_truth': ground_truth,
                    'model_output': original_output,
                    'input_size': (input_height, input_width),
                    'image_size': (image_height, image_width),
                    'extracted_answer': resized_model_answer,
                    'correct': correct
                }
                final_output.append(result)

            accuracy = correct_number / len(data) * 100
            print(f"\nDataset {ds} accuracy at steps={steps}: {accuracy:.2f}%")

            output_path = OUTPUT_PATH_TEMPLATE.format(DATASET=ds, RUN_NAME=RUN_NAME, STEPS=steps)
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, "w") as f:
                json.dump({'accuracy': accuracy, 'results': final_output}, f, indent=2)
            print(f"Results saved to: {output_path}")

        # Synchronize all processes
        dist.barrier()

    # --- Clean GPU memory to prepare for next loop ---
    if rank == main_rank:
        print(f"\nCompleted all evaluations for steps={steps}, cleaning memory...")
    del model
    del processor
    torch.cuda.empty_cache()
    dist.barrier()

if rank == main_rank:
    print("\nAll evaluation tasks completed.")