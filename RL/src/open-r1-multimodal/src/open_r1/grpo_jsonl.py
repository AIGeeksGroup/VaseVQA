# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
import torch
import torch.nn.functional as F
from json_repair import repair_json

from open_r1.vlm_modules import *

from typing import Tuple
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer, AutoModel

from openai import OpenAI

logger = logging.get_logger(__name__)

# Global variable definitions
vase_reward_model = None

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, \
    monkey_patch_torch_load

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

tokenizer = None


def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )


def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        'answer', 'correct', 'choose', 'select', 'right',
        'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos - 20):min(len(text), pos + 20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos + 1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]


def evaluate_answer_similarity(student_answer, ground_truth):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": "You are a evaluation expert. First, analyze the student's response to identify and extract their final answer. Then, compare the extracted answer with the correct solution. Output ONLY '1.0' if the extracted answer matches the correct solution in meaning, or '0.0' if the student's response does not contain a clear or correct answer. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student's response: {student_answer}\nCorrect solution: {ground_truth}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return float(result)

    except Exception as e:
        logger.error(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer == ground_truth else 0.0


def llm_reward(content, sol, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth)


def mcq_reward(content, sol, **kwargs):
    # For multiple choice, extract and compare choices
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    has_choices = extract_choice(ground_truth)
    correct_choice = has_choices.upper() if has_choices else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def yes_no_reward(content, sol, **kwargs):
    content = content.lower()
    sol = sol.lower()

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    ground_yes_no = re.search(r'(yes|no)', ground_truth)
    ground_yes_no = ground_yes_no.group(1) if ground_yes_no else ''
    student_yes_no = re.search(r'(yes|no)', student_answer)
    student_yes_no = student_yes_no.group(1) if student_yes_no else ''

    reward = 1.0 if ground_yes_no == student_yes_no else 0.0

    return reward


# score_type: 0 for mAP, 1 for mAP 50
def calculate_map(pred_bbox_list, gt_bbox_list, score_type=0):
    # Calculate mAP

    # Initialize COCO object for ground truth
    gt_json = {"annotations": [], "images": [], "categories": []}
    gt_json["images"] = [{
        "id": 0,
        "width": 2048,
        "height": 2048,
        "file_name": "image_0.jpg"
    }]

    gt_json["categories"] = []

    cats2id = {}
    cat_count = 0
    for idx, gt_bbox in enumerate(gt_bbox_list):
        if gt_bbox["label"] not in cats2id:
            cats2id[gt_bbox["label"]] = cat_count
            gt_json["categories"].append({
                "id": cat_count,
                "name": gt_bbox["label"]
            })
            cat_count += 1

        gt_json["annotations"].append({
            "id": idx + 1,
            "image_id": 0,
            "category_id": cats2id[gt_bbox["label"]],
            "bbox": [gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][1], gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0],
                     gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]],
            "area": (gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0]) * (gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]),
            "iscrowd": 0
        })
    coco_gt = COCO(gt_json)

    dt_json = []
    for idx, pred_bbox in enumerate(pred_bbox_list):
        try:
            dt_json.append({
                "image_id": 0,
                "category_id": cats2id[pred_bbox["label"]],
                "bbox": [pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][1],
                         pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0],
                         pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1]],
                "score": 1.0,
                "area": (pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0]) * (
                        pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1])
            })
        except:
            pass

    if len(dt_json) == 0:
        return 0.0

    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[score_type]


def map_reward(content, sol, length_reward=False, score_type=0, **kwargs):
    """
    Calculate mean average precision (mAP) reward between predicted and ground truth bounding boxes.
    """
    # Extract JSON content between ```json tags
    pattern = r'```json(.*?)```'
    json_match = re.findall(pattern, sol, re.DOTALL)
    bbox_json = json_match[-1].strip() if json_match else None

    # Parse ground truth JSON to get bbox list
    gt_bbox_list = []
    if bbox_json:
        bbox_data = json.loads(bbox_json)
        gt_bbox_list = [item for item in bbox_data]

    # Parse predicted JSON to get bbox list
    pred_bbox_list = []
    json_match = re.findall(pattern, content, re.DOTALL)
    if json_match:
        try:
            bbox_data = json.loads(json_match[-1].strip())
            pred_bbox_list = [item for item in bbox_data]
        except:
            # Return empty list if JSON parsing fails
            pred_bbox_list = []

    # Calculate mAP if both prediction and ground truth exist
    if len(pred_bbox_list) > 0 and len(gt_bbox_list) > 0:
        bbox_reward = calculate_map(pred_bbox_list, gt_bbox_list, score_type=score_type)
    elif len(pred_bbox_list) == 0 and len(gt_bbox_list) == 0:
        bbox_reward = 1.0
    else:
        bbox_reward = 0.0

    if length_reward:
        # Calculate length penalty based on ratio of ground truth to predicted bounding boxes
        gt_length = len(gt_bbox_list)
        pred_length = len(pred_bbox_list)
        # Full score if prediction has fewer boxes than ground truth, otherwise penalize proportionally
        length_score = 1.0 if gt_length >= pred_length else gt_length / pred_length
        return bbox_reward * length_score
    else:
        return bbox_reward


def od_reward(content, sol, score_type=0, **kwargs):
    """
    Calculate reward for object detection task by comparing predicted and ground truth answers.
    """
    # Pattern to extract content between <answer> tags
    match_pattern = r'<answer>(.*?)</answer>'

    # Extract ground truth answer
    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None

    # Extract predicted answer (using last match if multiple)
    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    # Return 0 if no prediction
    if student_answer is None:
        return 0.0
    # Return 1 if both prediction and ground truth are None
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    # Otherwise calculate mAP between prediction and ground truth
    else:
        return map_reward(student_answer, ground_truth, score_type=score_type)


def odLength_reward(content, sol, **kwargs):
    """
    Calculate reward for object detection task with length penalty.
    """
    # Pattern to extract content between <answer> tags
    match_pattern = r'<answer>(.*?)</answer>'

    # Extract ground truth answer
    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None
    # Extract predicted answer (using last match if multiple)
    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    # Return 0 if no prediction
    if student_answer is None:
        return 0.0
    # Return 1 if both prediction and ground truth are None
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    # Calculate mAP with length penalty
    else:
        bbox_reward = map_reward(student_answer, ground_truth, length_reward=True, score_type=0)
        return bbox_reward


def iou(box1, box2):
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


def detection_score(content, sol, iou_threshold=0.5, alpha=0.7, beta=0.0, gamma=0.3):
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, clean_text(content), re.DOTALL)
    content_bbox_json = json_match.group(1).strip() if json_match else None
    if content_bbox_json:
        try:
            bbox_data = json.loads(content_bbox_json)
            pred_boxes = [item for item in bbox_data]
        except:
            pred_boxes = []

    else:
        pred_boxes = []

    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, clean_text(sol), re.DOTALL)
    sol_bbox_json = json_match.group(1).strip() if json_match else None
    if sol_bbox_json:
        bbox_data = json.loads(sol_bbox_json)
        gt_boxes = [item for item in bbox_data]
    else:
        gt_boxes = []

    """
    Calculate the comprehensive score for object detection
    """
    # Handle edge cases
    if len(gt_boxes) == 0:
        return 1.0 if not pred_boxes else 0.0

    if len(pred_boxes) == 0:
        return 0.0

    # Initialize matching results
    matches = []  # Store matched pairs of predicted and ground truth boxes
    unmatched_preds = list(range(len(pred_boxes)))  # Indices of unmatched predicted boxes
    unmatched_gts = list(range(len(gt_boxes)))  # Indices of unmatched ground truth boxes

    # Calculate IoU matrix between all predicted and ground truth boxes
    iou_matrix = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        iou_row = []
        for gt_idx, gt_box in enumerate(gt_boxes):
            try:
                curr_iou = iou(pred_box["bbox_2d"], gt_box["bbox_2d"])
            except:
                curr_iou = 0.0
            iou_row.append(curr_iou)
        iou_matrix.append(iou_row)

    # Greedy matching: find the best match for each predicted box
    while unmatched_preds and unmatched_gts:
        # Find the maximum IoU
        max_iou = -1
        max_pred_idx = -1
        max_gt_idx = -1

        for pred_idx in unmatched_preds:
            for gt_idx in unmatched_gts:
                curr_iou = iou_matrix[pred_idx][gt_idx]
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_pred_idx = pred_idx
                    max_gt_idx = gt_idx

        # Stop matching if the maximum IoU is below the threshold
        if max_iou < iou_threshold:
            break

        # Record matching results
        try:
            pred_label = pred_boxes[max_pred_idx]["label"].lower()
        except:
            pred_box = ""
        try:
            gt_label = gt_boxes[max_gt_idx]["label"].lower()
        except:
            gt_label = ""
        label_correct = (pred_label == gt_label)

        if label_correct:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": max_iou,
                "label_correct": label_correct
            })
        else:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": 0,
                "label_correct": label_correct
            })

        # Remove matched boxes from the unmatched list
        unmatched_preds.remove(max_pred_idx)
        unmatched_gts.remove(max_gt_idx)

    # Calculate position accuracy score (average IoU)
    position_score = sum(m["iou"] for m in matches) / len(gt_boxes) if matches else 0.0

    # Calculate label accuracy score
    label_score = sum(1.0 for m in matches if m["label_correct"]) / len(gt_boxes) if matches else 0.0

    # Calculate completeness score (considering missed and false detections)
    miss_rate = len(unmatched_gts) / len(gt_boxes)
    false_alarm_rate = len(unmatched_preds) / len(pred_boxes) if pred_boxes else 0.0

    # Completeness score = 1 - (miss rate + false alarm rate) / 2
    completeness_score = 1.0 - (miss_rate + false_alarm_rate) / 2.0

    # Calculate the final comprehensive score
    final_score = (
                          alpha * position_score +
                          beta * label_score +
                          gamma * completeness_score
                  ) / (alpha + beta + gamma)

    return final_score


def cosine_reward(content, tokenizer, acc_reward, **kwargs):
    # [https://arxiv.org/abs/2502.03373](https://arxiv.org/abs/2502.03373)
    min_len_value_wrong = 0.0
    max_len_value_wrong = -0.5
    min_len_value_correct = 1.0
    max_len_value_correct = 0.5
    cosine_max_len = 1024

    gen_len = len(tokenizer.encode(content))
    acc_reward = 1.0
    is_correct = acc_reward >= 0.7

    if is_correct:
        # Swap min/max for correct answers
        min_value = max_len_value_correct
        max_value = min_len_value_correct
    else:
        min_value = min_len_value_wrong
        max_value = max_len_value_wrong

    reward = max_value - (max_value - min_value) * (1 - math.cos(gen_len * math.pi / cosine_max_len)) / 2

    return reward


def repetition_reward(content, **kwargs):
    max_penalty = -1.0

    if content == '':
        return 0.0

    # First, try to extract explicitly marked JSON sections
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, content, re.DOTALL)

    if json_match:
        bbox_json = json_match.group(1).strip()
    else:
        # If no explicitly marked JSON is found, try to find any possible JSON sections
        pattern = r'```(.*?)```'
        json_match = re.search(pattern, content, re.DOTALL)
        bbox_json = json_match.group(1).strip() if json_match else None

        # If still not found, try to find possible JSON array sections
        if not bbox_json:
            pattern = r'\[\s*{.*?"bbox_2d".*?"label".*?}\s*\]'
            json_match = re.search(pattern, content, re.DOTALL)
            bbox_json = json_match.group(0) if json_match else None

    # Try to parse JSON data
    if bbox_json:
        try:
            # Try direct parsing
            data = json.loads(bbox_json)
        except json.JSONDecodeError:
            try:
                # If direct parsing fails, try using json_repair to repair
                repaired_json = repair_json(bbox_json)
                data = json.loads(repaired_json)
            except:
                # If repair also fails, switch to plain text processing
                data = None
        if data and isinstance(data, list):
            # Ensure data is in list format
            try:
                # For JSON data, set ngram_size to 1
                ngram_size = 1
                # Combine 'bbox_2d' and 'label' of each object into a string
                items = []
                for item in data:
                    if 'bbox_2d' in item and 'label' in item:
                        items.append(f"{item['bbox_2d']}_{item['label']}")

                @staticmethod
                def zipngram(text: list, ngram_size: int):
                    return zip(*[text[i:] for i in range(ngram_size)])

                ngrams = set()
                total = 0

                for ng in zipngram(items, ngram_size):
                    ngrams.add(ng)
                    total += 1

                if total == 0:
                    return 0.0

                scaling = 1 - len(ngrams) / total
                reward = scaling * max_penalty

                return reward
            except KeyError:
                # If necessary keys are missing, switch to plain text processing
                pass

    # If no JSON section is found or JSON processing fails, treat as plain text
    ngram_size = 6

    if len(content.split()) < ngram_size:
        return 0.0

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    ngrams = set()
    total = 0

    for ng in zipngram(content, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward


def repetition_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = repetition_reward(content)
        rewards.append(reward)

    return rewards


def clean_json_comments(json_str):
    """Clean comments from JSON string"""
    lines = json_str.split('\n')
    cleaned_lines = []
    for line in lines:
        # Find # comment position, but consider # inside strings
        in_string = False
        quote_char = None
        comment_pos = -1

        for i, char in enumerate(line):
            if char in ['"', "'"] and (i == 0 or line[i - 1] != '\\'):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None
            elif char == '#' and not in_string:
                comment_pos = i
                break

        if comment_pos >= 0:
            # Remove comment part, but keep necessary whitespace
            cleaned_line = line[:comment_pos].rstrip()
        else:
            cleaned_line = line

        cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)


def stardojo_action_reward(content, sol, **kwargs):
    """Enhanced Stardojo action reward with format-robust comparison."""
    import json
    import re

    def normalize_action(action_str):
        """Normalize action string, handle format differences"""
        if not isinstance(action_str, str):
            return str(action_str) if action_str is not None else ""

        # Extract function name
        func_match = re.match(r'(\w+)\s*\(', action_str)
        if not func_match:
            return action_str

        func_name = func_match.group(1)

        # Extract parameters part
        # Remove function name and parentheses, get parameters part
        params_str = action_str[action_str.find('(') + 1:action_str.rfind(')')]

        # Parse parameters: handle various formats
        params = {}
        positional_args = []

        if params_str.strip():
            # Split parameters, considering nested quotes
            param_parts = []
            current_part = ""
            paren_level = 0
            in_quotes = False
            quote_char = None

            for char in params_str + ',':  # Add comma to ensure last parameter is processed
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif char == '(' and not in_quotes:
                    paren_level += 1
                elif char == ')' and not in_quotes:
                    paren_level -= 1
                elif char == ',' and paren_level == 0 and not in_quotes:
                    if current_part.strip():
                        param_parts.append(current_part.strip())
                    current_part = ""
                    continue

                current_part += char

            # Parse each parameter
            for part in param_parts:
                part = part.strip()
                if '=' in part:
                    # Named parameter: key=value
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes
                    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                        value = value[1:-1]
                    params[key] = value
                else:
                    # Positional parameter
                    value = part
                    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                        value = value[1:-1]
                    positional_args.append(value)

        # Normalize to unified format: func_name(param1=value1, param2=value2)
        if positional_args or params:
            # Standardize parameter names based on function type
            if func_name in ['use', 'interact'] and positional_args:
                params['direction'] = positional_args[0]
            elif func_name == 'move' and len(positional_args) >= 2:
                params['x'] = positional_args[0]
                params['y'] = positional_args[1]
            elif func_name == 'choose_item' and positional_args:
                params['slot_index'] = positional_args[0]
            elif func_name == 'craft' and positional_args:
                params['item'] = positional_args[0]
            elif func_name == 'menu' and len(positional_args) >= 2:
                params['option'] = positional_args[0]
                params['menu_name'] = positional_args[1]
            elif func_name == 'choose_option':
                if len(positional_args) >= 1:
                    params['option_index'] = positional_args[0]
                if len(positional_args) >= 2:
                    params['quantity'] = positional_args[1]
                if len(positional_args) >= 3:
                    params['direction'] = positional_args[2]

            # Build normalized string
            param_strs = []
            for key, value in sorted(params.items()):
                param_strs.append(f"{key}={value}")

            return f"{func_name}({', '.join(param_strs)})"
        else:
            return f"{func_name}()"

    def compare_actions(pred_action, gt_action):
        """Compare two actions, return similarity score"""
        pred_norm = normalize_action(pred_action)
        gt_norm = normalize_action(gt_action)

        # Ensure both are strings
        if not isinstance(pred_norm, str):
            pred_norm = str(pred_norm) if pred_norm is not None else ""
        if not isinstance(gt_norm, str):
            gt_norm = str(gt_norm) if gt_norm is not None else ""

        # Exact match
        if pred_norm == gt_norm:
            return 1.0

        # Function name match
        pred_func = re.match(r'(\w+)', pred_norm)
        gt_func = re.match(r'(\w+)', gt_norm)

        if pred_func and gt_func and pred_func.group(1) == gt_func.group(1):
            # Same function name, give partial score
            return 0.5

        return 0.0

    # Extract ground truth actions
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    if not sol_match:
        raise ValueError(f"No <answer> tag found in solution: {sol}")
    ground_truth = sol_match.group(1).strip()
    cleaned_ground_truth = clean_json_comments(ground_truth)
    try:
        gt_actions = json.loads(cleaned_ground_truth)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse ground truth actions JSON. Original: {repr(ground_truth)}. Cleaned: {repr(cleaned_ground_truth)}. Error: {e}",
            e.doc, e.pos
        )

    # Extract predicted actions
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if not content_match:
        raise ValueError(f"No <answer> tag found in content: {content}")

    answer_content = content_match.group(1).strip()
    cleaned_content = clean_json_comments(answer_content)

    try:
        pred_actions = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse predicted actions JSON after cleaning. Original: {repr(answer_content)}. Cleaned: {repr(cleaned_content)}. Error: {e}",
            e.doc, e.pos
        )

    if not isinstance(pred_actions, list):
        raise TypeError(f"Predicted actions must be a list, got {type(pred_actions)}: {pred_actions}")
    if not isinstance(gt_actions, list):
        raise TypeError(f"Ground truth actions must be a list, got {type(gt_actions)}: {gt_actions}")

    # Calculate action similarity
    total_score = 0.0
    min_length = min(len(pred_actions), len(gt_actions))
    max_length = max(len(pred_actions), len(gt_actions))

    # If either list is empty
    if max_length == 0:
        return 1.0 if min_length == 0 else 0.0

    # Compare actions one by one
    for i in range(min_length):
        action_score = compare_actions(pred_actions[i], gt_actions[i])
        total_score += action_score

    # Penalty for length mismatch
    length_penalty = (max_length - min_length) * 0.1
    total_score = max(0.0, total_score - length_penalty)

    # Normalize score (maximum 1.0)
    final_score = min(1.0, total_score / max_length)
    return final_score


def stardojo_action_rewards(completions, solution, **kwargs):
    """Stardojo action sequence reward function (batch processing)"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            reward = stardojo_action_reward(content, sol)
        except (ValueError, json.JSONDecodeError, TypeError):
            # Return 0 score on format error, but don't affect other reward calculations
            reward = 0.0
        rewards.append(reward)

    return rewards


def haversine_distance(coord1, coord2):
    """Calculate distance between two GPS coordinates (kilometers)"""
    import math

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Earth radius (kilometers)
    r = 6371

    return c * r


def geo_action_reward(content, sol, **kwargs):
    """GPS geolocation reward function"""
    import re
    import json

    def parse_geo_ground_truth(sol_text):
        """Extract true/false markers and coordinates from All-Geo-HH-Train-SFT-R1-V3.json style solution.
        """
        # Extract <answer> content
        sol_inner_match = re.search(r'<answer>(.*?)</answer>', sol_text, re.DOTALL)
        inner = sol_inner_match.group(1).strip() if sol_inner_match else sol_text.strip()

        # Refusal detection
        if re.search(r'\brefusal\b', inner, re.IGNORECASE):
            return {'is_real': False, 'coordinates': None}

        # Remove possible code fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', inner.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        # Replace single quotes with double quotes, nan/NaN with null for JSON parsing
        json_like = cleaned.replace("'", '"')
        json_like = re.sub(r'\bNaN\b|\bnan\b', 'null', json_like)

        # Parse as dict, prioritize LOC/coordinates/location keys
        coords = None
        try:
            data = json.loads(json_like)
            if isinstance(data, dict):
                loc = data.get('LOC') or data.get('coordinates') or data.get('location')
                if isinstance(loc, list) and len(loc) == 2:
                    try:
                        coords = (float(loc[0]), float(loc[1]))
                    except Exception:
                        coords = None
                return {'is_real': True, 'coordinates': coords}
        except json.JSONDecodeError:
            # Fallback: use coordinate regex as backup
            m = re.search(r'\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]', cleaned)
            if m:
                try:
                    coords = (float(m.group(1)), float(m.group(2)))
                except Exception:
                    coords = None
                return {'is_real': True if coords is not None else None, 'coordinates': coords}

        return {'is_real': None, 'coordinates': None}

    # Parse ground truth (from @geo dataset's solution)
    gt_info = parse_geo_ground_truth(sol)
    gt_is_real = gt_info['is_real'] if gt_info['is_real'] is not None else True
    gt_coords = gt_info['coordinates']

    # Parse prediction result
    extracted_info = extract_geo_content(content)

    # Determine prediction true/false and coordinates
    pred_is_real = None
    pred_coords = extracted_info['coordinates']

    if extracted_info['is_refusal']:
        pred_is_real = False
        pred_coords = None
    elif pred_coords is not None:
        pred_is_real = True
    elif (
            extracted_info['country'] and
            extracted_info['country'].lower() not in ['none', 'null', 'refusal', 'unknown']
    ) or (
            extracted_info['state'] and str(extracted_info['state']).lower() not in ['none', 'null', 'nan']
    ) or (
            extracted_info['city'] and str(extracted_info['city']).lower() not in ['none', 'null', 'nan']
    ):
        # Has geo hierarchy info but missing coords, also treat as real image (coordinates missing)
        pred_is_real = True
        pred_coords = None
    else:
        pred_is_real = None

    # Calculate reward
    if pred_is_real is None:
        return 0.0
    if not gt_is_real and not pred_is_real:
        return 0.5
    if not gt_is_real and pred_is_real:
        return 0.0
    if gt_is_real and not pred_is_real:
        return 0.0

    # Both are real images
    if gt_coords is None or pred_coords is None:
        return 0.0

    distance_km = haversine_distance(gt_coords, pred_coords)
    reward = max(0.5, 1 - distance_km / 2000)
    return reward


def extract_geo_content(content):
    """General function to extract geographic information from content"""
    import re
    import json

    extracted_info = {
        'country': None,
        'state': None,
        'city': None,
        'coordinates': None,
        'is_refusal': False
    }

    # Method 1: Try parsing JSON format
    try:
        json_pattern = r"\{[^}]+\}"
        json_match = re.search(json_pattern, content)
        if json_match:
            json_str = json_match.group(0)
            # Handle various quote formats
            json_str = json_str.replace("'", '"')
            data = json.loads(json_str)

            if isinstance(data, dict):
                extracted_info['country'] = data.get('country')
                extracted_info['state'] = data.get('state')
                extracted_info['city'] = data.get('city')

                # Handle coordinates
                loc = data.get('LOC') or data.get('coordinates') or data.get('location')
                if loc and isinstance(loc, list) and len(loc) == 2:
                    try:
                        extracted_info['coordinates'] = (float(loc[0]), float(loc[1]))
                    except (ValueError, TypeError):
                        pass

                return extracted_info
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Method 2: Try parsing key-value pair format (Country: xxx, State: xxx, etc.)
    country_match = re.search(r'Country:\s*([^\n\r,]+)', content, re.IGNORECASE)
    state_match = re.search(r'State:\s*([^\n\r,]+)', content, re.IGNORECASE)
    city_match = re.search(r'City:\s*([^\n\r,]+)', content, re.IGNORECASE)
    coord_match = re.search(r'Coordinates?:\s*([^\n\r,]+)', content, re.IGNORECASE)

    if country_match:
        country_val = country_match.group(1).strip()
        extracted_info['country'] = country_val

        # Check if it's a refusal format
        if 'refusal' in country_val.lower():
            extracted_info['is_refusal'] = True

    if state_match:
        state_val = state_match.group(1).strip()
        extracted_info['state'] = state_val

    if city_match:
        city_val = city_match.group(1).strip()
        extracted_info['city'] = city_val

    if coord_match:
        coord_val = coord_match.group(1).strip()
        # Try extracting coordinate numbers
        coord_numbers = re.findall(r'([+-]?\d+\.?\d*)', coord_val)
        if len(coord_numbers) >= 2:
            try:
                extracted_info['coordinates'] = (float(coord_numbers[0]), float(coord_numbers[1]))
            except ValueError:
                pass

    # Method 3: More lenient coordinate extraction
    if not extracted_info['coordinates']:
        # Find any pattern like [number, number]
        bracket_coords = re.search(r'\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]', content)
        if bracket_coords:
            try:
                extracted_info['coordinates'] = (float(bracket_coords.group(1)), float(bracket_coords.group(2)))
            except ValueError:
                pass

    return extracted_info


def geo_format_reward(content, sol, **kwargs):
    """GPS geolocation format reward function - lenient version"""

    # Extract content
    extracted_info = extract_geo_content(content)

    # Give reward as long as any valid information can be extracted
    has_content = (
            extracted_info['country'] is not None or
            extracted_info['state'] is not None or
            extracted_info['city'] is not None or
            extracted_info['coordinates'] is not None
    )

    if has_content:
        return 1.0
    else:
        return 0.0


def geo_action_rewards(completions, solution, **kwargs):
    """GPS geolocation reward function (batch processing)"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            reward = geo_action_reward(content, sol)
        except Exception:
            # Return 0 score on any error
            reward = 0.0
        rewards.append(reward)

    return rewards


def geo_format_rewards(completions, solution, **kwargs):
    """GPS geolocation format reward function (batch processing)"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            reward = geo_format_reward(content, sol)
        except Exception:
            # Return 0 score on any error
            reward = 0.0
        rewards.append(reward)

    return rewards


def vase_action_reward(predictions, questions, ground_truths, model, **kwargs):
    """Vase task action reward function"""
    import os
    global vase_reward_model

    # Load model only on first call
    if vase_reward_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers import util
            print("Loading sentence transformer model...")
            vase_reward_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load sentence transformer model: {e}")
            vase_reward_model = "failed"

    def extract_keywords(text):
        """A simple keyword extractor: extracts uppercase words, numbers and hyphenated number ranges."""
        keywords = set(re.findall(r'\b[A-Z]{3,}\b', text))
        keywords.update(re.findall(r'-\d+\s+to\s+-\d+|\b\d+\b', text))
        if not keywords:
            return set(text.upper().split())
        return keywords

    def calculate_date_iou(pred_str, gt_str):
        """Calculate IoU specifically for date ranges, with partial score rewards."""
        try:
            pred_nums = [int(n) for n in re.findall(r'-?\d+', pred_str)]
            gt_nums = [int(n) for n in re.findall(r'-?\d+', gt_str)]

            # Basic format reward: give format reward if can parse X to Y format
            format_reward = 0.0
            if re.search(r'-?\d+\s+to\s+-?\d+', pred_str) or re.search(r'-?\d+\s*-\s*-?\d+', pred_str):
                format_reward = 0.1  # Correct format reward
            
            # Number matching reward: give partial reward if any year number is correct
            number_reward = 0.0
            if pred_nums and gt_nums:
                for pred_num in pred_nums:
                    for gt_num in gt_nums:
                        if abs(pred_num - gt_num) <= 50:  # Within 50 years error considered partially correct
                            number_reward = max(number_reward, 0.2)
                        elif abs(pred_num - gt_num) <= 100:  # Within 100 years error give small reward
                            number_reward = max(number_reward, 0.1)

            # Complete IoU reward: range overlap degree
            iou_reward = 0.0
            if len(pred_nums) == 2 and len(gt_nums) == 2:
                pred_range = set(range(min(pred_nums), max(pred_nums) + 1))
                gt_range = set(range(min(gt_nums), max(gt_nums) + 1))
                intersection = len(pred_range.intersection(gt_range))
                union = len(pred_range.union(gt_range))
                iou_reward = intersection / union if union > 0 else 0.0

            # Return the highest reward score
            return max(format_reward, number_reward, iou_reward)
        except:
            return 0.0

    def calculate_decoration_entity_reward(pred_str, gt_str):
        """Calculate entity density reward for decoration descriptions."""
        try:
            # Simple entity extraction: extract noun-like words
            import re
            
            # Extract possible entities (capitalized words, proper nouns, etc.)
            pred_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', pred_str))
            gt_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', gt_str))
            
            # Extract descriptive adjectives and nouns
            pred_descriptors = set(re.findall(r'\b(?:red|black|white|blue|gold|silver|painted|decorated|figured|geometric|floral|animal|human|mythological|religious|ceremonial|ritual|classical|archaic|hellenistic|roman|greek|attic|corinthian|ionic)\b', pred_str.lower()))
            gt_descriptors = set(re.findall(r'\b(?:red|black|white|blue|gold|silver|painted|decorated|figured|geometric|floral|animal|human|mythological|religious|ceremonial|ritual|classical|archaic|hellenistic|roman|greek|attic|corinthian|ionic)\b', gt_str.lower()))
            
            # Calculate entity matching degree
            entity_intersection = pred_entities.intersection(gt_entities)
            entity_union = pred_entities.union(gt_entities)
            entity_score = len(entity_intersection) / len(entity_union) if entity_union else 0.0
            
            # Calculate descriptor matching degree
            desc_intersection = pred_descriptors.intersection(gt_descriptors)
            desc_union = pred_descriptors.union(gt_descriptors)
            desc_score = len(desc_intersection) / len(desc_union) if desc_union else 0.0
            
            # Information quantity reward: if prediction contains more relevant information
            info_density_score = 0.0
            if len(pred_entities) > 0 or len(pred_descriptors) > 0:
                info_density_score = 0.1  # Basic information quantity reward
            
            # Combined entity reward
            entity_reward = max(entity_score * 0.6 + desc_score * 0.4, info_density_score)
            return entity_reward
            
        except Exception:
            return 0.0

    def calculate_unified_reward(predicted_answer, ground_truth_answer, question):
        """A unified reward function applicable to all question types."""
        # 1. Fact keyword score
        pred_keywords = extract_keywords(predicted_answer)
        gt_keywords = extract_keywords(ground_truth_answer)
        intersecting_keywords = pred_keywords.intersection(gt_keywords)
        union_keywords = pred_keywords.union(gt_keywords)
        keyword_score = len(intersecting_keywords) / len(union_keywords) if union_keywords else 0.0

        # 2. Date IoU score (for date questions)
        date_iou_score = calculate_date_iou(predicted_answer, ground_truth_answer)
        
        # 3. Decoration entity reward (for decoration questions)
        decoration_entity_score = calculate_decoration_entity_reward(predicted_answer, ground_truth_answer)

        # 4. Semantic similarity calculation
        semantic_score = 0.0
        if vase_reward_model != "failed" and vase_reward_model is not None:
            try:
                from sentence_transformers import util
                embeddings = vase_reward_model.encode([predicted_answer, ground_truth_answer])
                semantic_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                semantic_score = (semantic_score + 1) / 2  # Normalize to [0, 1]
            except Exception as e:
                print(f"Semantic similarity calculation failed: {e}")
                semantic_score = 0.0

        # 5. Final reward combination - optimized for different tasks
        if "date" in question.lower():
            # Date questions: prioritize IoU, then keywords, finally semantic
            final_reward = 0.7 * date_iou_score + 0.2 * keyword_score + 0.1 * semantic_score
        elif "decoration" in question.lower():
            # Decoration questions: prioritize entity matching, then semantic similarity, finally keywords
            final_reward = 0.5 * decoration_entity_score + 0.4 * semantic_score + 0.1 * keyword_score
        else:
            # Other questions: maintain original weights
            final_reward = 0.7 * keyword_score + 0.3 * semantic_score
        return final_reward

    # Process input data format
    contents = [pred[0]["content"] for pred in predictions]
    rewards = []

    for content, question, solution in zip(contents, questions, ground_truths):
        reward = 0.0
        try:
            # Extract predicted answer
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            predicted_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Extract ground truth answer
            sol_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
            ground_truth_answer = sol_match.group(1).strip() if sol_match else solution.strip()
            
            reward = calculate_unified_reward(predicted_answer, ground_truth_answer, question)
        except Exception as e:
            print(f"Error in vase_action_reward: {e}")
            reward = 0.0
        rewards.append(reward)

    return rewards


def vase_action_rewards(completions, solution, **kwargs):
    """Vase task action reward function (batch processing) - includes task weighting mechanism"""
    # Extract question information from kwargs
    problems = kwargs.get('problem', [''] * len(completions))
    model = kwargs.get('model', None)
    
    # Remove model parameter to avoid duplicate passing
    kwargs_without_model = {k: v for k, v in kwargs.items() if k != 'model'}
    
    # Get base rewards
    base_rewards = vase_action_reward(completions, problems, solution, model, **kwargs_without_model)
    
    # Task weighting: increase weight for difficult questions
    weighted_rewards = []
    for reward, question in zip(base_rewards, problems):
        weight = 1.0  # Default weight
        
        question_lower = question.lower()
        if "date" in question_lower or "decoration" in question_lower:
            weight = 5.0  # Amplify weight for difficult questions by 5x
        elif "attribution" in question_lower:
            weight = 3.0  # Moderate weight for medium difficulty questions
        
        weighted_reward = reward * weight
        weighted_rewards.append(weighted_reward)
    
    return weighted_rewards


def vase_format_rewards(completions, solution, **kwargs):
    """Vase task format reward function (batch processing) - only check think/answer tag format"""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:

        think_answer_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        has_correct_format = bool(re.fullmatch(think_answer_pattern, content, re.DOTALL))
        
        reward = 1.0 if has_correct_format else 0.0
        rewards.append(reward)
    
    return rewards


def cosine_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        clean_content = clean_text(content)
        sol = clean_text(sol)
        if sol == "none":
            if clean_content == "none":
                acc_reward = 1.0
            else:
                acc_reward = 0.0
        else:
            acc_reward = detection_score(clean_content, sol)
        reward = cosine_reward(content, tokenizer, acc_reward)
        rewards.append(reward)

    return rewards


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None


def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)


def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')

    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()


def all_match_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return 1.0 if content == sol else 0.0


def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()

    # Try symbolic verification first for numeric answers
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)

            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mcq":
            reward = mcq_reward(content, sol)
        elif accu_reward_method == 'yes_no':
            reward = yes_no_reward(content, sol)
        elif accu_reward_method == 'llm':
            reward = llm_reward(content, sol)
        elif accu_reward_method == 'map':
            reward = map_reward(content, sol)
        elif accu_reward_method == 'math':
            reward = math_reward(content, sol)
        elif accu_reward_method == 'weighted_sum':
            clean_content = clean_text(content)
            sol = clean_text(sol)
            if sol == "none":
                if clean_content == "none":
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = detection_score(clean_content, sol)
        elif accu_reward_method == 'od_ap':
            reward = od_reward(content, sol)
        elif accu_reward_method == 'od_ap50':
            reward = od_reward(content, sol, score_type=1)
        elif accu_reward_method == 'odLength':
            reward = odLength_reward(content, sol)
        elif accu_reward_method == 'all_match':
            reward = all_match_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, solution, **kwargs):
    """Enhanced format reward that checks think/answer tags and action list format."""
    import json
    import re

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(completion_contents, solution):
        reward = 0.0


        think_answer_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        has_correct_format = bool(re.fullmatch(think_answer_pattern, content, re.DOTALL))
        if has_correct_format:
            reward += 0.5


        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            cleaned_content = clean_json_comments(answer_content)
            try:
                pred_actions = json.loads(cleaned_content)
            except json.JSONDecodeError:

                return reward


            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            gt_content = sol_match.group(1).strip() if sol_match else sol.strip()
            gt_cleaned_content = clean_json_comments(gt_content)
            try:
                gt_actions = json.loads(gt_cleaned_content)
            except json.JSONDecodeError:

                return reward


            if (isinstance(pred_actions, list) and
                    isinstance(gt_actions, list) and
                    len(pred_actions) == len(gt_actions)):
                reward += 0.5

        rewards.append(reward)

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": cosine_rewards,
    "repetition": repetition_rewards,
    "stardojo_action": stardojo_action_rewards,
    "geo_action": geo_action_rewards,
    "geo_format": geo_format_rewards,
    "vase_action": vase_action_rewards,
    "vase_format": vase_format_rewards,
}


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.task_type)

    # Get reward functions
    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in
                        script_args.reward_funcs]
    else:
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset

    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(
            data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            
            if isinstance(accu_reward_method, str) and (
                    accu_reward_method.startswith("geo") or accu_reward_method == "vase"):
                try:
                    loaded_data = json.load(f)
                    items_iter = loaded_data if isinstance(loaded_data, list) else [loaded_data]
                except json.JSONDecodeError:
                    f.seek(0)
                    items_iter = (json.loads(line) for line in f if line.strip())
            else:
                items_iter = (json.loads(line) for line in f if line.strip())

            for item in items_iter:

                if 'conversations' in item:

                    img_field_name = None
                    img_field_value = None

                    if 'image' in item:
                        img_field_name = 'image'
                        img_field_value = item['image']
                    elif 'images' in item:
                        img_field_name = 'images'
                        img_field_value = item['images']

                    if img_field_name and img_field_value:
                        if isinstance(img_field_value, str):
                            # Store image path instead of loading the image
                            item['image_path'] = [os.path.join(image_folder, img_field_value)]
                            del item[img_field_name]  # remove the image column so that it can be loaded later
                        elif isinstance(img_field_value, list):
                            # if the image is a list, then it is a list of images (for multi-image input)
                            item['image_path'] = [os.path.join(image_folder, image) for image in img_field_value]
                            del item[img_field_name]  # remove the image column so that it can be loaded later
                        else:
                            raise ValueError(f"Unsupported image type: {type(img_field_value)}")

                    # Remove immediate image loading
                    item['problem'] = item['conversations'][0]['value'].replace('<image>', '')

                    # Handle solution that could be a float or string
                    solution_value = item['conversations'][1]['value']
                    if isinstance(solution_value, str):
                        item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                    else:
                        # If it's a float or other non-string type, keep it as is
                        item['solution'] = str(solution_value)

                    del item['conversations']
                elif ('image' in item or 'images' in item) and 'problem' in item and 'solution' in item:

                    img_field = item.get('image') or item.get('images')
                    if isinstance(img_field, str):
                        item['image_path'] = [os.path.join(image_folder, img_field)]
                    else:
                        raise ValueError(f"Unsupported image type: {type(img_field)}")


                    item['problem'] = (item.get('problem') or '').replace('<image>', '')
                    item['solution'] = item['solution']


                    if 'image' in item:
                        del item['image']
                    if 'images' in item:
                        del item['images']
                elif 'starting_image_path' in item and 'ending_image_path' in item:
                    # Stardojo format: determine task type based on accu_reward_method
                    task_description = ', '.join(item['tasks']) if item['tasks'] else "No specific task"

                    if accu_reward_method == "stardojo1":
                        # stardojo1: use only initial state + task description
                        item['image_path'] = [os.path.join(image_folder, item['starting_image_path'])]
                        item['problem'] = f"Task: {task_description}"
                    elif accu_reward_method == "stardojo2":

                        item['image_path'] = [
                            os.path.join(image_folder, item['starting_image_path']),
                            os.path.join(image_folder, item['ending_image_path'])
                        ]
                        item['problem'] = "Analyze the state changes between the two images."
                    else:

                        item['image_path'] = [
                            os.path.join(image_folder, item['starting_image_path']),
                            os.path.join(image_folder, item['ending_image_path'])
                        ]
                        item['problem'] = f"Task: {task_description}"


                    item['solution'] = json.dumps(item['actions'])


                    del item['starting_image_path']
                    del item['ending_image_path']
                    del item['actions']
                    del item['tasks']
                else:
                    raise ValueError(f"Unsupported data format in item: {item}")
                item['accu_reward_method'] = item.get('accu_reward_method',
                                                       accu_reward_method)  # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            assert all(
                os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
            # Don't load image here, just store the path
            return {
                'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)