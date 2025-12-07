from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
import json
import os
from tqdm import tqdm
import sys
import platform
import torch
# from qwen_vl_utils import process_vision_info # Assuming qwen_vl_utils is in the path
from pathlib import Path
import argparse
from transformers import LlavaNextProcessor
import time

# --- ADDED: Helper function for unique filenames ---
def generate_unique_filename(directory, basename, extension=".jsonl"):
    """
    Generates a unique filename in a given directory to avoid overwriting.
    Example: if infer_sft.jsonl exists, it will return infer_sft_1.jsonl.
    """
    os.makedirs(directory, exist_ok=True)
    base_path = os.path.join(directory, f"{basename}{extension}")
    if not os.path.exists(base_path):
        return base_path
    
    counter = 1
    while True:
        unique_path = os.path.join(directory, f"{basename}_{counter}{extension}")
        if not os.path.exists(unique_path):
            return unique_path
        counter += 1
# --- MODIFICATION END ---


# Assuming qwen_vl_utils.py is in a discoverable path
# If not, you might need to add its location to sys.path
# For example: sys.path.append('/path/to/folder/containing/qwen_vl_utils')
from qwen_vl_utils import process_vision_info


if platform.system() == 'Darwin' and torch.backends.mps.is_available():
    current_file_abs = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_abs)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.append(parent_dir)

    from device import get_device
    device_map, device = get_device()
else:
    if torch.cuda.is_available():
        device_map, device = "auto", "cuda"
    else:
        device_map, device = "auto", "cpu"


def get_task_specific_prompt(question: str) -> str:
    """
    Generate task-specific format prompts based on question type to avoid data leakage
    """
    question_lower = question.lower()
    
    # Concise base prompt
    base_prompt = "Analyze the vase and answer in the exact format:\n"
    
    return base_prompt 


def read_vasevl_data(file_path: str, image_dir: str):
    """
    Read data from VaseVL dataset file and convert to inference format (modified with ID checking and path concatenation)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Expected list format in VaseVL dataset")
        
        messages = []
        for i, item in enumerate(data):
            if 'id' not in item:
                print(f"!!! WARNING: Data at line {i+1} is missing 'id' field. Using line number {len(messages)} as fallback ID.")
                print(f"!!! Problematic data: {item}")
                question_id = len(messages)
            else:
                question_id = item['id']

            image_field = item.get('images') or item.get('image', '')
            if not image_field:
                continue
            
            image_filename = os.path.basename(image_field)
            absolute_image_path = os.path.join(image_dir, image_filename)
            
            question = item['conversations'][0]['value'].replace('<image>', '').strip()
            
            # Get task-specific format prompt
            task_prompt = get_task_specific_prompt(question)
            
            # Combine full prompt - more concise format
            full_prompt = task_prompt + "\n" + question
            
            message = [
                {
                    "role": "user",
                    "content": [
                        { "type": "image", "image": absolute_image_path, "question_id": question_id },
                        { "type": "text", "text": full_prompt }
                    ]
                }
            ]
            messages.append(message)
        
        return messages
            
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return []
    except Exception as e:
        print(f"Unknown error: {str(e)}")
        return []

def prepare_message(question_file, image_dir):
    data_list = read_vasevl_data(question_file, image_dir)
    print("Eval ", len(data_list), " samples...")
    return data_list

def batch_process(images, texts, device, model, processor, batch_size=2, model_path=""):
    all_outputs = []
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
        # This conditional logic for processor can be simplified if not needed
        # For now, keeping it as is.
        if ("llava-1.5-7b-hf" in model_path):
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt").to(device)
        else:
            inputs = processor(text=batch_texts, images=batch_images, padding=True, return_tensors="pt").to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(output_texts)
    return all_outputs

def convert_to_jsonl(questions, answers, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, (q_item, answer) in enumerate(zip(questions, answers)):
            img_path = q_item[0]['content'][0]['image']
            img_file = Path(img_path).name
            
            question_text = q_item[0]['content'][1]['text']
            question_id = q_item[0]['content'][0]['question_id']
            
            record = {
                "question_id": question_id,
                "image": img_file,
                "instruction": question_text,
                "output": answer,
                "type": "qa"
            }
            f.write(json.dumps(record) + '\n')


if __name__=="__main__":

    # Get project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    # Default path configuration
    DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Models/VaseVL/checkpoint-1690")
    DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "Data/images")
    DEFAULT_QUESTION_FILE = os.path.join(PROJECT_ROOT, "Data/data_test_single_llava_vasevl_v9.json")
    DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    infer_parser = argparse.ArgumentParser(description="Evaluation Script with Task-Specific Prompts")
    
    # Model path parameter
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model (default: Models/VaseVL/checkpoint-1690)"
    )
    
    infer_parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    infer_parser.add_argument("--question-file", type=str, default=DEFAULT_QUESTION_FILE, help="Path to dataset file")
    
    infer_parser.add_argument("--output-file", type=str, default=None, help="[Optional] Path to save inference results. If not provided, a name will be auto-generated.")
    infer_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save auto-generated output files.")

    infer_parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    
    args = infer_parser.parse_args()

    # Get model path
    model_path = args.model_path
    
    # Extract model name from model path for output file naming
    model_name = os.path.basename(model_path.rstrip('/'))
    
    if args.output_file:
        output_file = args.output_file
    else:
        basename = f"infer_{model_name}"
        output_file = generate_unique_filename(args.output_dir, basename)
    
    print(f"--- Running Evaluation ---")
    print(f"Model Path: {model_path}")
    print(f"Output File: {output_file}")
    print(f"--------------------------")
    # --- MODIFICATION END ---

    os.makedirs(os.path.dirname(output_file), exist_ok=True) 

    # --- MODIFICATION: Using the resolved model_path variable ---
    if ("Qwen2.5" in model_path) or ("qwen2_5" in model_path):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=device_map)
    elif (model_path.startswith("Qwen/Qwen2-")):
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # ... (other elif blocks for different models would go here)
    else:
        # A general fallback, assuming Qwen2.5-VL architecture if not specified. Adjust if needed.
        print(f"Warning: Model type for '{model_path}' not explicitly matched. Assuming a Qwen2.5-VL compatible architecture.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=device_map)

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, padding_side='left')
    # --- MODIFICATION END ---
    
    messages = prepare_message(args.question_file, args.image_dir)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    
    image_inputs, _ = process_vision_info(messages) # Assuming no videos
    
    print("Eval batch size: ", args.batch_size)
    pred_answer = batch_process(image_inputs, texts, device, model, processor, batch_size=args.batch_size, model_path=model_path)
    
    print("SAVING JSON FILE...")
    convert_to_jsonl(messages, pred_answer, output_file)
    print(f"Json file saved to {output_file}. FINISHED.")