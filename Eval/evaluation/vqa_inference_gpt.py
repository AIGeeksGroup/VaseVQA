import argparse
import json
import os
import time
from tqdm import tqdm
import sys
from pathlib import Path
from openai import OpenAI
import base64

# ==========================================
# Prompt Alignment (from vqa_prompt.py)
# ==========================================
def get_task_specific_prompt(question: str) -> str:
    """
    Generate task-specific format prompts based on question type to avoid data leakage
    Aligned with VaseVL/evaluation/vqa_prompt.py
    """
    question_lower = question.lower()
    
    # Concise base prompt
    base_prompt = "Analyze the vase and answer in the exact format:\n"
    
    if "fabric" in question_lower:
        return base_prompt + "The fabric of the vase is [FABRIC_NAME]."
    
    elif "technique" in question_lower:
        return base_prompt + "The technique of the vase is [TECHNIQUE_NAME]."
    
    elif "shape name" in question_lower or "shape" in question_lower:
        return base_prompt + "The shape name of the vase is [SHAPE_NAME]."
    
    elif "provenance" in question_lower:
        return base_prompt + "The provenance of the vase is [LOCATION]."
    
    elif "date" in question_lower:
        return base_prompt + "The date of the vase is [START_YEAR] to [END_YEAR]. Use BCE as negative years."
    
    elif "attribution" in question_lower:
        return base_prompt + "The attribution of the vase is [ATTRIBUTION_INFO]."
    
    elif "decoration" in question_lower:
        return base_prompt + "The decoration of the vase is [DESCRIPTION]. Use format A: [SIDE_A] | B: [SIDE_B] for both sides."
    
    else:
        return base_prompt + "Answer based on your analysis."

# ==========================================
# Data Loading
# ==========================================
def read_vasevl_data(file_path: str, image_dir: str):
    """
    Read data from VaseVL dataset file and convert to inference format
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Expected list format in VaseVL dataset")
        
        items = []
        for i, item in enumerate(data):
            if 'id' not in item:
                question_id = i
            else:
                question_id = item['id']

            image_field = item.get('images') or item.get('image', '')
            if not image_field:
                continue
            
            image_filename = os.path.basename(image_field)
            absolute_image_path = os.path.join(image_dir, image_filename)
            
            question = item['conversations'][0]['value'].replace('<image>', '').strip()
            
            # Get task-specific aligned prompt
            task_prompt = get_task_specific_prompt(question)
            full_prompt = task_prompt + "\n" + question
            
            items.append({
                "question_id": question_id,
                "image_path": absolute_image_path,
                "image_filename": image_filename,
                "prompt": full_prompt,
                "raw_question": question
            })
        
        return items
            
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return []
    except Exception as e:
        print(f"Unknown error: {str(e)}")
        return []

# ==========================================
# Image Encoding
# ==========================================
def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 string (required by OpenAI API)
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# Inference Logic
# ==========================================
def run_inference(client, model_name, item):
    try:
        image_path = item['image_path']
        prompt = item['prompt']
        
        if not os.path.exists(image_path):
            return None, f"Error: Image not found at {image_path}"

        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Determine image format
        image_ext = os.path.splitext(image_path)[1].lower()
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(image_ext, 'image/jpeg')
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,  # Deterministic
            max_tokens=128
        )
        
        return None, response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {item['question_id']}: {error_msg}")
        
        # Check for fatal errors
        if "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            return "FATAL_QUOTA", f"Error: Insufficient Quota - {error_msg}"
        elif "rate_limit" in error_msg.lower() or "429" in error_msg:
            time.sleep(10)
            return "RATE_LIMIT", f"Error: Rate Limited - {error_msg}"
        elif "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "FATAL_AUTH", f"Error: Invalid API Key - {error_msg}"
        
        return "ERROR", f"Error: {error_msg}"

def main():
    # Get project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    # Default path configuration
    DEFAULT_QUESTION_FILE = os.path.join(PROJECT_ROOT, "Data/data_test_single_llava_vasevl_v9.json")
    DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "Data/images")
    DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output/gpt_results.jsonl")
    
    parser = argparse.ArgumentParser(description="VaseVL Inference using GPT-4o-mini")
    parser.add_argument("--question-file", type=str, default=DEFAULT_QUESTION_FILE, help="Path to dataset file")
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="Path to save inference results")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (optional)")
    
    args = parser.parse_args()

    # Setup API Key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide API Key via --api-key or OPENAI_API_KEY env var.")
        return

    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    
    client = OpenAI(**client_kwargs)

    # Load Data
    print(f"Loading data from {args.question_file}...")
    items = read_vasevl_data(args.question_file, args.image_dir)
    
    if args.limit:
        items = items[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    print(f"Found {len(items)} samples.")

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Load existing results to resume from interruption
    processed_ids = set()
    if os.path.exists(args.output_file):
        print(f"\n[INFO] Found existing output file. Loading processed samples...")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    processed_ids.add(record['question_id'])
                except:
                    pass
        print(f"[INFO] Already processed: {len(processed_ids)} samples. Will skip them.")
    
    # Open file in append mode to preserve existing results
    mode = 'a' if processed_ids else 'w'
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    try:
        with open(args.output_file, mode, encoding='utf-8') as f:
            for item in tqdm(items, desc="Processing"):
                # Skip if already processed
                if item['question_id'] in processed_ids:
                    continue
                
                error_type, answer = run_inference(client, args.model_name, item)
                
                # Handle fatal errors
                if error_type == "FATAL_QUOTA":
                    print(f"\n[FATAL] Insufficient quota detected. Stopping...")
                    print(f"[INFO] Progress saved. Processed {len(processed_ids)} samples.")
                    break
                elif error_type == "FATAL_AUTH":
                    print(f"\n[FATAL] Authentication error. Please check your API key.")
                    break
                elif error_type in ["ERROR", "RATE_LIMIT"]:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\n[WARNING] Too many consecutive errors ({consecutive_errors}). Stopping...")
                        print(f"[INFO] Progress saved. Processed {len(processed_ids)} samples.")
                        break
                else:
                    consecutive_errors = 0  # Reset on success
                
                result_record = {
                    "question_id": item['question_id'],
                    "image": item['image_filename'],
                    "instruction": item['raw_question'],
                    "output": answer,
                    "type": "qa"
                }
                
                f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                f.flush()  # Ensure written to disk immediately
                processed_ids.add(item['question_id'])
                
                # Basic rate limiting
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user. Progress saved.")
        print(f"[INFO] Processed {len(processed_ids)} samples so far.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print(f"[INFO] Progress saved. Processed {len(processed_ids)} samples.")

    print(f"\nFinished. Results saved to {args.output_file}")
    print(f"Total processed: {len(processed_ids)} samples")

if __name__ == "__main__":
    main()

