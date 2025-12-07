import argparse
import json
import os
import time
from tqdm import tqdm
import sys
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

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
# Inference Logic
# ==========================================
def run_inference(model, item):
    try:
        image_path = item['image_path']
        prompt = item['prompt']
        
        print(f"\n[DEBUG] Processing question_id: {item['question_id']}")
        print(f"[DEBUG] Image path: {image_path}")
        print(f"[DEBUG] Image exists: {os.path.exists(image_path)}")
        
        if not os.path.exists(image_path):
            return None, f"Error: Image not found at {image_path}"

        # Load image using PIL
        print(f"[DEBUG] Loading image...")
        img = Image.open(image_path)
        print(f"[DEBUG] Image loaded successfully. Size: {img.size}, Mode: {img.mode}")
        
        print(f"[DEBUG] Prompt (first 200 chars): {prompt[:200]}...")
        print(f"[DEBUG] Calling Gemini API...")
        
        # Generate content
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, # Deterministic
                max_output_tokens=128
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        print(f"[DEBUG] API call completed")
        print(f"[DEBUG] Response text: {response.text[:200] if response.text else 'None'}...")
        
        return None, response.text
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] Exception occurred while processing {item['question_id']}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Exception message: {error_msg}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        
        # Check for fatal errors
        if "quota" in error_msg.lower() or "billing" in error_msg.lower():
            return "FATAL_QUOTA", f"Error: Insufficient Quota - {error_msg}"
        elif "429" in error_msg or "resource_exhausted" in error_msg.lower():
            time.sleep(10)
            return "RATE_LIMIT", f"Error: Rate Limited - {error_msg}"
        elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "FATAL_AUTH", f"Error: Invalid API Key - {error_msg}"
        
        return "ERROR", f"Error: {error_msg}"

def main():
    # Get project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    # Default path configuration
    DEFAULT_QUESTION_FILE = os.path.join(PROJECT_ROOT, "Data/data_test_single_llava_vasevl_v9.json")
    DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "Data/images")
    DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output/gemini_results.jsonl")
    
    parser = argparse.ArgumentParser(description="VaseVL Inference using Gemini 2.0 Flash")
    parser.add_argument("--question-file", type=str, default=DEFAULT_QUESTION_FILE, help="Path to dataset file")
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="Path to save inference results")
    parser.add_argument("--api-key", type=str, default=None, help="Google API Key (or set GCP_API_KEY/GOOGLE_API_KEY env var)")
    parser.add_argument("--model-name", type=str, default="gemini-2.0-flash-exp", help="Gemini model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    
    args = parser.parse_args()

    # Setup API Key
    api_key = args.api_key or os.getenv("GCP_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Please provide API Key via --api-key or GCP_API_KEY/GOOGLE_API_KEY env var.")
        return

    print(f"\n[DEBUG] Configuring Gemini API...")
    print(f"[DEBUG] API Key (first 10 chars): {api_key[:10]}...")
    genai.configure(api_key=api_key)
    
    # Init Model
    print(f"[DEBUG] Initializing model: {args.model_name}")
    model = genai.GenerativeModel(args.model_name)
    print(f"[DEBUG] Model initialized successfully")

    # Load Data
    print(f"\n[DEBUG] Loading data from {args.question_file}...")
    items = read_vasevl_data(args.question_file, args.image_dir)
    
    if args.limit:
        items = items[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    print(f"Found {len(items)} samples.")
    
    if len(items) > 0:
        print(f"\n[DEBUG] First item preview:")
        print(f"  - question_id: {items[0]['question_id']}")
        print(f"  - image_filename: {items[0]['image_filename']}")
        print(f"  - image_path: {items[0]['image_path']}")
        print(f"  - raw_question: {items[0]['raw_question'][:100]}...")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    print(f"\n[DEBUG] Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

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
    
    print(f"[DEBUG] Opening output file: {args.output_file} (mode: {mode})")
    
    try:
        with open(args.output_file, mode, encoding='utf-8') as f:
            for idx, item in enumerate(tqdm(items, desc="Processing")):
                # Skip if already processed
                if item['question_id'] in processed_ids:
                    print(f"[DEBUG] Skipping already processed question_id: {item['question_id']}")
                    continue
                
                print(f"\n{'='*60}")
                print(f"[DEBUG] Starting inference for item {idx+1}/{len(items)}")
                
                error_type, answer = run_inference(model, item)
                
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
                
                print(f"[DEBUG] Inference completed. Answer length: {len(answer) if answer else 0}")
                
                result_record = {
                    "question_id": item['question_id'],
                    "image": item['image_filename'],
                    "instruction": item['raw_question'],
                    "output": answer,
                    "type": "qa"
                }
                
                print(f"[DEBUG] Writing result to file...")
                f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                f.flush() # Ensure written to disk immediately
                print(f"[DEBUG] Result written successfully")
                processed_ids.add(item['question_id'])
                
                # Basic rate limiting
                print(f"[DEBUG] Sleeping for 1 second (rate limiting)...")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user. Progress saved.")
        print(f"[INFO] Processed {len(processed_ids)} samples so far.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print(f"[INFO] Progress saved. Processed {len(processed_ids)} samples.")

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Finished. Results saved to {args.output_file}")
    print(f"Total processed: {len(processed_ids)} samples")

if __name__ == "__main__":
    main()

