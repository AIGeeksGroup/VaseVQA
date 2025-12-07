import os
import json
import re
import sys
import argparse
import torch
from accuracy_valuator import TextCapsBleu4Evaluator, DateAccuracyEvaluator, STVQAANLSEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add current directory to path for importing other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. AlignScore Evaluator
# ==========================================
class AlignScoreEvaluator:
    def __init__(self, model_path, device="cuda"):
        print(f"[AlignScore] Loading model from: {model_path}")
        try:
            from alignscore import AlignScore
        except ImportError:
            print("Warning: AlignScore not installed or failed to import. AlignScore metrics will be 0.")
            self.scorer = None
            return
            
        # AlignScore usually loads a checkpoint like 'AlignScore-large.ckpt'
        try:
            self.scorer = AlignScore(model='roberta-large', batch_size=32, device=device, ckpt_path=model_path, evaluation_mode='nli_sp')
            print(f"[AlignScore] Model loaded successfully.")
        except Exception as e:
            print(f"[AlignScore] Error loading model: {e}")
            self.scorer = None

    def eval_pred_list(self, pred_list, target_question):
        """
        Evaluate predictions in the list
        :param pred_list: List containing 'question', 'pred_answer', 'gt_answers'
        :param target_question: Target question string
        :return: Average AlignScore
        """
        if self.scorer is None:
            return 0.0

        relevant_preds = []
        relevant_gts = []
        
        for entry in pred_list:
            if entry["question"] == target_question:
                relevant_preds.append(entry["pred_answer"])
                relevant_gts.append(entry["gt_answers"]) # gt_answers is a string in this context based on load logic

        if not relevant_preds:
            return 0.0

        # AlignScore.score(contexts, claims) -> contexts=Reference, claims=Prediction
        try:
            scores = self.scorer.score(contexts=relevant_gts, claims=relevant_preds)
            return sum(scores) / len(scores)
        except Exception as e:
            print(f"[AlignScore] Evaluation error: {e}")
            return 0.0

# ==========================================
# 2. Prometheus Evaluator (1-5 Score)
# ==========================================
class PrometheusEvaluator:
    def __init__(self, model_path, device="cuda"):
        print(f"[Prometheus] Loading model from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            self.device = device
            print(f"[Prometheus] Model loaded successfully.")
        except Exception as e:
            print(f"[Prometheus] Error loading model: {e}")
            self.model = None

        self.ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        
        self.ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

        self.RUBRIC = """[Is the response comprehensive and accurate?]
Score 1: The response is completely incorrect, irrelevant, or hallucinates content not present in the image/description.
Score 2: The response misses major details or contains significant errors, though it touches on the subject.
Score 3: The response captures the main subject but misses some important details or has minor inaccuracies.
Score 4: The response is mostly accurate and complete, with only very minor omissions or slight phrasing issues.
Score 5: The response is comprehensive, accurate, and perfectly matches the level of detail in the reference answer."""

    def _parse_score(self, decoded_text):
        try:
            if "[RESULT]" in decoded_text:
                score_part = decoded_text.split("[RESULT]")[-1].strip()
                score = int(re.search(r'\d+', score_part).group())
            else:
                # Fallback: look for last number
                matches = re.findall(r'\b[1-5]\b', decoded_text)
                score = int(matches[-1]) if matches else 1
                
            # Clamp between 1 and 5
            score = max(1, min(5, score))
        except:
            score = 1
        return score

    def evaluate_batch(self, batch_data):
        if self.model is None:
            return [0] * len(batch_data)

        prompts = []
        for entry in batch_data:
            user_content = self.ABS_SYSTEM_PROMPT + "\n\n" + self.ABSOLUTE_PROMPT.format(
                instruction=entry["question"],
                response=entry["pred_answer"],
                reference_answer=entry["gt_answers"],
                rubric=self.RUBRIC
            )
            prompts.append(user_content)
        
        # Format as chat messages
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        
        # Apply chat template to get text prompts
        text_prompts = [self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_list]
        
        # Batch tokenize
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            decoded_batch = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
        scores = [self._parse_score(decoded) for decoded in decoded_batch]
        return scores

    def eval_pred_list(self, pred_list, target_question, batch_size=8):
        """
        Evaluate predictions in the list (Batch version)
        :return: Average normalized score (0.0 - 1.0)
        """
        if self.model is None:
            return 0.0

        target_entries = [entry for entry in pred_list if entry["question"] == target_question]
        
        if not target_entries:
            return 0.0
            
        print(f"Evaluating {len(target_entries)} samples with Prometheus (Batch Size: {batch_size})...")
        
        total_score = 0
        count = 0
        
        # Batch processing
        for i in tqdm(range(0, len(target_entries), batch_size)):
            batch = target_entries[i : i + batch_size]
            scores = self.evaluate_batch(batch)
            total_score += sum(scores)
            count += len(scores)
            
        if count == 0:
            return 0.0
            
        avg_score = total_score / count
        # Normalize 1-5 to 0.0-1.0
        # (Score - 1) / (5 - 1) -> 1=0.0, 5=1.0
        normalized_score = (avg_score - 1) / 4.0
        return normalized_score


def eval_single(annotation_file, infer_file, result_file, alignscore_path=None, prometheus_path=None):

    Q1 = "<image>\n What is the fabric of the vase?"
    Q2 = "<image>\n What is the technique of the vase?"
    Q3 = "<image>\n What is the shape name of the vase?"
    Q4 = "<image>\n What is the provenance of the vase?"
    Q5 = "<image>\n What is the date of the vase?"
    Q6 = "<image>\n What is the attribution of the vase?"
    Q7 = "<image>\n What is the decoration of the vase?"

    experiment_name = os.path.splitext(os.path.basename(infer_file))[0]
    print(f"Processing: {experiment_name}")
    
    # Read Ground Truth
    try:
        vasevl_data = json.load(open(annotation_file))
    except FileNotFoundError:
        print(f"Error: Annotation file not found: {annotation_file}")
        return

    annotations = {}
    for idx, item in enumerate(vasevl_data):
        instruction = item['conversations'][0]['value'].strip()
        ground_truth = item['conversations'][1]['value'].strip()
        annotations[idx] = {
            'id': item['id'],
            'instruction': instruction,
            'output': ground_truth
        }
    
    try:
        results_data = [json.loads(line) for line in open(infer_file)]
    except FileNotFoundError:
        print(f"Error: Inference file not found: {infer_file}")
        return

    pred_list = []
    for idx, result in enumerate(results_data):
        try:
            if idx in annotations:
                annotation = annotations[idx]
                pred_list.append({
                    "pred_answer": result['output'],
                    "gt_answers": annotation['output'],
                    "question": annotation['instruction'],
                })
            else:
                # Try finding by ID if index mismatch (implement if needed)
                print(f"Warning: No ground truth found for result index {idx}")
        except Exception as e:
            print(f"Error processing result {idx}: {e}")
            continue

    # Standard accuracy metrics
    acc_list = [0.0 for i in range(0, 5)]

    evaluator = STVQAANLSEvaluator()
    evaluator.set_q(Q1)
    acc_list[0] = evaluator.eval_pred_list(pred_list)

    evaluator.set_q(Q2)
    acc_list[1] = evaluator.eval_pred_list(pred_list)
    evaluator.set_q(Q3)
    acc_list[2] = evaluator.eval_pred_list(pred_list)
    evaluator.set_q(Q4)
    acc_list[3] = evaluator.eval_pred_list(pred_list)
    evaluator.set_q(Q6)
    acc_list[4] = evaluator.eval_pred_list(pred_list)

    evaluator_date = DateAccuracyEvaluator()
    evaluator_date.set_q(Q5)
    acc_Q5 = evaluator_date.eval_pred_list(pred_list)

    evaluator_bleu = TextCapsBleu4Evaluator()
    evaluator_bleu.set_q(Q7)
    bleu1_Q7, score = evaluator_bleu.eval_pred_list(pred_list)
    print(f"Bleu@1 (Q7): {bleu1_Q7}")

    # LLM Judge metrics for Q7 (Decoration)
    alignscore_Q7 = 0.0
    prometheus_Q7 = 0.0
    
    # 1. AlignScore for Q7
    if alignscore_path:
        print("\nCalculating AlignScore for Q7...")
        evaluator_align = AlignScoreEvaluator(alignscore_path)
        alignscore_Q7 = evaluator_align.eval_pred_list(pred_list, Q7)
        print(f"AlignScore (Q7): {alignscore_Q7}")
    
    # 2. Prometheus for Q7
    if prometheus_path:
        print("\nCalculating Prometheus Score for Q7...")
        evaluator_prom = PrometheusEvaluator(prometheus_path)
        prometheus_Q7 = evaluator_prom.eval_pred_list(pred_list, Q7)
        print(f"Prometheus Normalized (Q7): {prometheus_Q7}")

    # Generate report
    # Shortened question strings for table display
    Q1_s = "What is the fabric of the vase?"
    Q2_s = "What is the technique of the vase?"
    Q3_s = "What is the shape name of the vase?"
    Q4_s = "What is the provenance of the vase?"
    Q5_s = "What is the date of the vase?"
    Q6_s = "What is the attribution of the vase?"
    Q7_s = "What is the decoration of the vase?"

    results = [
        ("Q1", Q1_s, "Accuracy", acc_list[0]),
        ("Q2", Q2_s, "Accuracy", acc_list[1]),
        ("Q3", Q3_s, "Accuracy", acc_list[2]),
        ("Q4", Q4_s, "Accuracy", acc_list[3]),
        ("Q5", Q5_s, "Accuracy", acc_Q5),
        ("Q6", Q6_s, "Accuracy", acc_list[4]),
        ("Q7", Q7_s, "Bleu@1", bleu1_Q7),
    ]
    
    if alignscore_path:
        results.append(("Q7", Q7_s, "AlignScore", alignscore_Q7))
    if prometheus_path:
        results.append(("Q7", Q7_s, "Prometheus", prometheus_Q7))

    # Define table format
    header = f"{'Question':<50}\t{'Metric':<15}\t{'Value':>8}"
    separator = "-"*100

    # Generate table content
    table_content = []
    for qid, question, metric, value in results:
        formatted_line = f"{qid}:{question:<45}\t{metric + ':':<15}\t{value*100:>7.2f}%"
        table_content.append(formatted_line)

    # Console output
    print("\n" + infer_file + "\n")
    print("\nVASE ANALYSIS RESULTS (FULL):")
    print(header)
    print(separator)
    print("\n".join(table_content))

    # File output
    with open(result_file, "w") as f:
        f.write(infer_file + "\n")
        f.write("VASE ANALYSIS RESULTS (FULL)\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        f.write("\n".join(table_content))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Evaluation Script (Standard + LLM Judge)")
    
    # Get project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    # Default paths
    DEFAULT_ANNOTATION = os.path.join(PROJECT_ROOT, "Data/data_test_single_llava_vasevl_v9.json")
    DEFAULT_INFER = os.path.join(SCRIPT_DIR, "output/infer_original_1.jsonl")
    DEFAULT_ALIGNSCORE = os.path.join(PROJECT_ROOT, "Models/AlignScore-large.ckpt")
    DEFAULT_PROMETHEUS = os.path.join(PROJECT_ROOT, "Models/prometheus-7b-v2.0")
    
    parser.add_argument("--annotation-file", type=str, default=DEFAULT_ANNOTATION, help="Path to Ground Truth JSON")
    parser.add_argument("--infer-file", type=str, default=DEFAULT_INFER, help="Path to Prediction JSONL")
    
    parser.add_argument(
        "--alignscore-path", 
        type=str, 
        default=DEFAULT_ALIGNSCORE,
        help="Path to AlignScore Checkpoint (e.g., Models/AlignScore-large.ckpt)"
    )
    parser.add_argument(
        "--prometheus-path", 
        type=str, 
        default=DEFAULT_PROMETHEUS,
        help="Path to Prometheus Model (e.g., Models/prometheus-7b-v2.0)"
    )

    args = parser.parse_args()
    
    # Auto-generate result filename
    result_dir = os.path.dirname(args.infer_file)
    base_name = os.path.splitext(os.path.basename(args.infer_file))[0]
    result_base_name = base_name.replace('_inference_answers', '_evaluation_full')
    if result_base_name == base_name:  # if replacement didn't happen
        result_base_name = base_name + "_eval_full"
    result_file = os.path.join(result_dir, result_base_name + '.txt')

    print(f"✅ Result file will be saved to: {result_file}")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)                 

    eval_single(
        args.annotation_file, 
        args.infer_file, 
        result_file,
        alignscore_path=args.alignscore_path,
        prometheus_path=args.prometheus_path
    )

    print("FINISHED.........")