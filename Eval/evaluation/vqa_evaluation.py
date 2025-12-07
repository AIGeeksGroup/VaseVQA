import os
import json
from accuracy_valuator import TextCapsBleu4Evaluator, DateAccuracyEvaluator, STVQAANLSEvaluator
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import remove_tags


def eval_single(annotation_file, infer_file, result_file):

    Q1 = "<image>\n What is the fabric of the vase?"
    Q2 = "<image>\n What is the technique of the vase?"
    Q3 = "<image>\n What is the shape name of the vase?"
    Q4 = "<image>\n What is the provenance of the vase?"
    Q5 = "<image>\n What is the date of the vase?"
    Q6 = "<image>\n What is the attribution of the vase?"
    Q7 = "<image>\n What is the decoration of the vase?"
    # Q8 = "<image>\n What is the overall of the vase?"

    experiment_name = os.path.splitext(os.path.basename(infer_file))[0]
    print(experiment_name)
    
    # Read ground truth directly from VaseVL dataset
    vasevl_data = json.load(open(annotation_file))
    
    # Build annotation dictionary: use array index as key since inference results are generated in order
    annotations = {}
    for idx, item in enumerate(vasevl_data):
        # Keep complete question format including <image> tag to match with evaluator
        instruction = item['conversations'][0]['value'].strip()
        ground_truth = item['conversations'][1]['value'].strip()
        annotations[idx] = {
            'id': item['id'],
            'instruction': instruction,
            'output': ground_truth
        }
    
    results = [json.loads(line) for line in open(infer_file)]

    pred_list = []
    for idx, result in enumerate(results):
        try:
            # Use result index to match annotations
            if idx in annotations:
                annotation = annotations[idx]
                pred_list.append({
                    "pred_answer": result['output'],
                    "gt_answers": annotation['output'],
                    "question": annotation['instruction'],
                })
            else:
                print(f"Warning: No ground truth found for result index {idx}")
        except Exception as e:
            print(f"Error processing result {idx}: {e}")
            continue

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

    Q1 = "What is the fabric of the vase?"
    Q2 = "What is the technique of the vase?"
    Q3 = "What is the shape name of the vase?"
    Q4 = "What is the provenance of the vase?"
    Q5 = "What is the date of the vase?"
    Q6 = "What is the attribution of the vase?"
    Q7 = "What is the decoration of the vase?"

    results = [
        ("Q1", Q1, "Accuracy", acc_list[0]),
        ("Q2", Q2, "Accuracy", acc_list[1]),
        ("Q3", Q3, "Accuracy", acc_list[2]),
        ("Q4", Q4, "Accuracy", acc_list[3]),
        ("Q5", Q5, "Accuracy", acc_Q5),
        ("Q6", Q6, "Accuracy", acc_list[4]),
        ("Q7", Q7, "Bleu@1", bleu1_Q7),
    ]

    # Define table format
    header = f"{'Question':<50}\t{'Metric':<10}\t{'Value':>8}"
    separator = "-"*100

    # Generate table content
    table_content = []
    for qid, question, metric, value in results:
        formatted_line = f"{qid}:{question:<45}\t{metric + ':':<10}\t{value*100:>7.2f}%"
        table_content.append(formatted_line)

    # Console output
    print(infer_file + "\n")
    print("\nVASE ANALYSIS RESULTS:")
    print(header)
    print(separator)
    print("\n".join(table_content))

    # File output
    with open(result_file, "w") as f:
        f.write(infer_file + "\n")
        f.write("VASE ANALYSIS RESULTS\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        f.write("\n".join(table_content))


if __name__ == "__main__":
    """Configure argument parser for inference parameters"""
    infer_parser = argparse.ArgumentParser(description="Evaluation Script")
    
    # Get project root directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    # Default paths
    DEFAULT_ANNOTATION = os.path.join(PROJECT_ROOT, "Data/data_test_single_llava_vasevl_v9.json")
    DEFAULT_INFER = os.path.join(SCRIPT_DIR, "output/gemini_results.jsonl")
    
    # Required parameters
    infer_parser.add_argument(
        "--annotation-file",
        type=str,
        default=DEFAULT_ANNOTATION,
        help="Path to VaseVL test dataset file (default: %(default)s)"
    )
    
    # Data configuration
    infer_parser.add_argument(
        "--infer-file",
        type=str,
        default=DEFAULT_INFER,
        help="Path to inference file (default: %(default)s)"
    )

    # Parse arguments
    args = infer_parser.parse_args()
    print(args)

    # Auto-generate result_file path based on infer-file
    # Get directory of infer_file
    result_dir = os.path.dirname(args.infer_file)
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(args.infer_file))[0]
    # Replace "_inference_answers" with "_evaluation" in filename
    result_base_name = base_name.replace('_inference_answers', '_evaluation')
    # Concatenate to create new result_file path with .txt extension
    result_file = os.path.join(result_dir, result_base_name + '.txt')

    print(f"✅ Result file will be saved to: {result_file}")

    # Create directory and call evaluation function
    os.makedirs(os.path.dirname(result_file), exist_ok=True)                 

    eval_single(args.annotation_file, args.infer_file, result_file)

    print("FINISHED.........")