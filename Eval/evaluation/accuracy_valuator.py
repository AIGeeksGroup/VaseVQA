from tqdm import tqdm
import re
from difflib import SequenceMatcher
from collections import defaultdict
import re
from dateutil.parser import parse
import editdistance


class TextCapsBleu4Evaluator:
    def __init__(self):
        # The following script requires Java 1.8.0 and pycocotools installed.
        # The pycocoevalcap can be installed with pip as
        # pip install git+https://github.com/ronghanghu/coco-caption.git@python23
        # Original pycocoevalcap code is at https://github.com/tylin/coco-caption
        # but has no python3 support yet.
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        except ModuleNotFoundError:
            print(
                "Please install pycocoevalcap module using "
                "pip install git+https://github.com/ronghanghu/coco-caption.git@python23"  # noqa
            )
            raise

        self.tokenizer = PTBTokenizer()
        self.scorer = Bleu(4)
        # self.scorer = Bleu(1)

    def set_q(self, Q):
        self.Q = Q

    def eval_pred_list(self, pred_list):
        # Create reference and hypotheses captions.
        gts = {}
        res = {}
        for idx, entry in enumerate(pred_list):
            if (entry["question"] == self.Q):
                gts[idx] = [{"caption": a} for a in entry["gt_answers"]]
                res[idx] = [{"caption": entry["pred_answer"]}]
        
        # you need to install Java
        gts = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)
        score, _ = self.scorer.compute_score(gts, res)

        bleu1 = score[0]  # score is (Bleu-1, Bleu-2, Bleu-3, Bleu-4)
        # print("score:", score)
        # print("bleu:", bleu1)
        return bleu1, score


class STVQAANLSEvaluator:
    def __init__(self, threshold=0.5):
        import editdistance  # install with `pip install editdistance`

        self.get_edit_distance = editdistance.eval
        self.threshold = threshold  # Threshold for ANLS similarity

    def set_q(self, Q):
        self.Q = Q
        
    # def get_anls(self, s1, s2):
    #     s1 = s1.lower().strip()
    #     s2 = s2.lower().strip()
    #     iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
    #     anls = iou if iou >= 0.5 else 0.0
    #     return anls


    def exact_match(self, pred, gt):
        """Returns 1 if the prediction exactly matches the ground truth, else 0."""
        return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


    def get_anls(self, pred, gt):
        """Calculates ANLS (soft accuracy) between prediction and ground truth."""
        pred, gt = pred.lower().strip(), gt.lower().strip()
        dist = editdistance.eval(pred, gt)
        iou = 1 - dist / max(len(pred), len(gt))
        return iou if iou >= self.threshold else 0.0

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            if (entry["question"] == self.Q):
                # anls = max(
                #     self.get_anls(entry["pred_answer"], gt) for gt in entry["gt_answers"]
                # )
                anls = self.get_anls(entry["pred_answer"], entry["gt_answers"])
                pred_scores.append(anls)

        if len(pred_scores) == 0: # Avoid division by zero
            accuracy = 0.0
        else:
            accuracy = sum(pred_scores) / len(pred_scores) 
        return accuracy


class VaseVQADateEvaluator:

    def _extract_date_range(self, text):
        """Extracts a date range (e.g., '-450 to -400') from a given text."""
        match = re.search(r"(-?\d+)\s*to\s*(-?\d+)", text)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return start, end
        return None
    
    def set_q(self, Q):
        self.Q = Q

    def _is_date_correct(self, pred_date, gt_date):
        """Checks if the predicted date range overlaps with the ground truth range."""
        if not pred_date or not gt_date:
            return False  # If no valid date found, prediction is incorrect.
        
        pred_start, pred_end = pred_date
        gt_start, gt_end = gt_date
        
        return not (pred_end < gt_start or pred_start > gt_end)  # Overlap check

    def eval_pred_list(self, data_list):
        """
        Evaluates the accuracy of date predictions.
        :param data_list: List of dictionaries with keys 'question', 'pred_answer', and 'gt_answer'
        :return: Accuracy as a float (0.0 - 1.0)
        """
        correct_count = 0
        total_count = 0

        # import pdb
        for entry in tqdm(data_list):
            if (entry["question"] == self.Q):
                
                pred_date = self._extract_date_range(entry["pred_answer"])
                gt_date = self._extract_date_range(entry["gt_answers"])
                
                if self._is_date_correct(pred_date, gt_date):
                    correct_count += 1
                total_count += 1

        return correct_count / total_count if total_count > 0 else 0.0  # Avoid division by zero 


class DateAccuracyEvaluator:
    def __init__(self):
        # self.date_pattern = r"(\b\d{3,4}\s*(BCE?|AD)?\b)(\s*to\s*|\s*-\s*)(\b\d{3,4}\s*(BCE?|AD)?\b)"
        self.date_pattern = r'(-?\d+)(?:\s*(BC|B\.C\.|BCE))?|(\d+)'
    def _convert_number(self, value: str) -> int:
        """
        Convert a numeric string with optional era notation (BC/BCE/AD/CE) to integer.
        
        Handles:
        - BC/BCE as negative years
        - AD/CE as positive years (default if no era specified)
        - Pure numeric strings
        - Thousand separators (commas)
        
        Args:
            value: Input string containing numbers and optional era notation
            
        Returns:
            Integer representation of the input value
            
        Raises:
            ValueError: If no numeric value can be extracted
        """
        # Normalize era notations and remove commas
        cleaned = value.upper().replace(',', '').strip()
        
        # Identify era markers
        era_multiplier = 1
        for era in ['BC', 'BCE', 'B.C.', 'BC.']:
            if era in cleaned:
                cleaned = cleaned.replace(era, '')
                era_multiplier = -1
                break
                
        # Extract numeric part using regular expression
        match = re.search(r'-?\d+', cleaned)
        if not match:
            raise ValueError(f"No numeric value found in: {value}")
            
        return int(match.group()) * era_multiplier
    
    def _extract_dates(self, text):
        """
        Extract and process dates from text containing various date formats.
        Handles BC/AD conversion, ranges, and individual dates.
        Returns tuple of (start_date, end_date)
        """
        dates = []
        text = text.upper()  # Normalize for case-insensitive matching

        # First pass: Capture individual numbers with BC context
        pattern = r'(-?\d+)(?:\s*(BC|B\.C\.|BCE))?|(\d+)'
        for match in re.finditer(pattern, text):
            if match.group(1):
                num = int(match.group(1))
                if match.group(2) and num > 0:
                    num = -num
                dates.append(num)
            elif match.group(3):
                dates.append(int(match.group(3)))

        # Second pass: Find and correct date ranges with BC context
        range_pattern = r'(\d+)\s*[-to]+\s*(\d+)\s*(?:BC|B\.C\.|BCE)\b'
        for match in re.finditer(range_pattern, text):
            start = -int(match.group(1))
            end = -int(match.group(2))
            
            # Remove original positive values if present
            try:
                dates.remove(int(match.group(1)))
            except ValueError:
                pass
            try:
                dates.remove(int(match.group(2)))
            except ValueError:
                pass
            
            dates.extend([start, end])

        # Remove duplicates and sort
        unique_dates = sorted(list(set(dates)))
        
        if not unique_dates:
            return [None, None]
        
        return [min(unique_dates), max(unique_dates)]


    def _extract_date_range(self, text):
        """Extract date range using regular expressions"""
        matches = re.findall(self.date_pattern, text, re.IGNORECASE)
        # if not matches:
        #     return None
            
        try:
            # Convert era markers to negative numbers
            def convert_year(y):
                if 'BC' in y.upper():
                    return -int(y.replace('BC', '').strip())
                return int(y.replace('AD', '').strip())
                
            start = convert_year(matches[0][0])
            end = self.convert_number(matches[0][3])
            return sorted([start, end])
        except:
            return None

    def _compare_ranges(self, gt_range, pred_range):
        """Verify date range overlap"""
        if not pred_range:
            return 0.0
            
        # Calculate overlap ratio
        overlap_start = max(gt_range[0], pred_range[0])
        overlap_end = min(gt_range[1], pred_range[1])
        overlap = max(0, overlap_end - overlap_start)
        total_span = gt_range[1] - gt_range[0]
        return overlap / total_span if total_span !=0 else 0.0

    def set_q(self, Q):
        self.Q = Q

    def eval_pred_list(self, pred_list):
        """Main evaluation method"""
        scores = []
        for entry in pred_list:
            if (entry["question"] == self.Q):
                # Extract ground truth date range
                gt_dates = self._extract_dates(entry["gt_answers"])
                
                if None in gt_dates:
                    # scores.append(1.0 if "not available" in entry["gt_answers"] else 0.0)
                    continue
                
                # Extract predicted dates
                pred_range = self._extract_dates(entry["pred_answer"])
                if None in pred_range:
                    scores.append(0.0)
                    continue
                    
                # Calculate matching score
                max_score = self._compare_ranges(gt_dates, pred_range)
                scores.append(max_score)  
        if len(scores) == 0:
            return 0.0
        else:
            return sum(scores) / len(scores)