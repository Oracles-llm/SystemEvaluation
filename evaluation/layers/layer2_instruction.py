import json
from evaluation.utils import EvalResult

def evaluate_json_format(text_output):
    """Checks if output is valid JSON."""
    try:
        json.loads(text_output)
        return EvalResult(score=1.0, reasoning="Valid JSON", metadata={})
    except:
        return EvalResult(score=0.0, reasoning="Invalid JSON format", metadata={})

def evaluate_instruction_following(query, actual_answer, constraints):
    """
    Example constraint check.
    constraints = ["no_apology", "under_100_words"]
    """
    score = 1.0
    reasoning = []
    
    if "no_apology" in constraints:
        if "sorry" in actual_answer.lower() or "apologize" in actual_answer.lower():
            score -= 0.5
            reasoning.append("Failed constraint: no_apology")
            
    return EvalResult(score=max(0.0, score), reasoning="; ".join(reasoning), metadata={})