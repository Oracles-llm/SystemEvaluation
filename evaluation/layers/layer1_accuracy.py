import re

from evaluation.judges import judge_client
from evaluation.utils import EvalResult


def parse_score(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

def evaluate_accuracy(query, actual_answer, ground_truth):
    if ground_truth:
        prompt = f"""
        You are an impartial judge.
        Question: {query}
        Ground Truth: {ground_truth}
        Student Answer: {actual_answer}

        Does the Student Answer convey the same meaning as the Ground Truth?
        Return ONLY a score between 0.0 and 1.0.
        """
    else:
        prompt = f"""
        You are an impartial judge.
        Question: {query}
        Student Answer: {actual_answer}

        Rate how correct and complete the Student Answer is for the Question.
        Return ONLY a score between 0.0 and 1.0.
        """
    
    response_text = judge_client.evaluate(prompt)
    score = parse_score(response_text)
    if score is None:
        score = 0.0
        
    return EvalResult(score=score, reasoning="LLM Judge Evaluated", metadata={})
