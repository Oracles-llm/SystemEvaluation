import re

from evaluation.judges import judge_client
from evaluation.utils import EvalResult


def parse_score(text: str) -> float | None:
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    for token in reversed(matches):
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
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
    if judge_client.last_error:
        metadata = {"judge_error": judge_client.last_error}
        if judge_client.last_error_code is not None:
            metadata["judge_error_code"] = judge_client.last_error_code
        if judge_client.last_model_used:
            metadata["judge_model"] = judge_client.last_model_used
        if judge_client.last_model_fallback:
            metadata["judge_model_fallback"] = True
        return EvalResult(
            score=0.0,
            reasoning=f"Judge Error: {judge_client.last_error}",
            metadata=metadata,
        )
    score = parse_score(response_text)
    if score is None:
        score = 0.0
    score = max(0.0, min(score, 1.0))
        
    return EvalResult(score=score, reasoning="LLM Judge Evaluated", metadata={})
