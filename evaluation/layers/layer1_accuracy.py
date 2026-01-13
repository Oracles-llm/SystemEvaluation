from evaluation.judges import judge_client
from evaluation.utils import EvalResult

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
    
    try:
        score = float(response_text.strip())
    except:
        score = 0.0
        
    return EvalResult(score=score, reasoning="LLM Judge Evaluated", metadata={})
