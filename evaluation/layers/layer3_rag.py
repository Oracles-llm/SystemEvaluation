from evaluation.utils import EvalResult
# In a real project, import 'ragas' here for advanced metrics

def evaluate_rag_context(retrieved_docs, ground_truth_keywords):
    """
    Simple keyword overlap check. 
    In future, replace this with RAGAS 'Context Precision'.
    """
    if not retrieved_docs:
        return EvalResult(score=0.0, reasoning="No docs retrieved", metadata={})

    hits = 0
    full_text = " ".join(retrieved_docs).lower()
    
    for kw in ground_truth_keywords:
        if kw.lower() in full_text:
            hits += 1
            
    score = hits / len(ground_truth_keywords) if ground_truth_keywords else 1.0
    
    return EvalResult(score=score, reasoning=f"Found {hits} keywords", metadata={"hits": hits})