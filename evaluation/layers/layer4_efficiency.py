import psutil
import os
from evaluation.utils import EvalResult

def evaluate_efficiency(latency, word_count):
    # Pass/Fail Thresholds (Example: Mobile requirements)
    MAX_LATENCY = 2.0 # seconds
    MIN_TPS = 5.0     # tokens per sec approx
    
    tps = word_count / latency if latency > 0 else 0
    
    score = 1.0
    reasons = []
    
    if latency > MAX_LATENCY:
        score -= 0.5
        reasons.append(f"High Latency ({latency:.2f}s > {MAX_LATENCY}s)")
        
    if tps < MIN_TPS:
        score -= 0.5
        reasons.append(f"Low Speed ({tps:.1f} wps < {MIN_TPS})")
        
    # Get RAM usage of current process
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    
    return EvalResult(
        score=max(0.0, score),
        reasoning="; ".join(reasons) if reasons else "Efficiency Checks Passed",
        metadata={"ram_mb": ram_mb, "tps": tps, "latency": latency}
    )
