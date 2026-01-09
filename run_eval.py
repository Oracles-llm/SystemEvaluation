import json
import os
from datetime import datetime

# Import System & Layers
from MainProject.src.my_system import MyEdgeSystem
from evaluation.layers import (
    evaluate_accuracy, 
    evaluate_efficiency,
    evaluate_rag_context
)

# 1. Configuration
DATASET_PATH = "./datasets/gsm8k_test.json" # Change this based on what you test
OUTPUT_FILE = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def main():
    print("Starting System Evaluation...")
    
    # Initialize your system
    system = MyEdgeSystem(config_name="rag_optimized")
    
    # Load Data (Handling JSONL or JSON list)
    data = []
    try:
        with open(DATASET_PATH, "r") as f:
            # Try loading as list
            try:
                data = json.load(f)
            except:
                # Fallback to JSONL
                f.seek(0)
                data = [json.loads(line) for line in f]
        
        # Limit to 5 for testing purposes
        data = data[:5]
        print(f"Loaded {len(data)} test cases.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}. Run dataset_downloader.py first.")
        return

    results = []

    for i, case in enumerate(data):
        # Extract fields (GSM8K format usually has 'question' and 'answer')
        query = case.get('question', '')
        ground_truth = case.get('answer', '')
        
        print(f"\nTest Case {i+1}: {query[:50]}...")
        
        # --- RUN SYSTEM ---
        sys_out = system.predict(query)
        # ------------------
        
        # --- EVALUATE LAYERS ---
        
        # Layer 1: Accuracy (Using Judge)
        l1 = evaluate_accuracy(query, sys_out['answer'], ground_truth)
        
        # Layer 3: RAG (Mock keywords for now, assuming ground truth has them)
        # In real logic, you'd extract keywords from ground_truth first
        l3 = evaluate_rag_context(sys_out['retrieved_docs'], ["July", "Independence"]) 
        
        # Layer 4: Efficiency
        word_count = len(sys_out['answer'].split())
        l4 = evaluate_efficiency(sys_out['latency'], word_count)
        
        # Combine Result
        report_item = {
            "id": i,
            "query": query,
            "system_answer": sys_out['answer'],
            "scores": {
                "accuracy": l1.score,
                "rag": l3.score,
                "efficiency": l4.score
            },
            "metrics": {
                "latency": sys_out['latency'],
                "ram_mb": l4.metadata['ram_mb']
            },
            "reasoning": {
                "accuracy": l1.reasoning,
                "efficiency": l4.reasoning
            }
        }
        results.append(report_item)

    # Calculate Averages
    avg_acc = sum(r['scores']['accuracy'] for r in results) / len(results)
    
    summary = {
        "timestamp": str(datetime.now()),
        "total_cases": len(results),
        "average_accuracy": avg_acc,
        "details": results
    }

    # Save Report
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nEvaluation Complete! Report saved to {OUTPUT_FILE}")
    print(f"Average Accuracy: {avg_acc:.2f}")

if __name__ == "__main__":
    main()