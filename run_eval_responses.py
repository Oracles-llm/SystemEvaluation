import json
import os
from datetime import datetime
from pathlib import Path

from evaluation.layers import evaluate_accuracy

RESPONSES_PATH = os.getenv(
    "RESPONSES_FILE",
    os.path.join("..", "TestEval2LLM", "results", "responses.jsonl"),
)

OUTPUT_FILE = os.getenv(
    "EVAL_OUTPUT_FILE",
    os.path.join(
        "results",
        f"eval_report_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    ),
)


def load_responses(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    print("Starting System Evaluation for TestEval2LLM responses...")

    responses_path = Path(RESPONSES_PATH).resolve()
    if not responses_path.exists():
        print(f"Error: Responses file not found at {responses_path}")
        return

    records = load_responses(responses_path)
    if not records:
        print("No responses found to evaluate.")
        return

    results = []
    for i, record in enumerate(records):
        query = record.get("prompt", "")
        answer = record.get("response", "")
        if not query:
            continue

        print(f"\nTest Case {i + 1}: {query[:50]}...")
        l1 = evaluate_accuracy(query, answer, "")

        results.append(
            {
                "id": i,
                "query": query,
                "system_answer": answer,
                "scores": {"accuracy": l1.score},
                "reasoning": {"accuracy": l1.reasoning},
                "metadata": {"source": "TestEval2LLM"},
            }
        )

    avg_acc = sum(r["scores"]["accuracy"] for r in results) / len(results)
    summary = {
        "timestamp": str(datetime.now()),
        "total_cases": len(results),
        "average_accuracy": avg_acc,
        "details": results,
    }

    output_path = Path(OUTPUT_FILE).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nEvaluation Complete! Report saved to {output_path}")
    print(f"Average Accuracy: {avg_acc:.2f}")


if __name__ == "__main__":
    main()
