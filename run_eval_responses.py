import json
import os
import re
from datetime import datetime
from pathlib import Path

from evaluation.judges import judge_client
from evaluation.utils import EvalResult

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


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def truncate_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def normalize_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    return None


def extract_json_array(text: str) -> list[dict] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def build_batch_prompt(items: list[dict]) -> str:
    payload = json.dumps(items, ensure_ascii=True)
    return (
        "You are an impartial judge.\n"
        "For each item, decide if the Student Answer conveys the same meaning as the Ground Truth. "
        "If Ground Truth is empty, rate how correct and complete the Student Answer is for the Question.\n"
        "Return ONLY a JSON array of objects with fields: id (integer), score (number between 0.0 and 1.0).\n"
        f"Items: {payload}"
    )


def evaluate_accuracy_batch(items: list[dict]) -> tuple[list[EvalResult], dict[str, int]]:
    prompt = build_batch_prompt(items)
    response_text = judge_client.evaluate(prompt)

    if judge_client.last_error:
        error_meta = {
            "judge_error": judge_client.last_error,
        }
        if judge_client.last_error_code is not None:
            error_meta["judge_error_code"] = judge_client.last_error_code
        if judge_client.last_model_used:
            error_meta["judge_model"] = judge_client.last_model_used
        if judge_client.last_model_fallback:
            error_meta["judge_model_fallback"] = True
        results = [
            EvalResult(
                score=0.0,
                reasoning=f"Judge Error: {judge_client.last_error}",
                metadata=error_meta,
            )
            for _ in items
        ]
        return results, {"judge_error": len(items), "judge_parse_error": 0, "judge_missing_score": 0}

    data = extract_json_array(response_text)
    if data is None:
        error_meta = {
            "judge_error": "batch_parse_failed",
            "judge_raw_response": truncate_text(response_text),
        }
        if judge_client.last_model_used:
            error_meta["judge_model"] = judge_client.last_model_used
        if judge_client.last_model_fallback:
            error_meta["judge_model_fallback"] = True
        results = [
            EvalResult(
                score=0.0,
                reasoning="Judge Error: batch_parse_failed",
                metadata=error_meta,
            )
            for _ in items
        ]
        return results, {"judge_error": 0, "judge_parse_error": len(items), "judge_missing_score": 0}

    scores_by_id: dict[int, float] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        score = normalize_score(entry.get("score"))
        if raw_id is None or score is None:
            continue
        try:
            scores_by_id[int(raw_id)] = score
        except (TypeError, ValueError):
            continue

    results: list[EvalResult] = []
    missing = 0
    for item in items:
        item_id = item["id"]
        score = scores_by_id.get(item_id)
        if score is None:
            missing += 1
            meta = {"judge_error": "missing_score"}
            if judge_client.last_model_used:
                meta["judge_model"] = judge_client.last_model_used
            if judge_client.last_model_fallback:
                meta["judge_model_fallback"] = True
            results.append(
                EvalResult(
                    score=0.0,
                    reasoning="Judge Error: missing_score",
                    metadata=meta,
                )
            )
            continue

        meta = {}
        if judge_client.last_model_used:
            meta["judge_model"] = judge_client.last_model_used
        if judge_client.last_model_fallback:
            meta["judge_model_fallback"] = True
        results.append(EvalResult(score=score, reasoning="LLM Judge Evaluated", metadata=meta))

    return results, {"judge_error": 0, "judge_parse_error": 0, "judge_missing_score": missing}


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
    judge_error_count = 0
    judge_parse_error_count = 0
    judge_missing_score_count = 0
    cases = []
    for i, record in enumerate(records):
        query = record.get("prompt", "")
        answer = record.get("response", "")
        ground_truth = record.get("ground_truth", "")
        dataset_id = record.get("dataset_id", "unknown")
        if not query:
            continue
        cases.append(
            {
                "id": i,
                "dataset_id": dataset_id,
                "query": query,
                "system_answer": answer,
                "ground_truth": ground_truth,
            }
        )

    batch_size = max(1, int(os.getenv("JUDGE_BATCH_SIZE", "10")))
    batches = chunked(cases, batch_size)

    for batch_index, batch in enumerate(batches):
        print(f"\nBatch {batch_index + 1}/{len(batches)}: {len(batch)} cases")
        items = [
            {
                "id": item["id"],
                "question": item["query"],
                "ground_truth": item["ground_truth"],
                "student_answer": item["system_answer"],
            }
            for item in batch
        ]

        l1_results, counts = evaluate_accuracy_batch(items)
        judge_error_count += counts["judge_error"]
        judge_parse_error_count += counts["judge_parse_error"]
        judge_missing_score_count += counts["judge_missing_score"]

        for item, l1 in zip(batch, l1_results):
            metadata = {"source": "TestEval2LLM"}
            if l1.metadata:
                metadata.update(l1.metadata)

            results.append(
                {
                    "id": item["id"],
                    "dataset_id": item["dataset_id"],
                    "query": item["query"],
                    "system_answer": item["system_answer"],
                    "ground_truth": item["ground_truth"],
                    "scores": {"accuracy": l1.score},
                    "reasoning": {"accuracy": l1.reasoning},
                    "metadata": metadata,
                }
            )

    avg_acc = sum(r["scores"]["accuracy"] for r in results) / len(results)
    summary = {
        "timestamp": str(datetime.now()),
        "total_cases": len(results),
        "average_accuracy": avg_acc,
        "error_counts": {
            "judge_accuracy": judge_error_count,
            "judge_parse": judge_parse_error_count,
            "judge_missing_score": judge_missing_score_count,
        },
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
