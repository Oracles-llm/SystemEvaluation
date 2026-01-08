import os
from datasets import load_dataset

# 1. Define where you want to save them locally
LOCAL_DIR = "./datasets/"
os.makedirs(LOCAL_DIR, exist_ok=True)

DATASETS = [
    {
        "id": "gsm8k_main",
        "dataset": "gsm8k",
        "config": "main",
        "split": "test",
        "filename": "gsm8k_test.json",
        "description": "GSM8K (math reasoning)",
    },
    {
        "id": "ai2_arc_challenge",
        "dataset": "ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "filename": "arc_challenge_test.json",
        "description": "ARC-Challenge (science/logic)",
    },
    {
        "id": "ifeval",
        "dataset": "google/IFEval",
        "config": None,
        "split": "train",
        "filename": "ifeval.json",
        "description": "IFEval (instruction following)",
    },
    # MMLU: keep a small selection to avoid massive downloads
    {
        "id": "mmlu_college_cs",
        "dataset": "cais/mmlu",
        "config": "college_computer_science",
        "split": "test",
        "filename": "mmlu_college_computer_science.json",
        "description": "MMLU - college_computer_science",
    },
    {
        "id": "mmlu_elementary_math",
        "dataset": "cais/mmlu",
        "config": "elementary_mathematics",
        "split": "test",
        "filename": "mmlu_elementary_mathematics.json",
        "description": "MMLU - elementary_mathematics",
    },
    {
        "id": "bbh_logical_deduction",
        "dataset": "lukaemon/bbh",
        "config": "logical_deduction_five_objects",
        "split": "test",
        "filename": "bbh_logical_deduction.json",
        "description": "BBH - logical deduction subset",
    },
]


def download_dataset(entry, local_dir=LOCAL_DIR):
    target = os.path.join(local_dir, entry["filename"])
    if os.path.exists(target):
        print(f"Skipping (exists): {entry['filename']}")
        return "skipped"

    ds_name = entry["dataset"]
    cfg = entry.get("config")
    split = entry.get("split")

    try:
        print(f"\nDownloading {entry['description']} -> {entry['filename']}")
        if cfg:
            ds = load_dataset(ds_name, cfg, split=split)
        else:
            ds = load_dataset(ds_name, split=split)

        ds.to_json(target)
        print(f"Saved {len(ds)} examples to {entry['filename']}")
        return "downloaded"
    except Exception as e:
        print(f"Failed to download {entry['id']}: {e}")
        return "failed"


def main():
    print(f"Data directory: {LOCAL_DIR}\n")
    print("Datasets configured for download:")
    for e in DATASETS:
        print(f" - {e['filename']}: {e['description']}")

    # Download missing datasets
    results = {"downloaded": [], "skipped": [], "failed": []}
    for e in DATASETS:
        status = download_dataset(e)
        results.setdefault(status, []).append(e["filename"]) if status else None

    print("\nSummary:")
    print(f" - Downloaded: {len(results.get('downloaded', []))}")
    print(f" - Skipped (already present): {len(results.get('skipped', []))}")
    print(f" - Failed: {len(results.get('failed', []))}")


if __name__ == "__main__":
    main()