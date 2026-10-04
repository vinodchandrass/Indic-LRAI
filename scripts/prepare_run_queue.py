from pathlib import Path
import csv, json, sys
from benchmark_loader import load_and_validate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "config" / "experiment_manifest.json").read_text(encoding="utf-8"))

# Put the frozen workbook path here, or pass it as argv[1].
benchmark = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / MANIFEST["benchmark_filename"]

records = load_and_validate(benchmark, MANIFEST["benchmark_sha256"])

out = ROOT / "data" / "run_queue.csv"
fields = [
    "experiment_id","benchmark_version","item_id","concept_id","language","language_code",
    "task_code","task_family","dataset_role","target_script","condition",
    "prompt","reference_answer","model_family","model_id","status"
]

with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in records:
        w.writerow({
            "experiment_id": MANIFEST["experiment_id"],
            "benchmark_version": MANIFEST["benchmark_version"],
            "item_id": r["item_id"],
            "concept_id": r["concept_id"],
            "language": r["language"],
            "language_code": r["language_code"],
            "task_code": r["task_code"],
            "task_family": r["task_family"],
            "dataset_role": r["dataset_role"],
            "target_script": r["target_script"],
            "condition": "direct_target_language",
            "prompt": r["FINAL target-language prompt"],
            "reference_answer": r["FINAL target-language reference answer"],
            "model_family": "",
            "model_id": "",
            "status": "PENDING"
        })

print(f"PASS: validated {len(records)} frozen benchmark items")
print(f"Created: {out}")
