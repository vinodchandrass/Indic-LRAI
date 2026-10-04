from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

QUEUE = ROOT / "data" / "run_queue.csv"
RAW = ROOT / "outputs" / "production" / "raw"
OUT = ROOT / "outputs" / "production" / "primary_completion_matrix.csv"

MODELS = ["claude", "llama", "openai", "gemini"]

# ---------------------------------------------------------
# Load frozen 1,080-item benchmark queue
# ---------------------------------------------------------

with QUEUE.open("r", encoding="utf-8-sig", newline="") as f:
    benchmark = list(csv.DictReader(f))

if len(benchmark) != 1080:
    raise RuntimeError(
        f"Expected 1080 benchmark items, found {len(benchmark)}"
    )

item_ids = [r["item_id"] for r in benchmark]

if len(set(item_ids)) != 1080:
    raise RuntimeError("Benchmark queue contains duplicate item_id values")

# ---------------------------------------------------------
# Read successful production records exactly as
# run_production.py SAFE RESUME does
# ---------------------------------------------------------

successful = {}

for model in MODELS:

    path = RAW / f"primary_{model}.jsonl"
    ids = set()

    if path.exists():

        for line in path.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            try:
                x = json.loads(line)

                if x.get("request_status") == "SUCCESS":
                    ids.add(x["item_id"])

            except Exception:
                pass

    unknown = ids - set(item_ids)

    if unknown:
        raise RuntimeError(
            f"{model}: successful item IDs not present "
            f"in benchmark: {sorted(unknown)}"
        )

    successful[model] = ids

# ---------------------------------------------------------
# Construct model x benchmark completion matrix
# ---------------------------------------------------------

fields = [
    "model",
    "item_id",
    "language",
    "language_code",
    "task_code",
    "task_family",
    "dataset_role",
    "condition",
    "collection_status"
]

rows = []

for model in MODELS:

    for r in benchmark:

        item_id = r["item_id"]

        rows.append({
            "model": model,
            "item_id": item_id,
            "language": r["language"],
            "language_code": r["language_code"],
            "task_code": r["task_code"],
            "task_family": r["task_family"],
            "dataset_role": r["dataset_role"],
            "condition": r["condition"],
            "collection_status":
                "COMPLETE"
                if item_id in successful[model]
                else "REMAINING"
        })

# ---------------------------------------------------------
# Write audit artifact
# ---------------------------------------------------------

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

complete = sum(
    r["collection_status"] == "COMPLETE"
    for r in rows
)

remaining = sum(
    r["collection_status"] == "REMAINING"
    for r in rows
)

print("=" * 70)
print("PRIMARY COMPLETION MATRIX AUDIT")
print("=" * 70)

for model in MODELS:

    n = len(successful[model])

    print(
        f"{model.upper():8s} "
        f"COMPLETE={n:4d} "
        f"REMAINING={1080-n:4d}"
    )

print("-" * 70)
print("TOTAL CELLS :", len(rows))
print("COMPLETE    :", complete)
print("REMAINING   :", remaining)
print("OUTPUT      :", OUT)

assert len(rows) == 4320
assert complete + remaining == 4320

print("=" * 70)
print("AUDIT PASS")
print("=" * 70)