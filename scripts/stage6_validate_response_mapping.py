from pathlib import Path
import json
import pandas as pd

BENCHMARK = Path(
    r"PaperA_9Language_1080_FINAL_MASTER_BENCHMARK(3).xlsx"
)

RAW_DIR = Path(r"outputs\production\raw")

FILES = {
    "claude": RAW_DIR / "primary_claude.jsonl",
    "llama": RAW_DIR / "primary_llama.jsonl",
}

print("=" * 72)
print("STAGE 6.1 — RESPONSE ↔ BENCHMARK MAPPING AUDIT")
print("=" * 72)

# Load benchmark
df = pd.read_excel(
    BENCHMARK,
    sheet_name="Master_1080"
)

print("Benchmark rows:", len(df))
print("Benchmark columns:")
print(list(df.columns))

# Detect item-ID column
possible = [
    c for c in df.columns
    if str(c).strip().lower() in
    {"item_id", "item id", "itemid"}
]

if len(possible) != 1:
    raise RuntimeError(
        f"Could not uniquely identify benchmark item-ID column: {possible}"
    )

id_col = possible[0]

benchmark_ids = df[id_col].astype(str).str.strip()

print("\nItem-ID column:", id_col)
print("Unique benchmark IDs:", benchmark_ids.nunique())
print(
    "Benchmark duplicate IDs:",
    len(benchmark_ids) - benchmark_ids.nunique()
)

benchmark_set = set(benchmark_ids)

all_response_ids = []

for model, path in FILES.items():

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    response_ids = [
        str(r["item_id"]).strip()
        for r in rows
    ]

    all_response_ids.extend(
        (model, item_id)
        for item_id in response_ids
    )

    missing = [
        item_id
        for item_id in response_ids
        if item_id not in benchmark_set
    ]

    print(f"\n{model.upper()}")
    print("Responses:", len(response_ids))
    print("Unique response IDs:", len(set(response_ids)))
    print("Unmatched benchmark IDs:", len(missing))

    if missing:
        print("UNMATCHED:", missing)

print("\n" + "-" * 72)

combined_ids = [item_id for _, item_id in all_response_ids]

print("Collected model-response records:", len(combined_ids))
print("Distinct benchmark items represented:", len(set(combined_ids)))

unmatched_total = [
    (model, item_id)
    for model, item_id in all_response_ids
    if item_id not in benchmark_set
]

print("Total unmatched records:", len(unmatched_total))

assert len(df) == 1080
assert benchmark_ids.nunique() == 1080
assert len(combined_ids) == len(all_response_ids)
assert len(unmatched_total) == 0

print("\nSTAGE 6.1 RESPONSE MAPPING AUDIT: PASS")