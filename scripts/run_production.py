"""Paper A unified primary-production runner.

IMPORTANT:
- Production responses only.
- QA/pilot responses must never be written here.
- One independent request per benchmark item.
- Successful requests, refusals, and other model behaviours are never
  regenerated for answer quality.
- Execution is blocked unless production_manifest_v1.0.json explicitly
  authorizes production.
"""

from pathlib import Path
import sys
import os
import json
import time
import argparse
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_loader import load_and_validate


MANIFEST_PATH = ROOT / "config" / "production_manifest_v1.0.json"

MANIFEST = json.loads(
    MANIFEST_PATH.read_text(encoding="utf-8")
)
PLAN_PATH = ROOT / "config" / "production_execution_plan_v1.0.json"

PLAN = json.loads(
    PLAN_PATH.read_text(encoding="utf-8")
)

# ---------------------------------------------------------------------
# HARD PRODUCTION INTERLOCK
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# HARD DUAL PRODUCTION INTERLOCK
# ---------------------------------------------------------------------

if MANIFEST.get("production_locked") is not True:
    raise RuntimeError(
        "ABORT: production protocol is not locked. "
        "No API request has been sent."
    )

if PLAN.get("execution_plan_locked") is not True:
    raise RuntimeError(
        "ABORT: production execution plan is not locked. "
        "No API request has been sent."
    )

if (
    MANIFEST.get("experiment_id")
    != PLAN.get("experiment_id")
):
    raise RuntimeError(
        "ABORT: manifest and execution-plan experiment IDs differ. "
        "No API request has been sent."
    )

if (
    MANIFEST.get("production_protocol_version")
    != PLAN.get("production_protocol_version")
):
    raise RuntimeError(
        "ABORT: manifest and execution-plan protocol versions differ. "
        "No API request has been sent."
    )

if MANIFEST.get("production_execution_authorized") is not True:
    raise RuntimeError(
        "ABORT: production execution is NOT authorized. "
        "No API request has been sent."
    )

if PLAN.get("stage4_execution_authorized") is not True:
    raise RuntimeError(
        "ABORT: Stage 4 execution is NOT authorized. "
        "No API request has been sent."
    )


# ---------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    choices=[
        "openai",
        "gemini",
        "claude",
        "llama"
    ]
)

args = parser.parse_args()


# ---------------------------------------------------------------------
# BENCHMARK VALIDATION
# ---------------------------------------------------------------------

benchmark = ROOT / MANIFEST["benchmark_filename"]

records = load_and_validate(
    benchmark,
    MANIFEST["benchmark_sha256"]
)

if len(records) != 1080:
    raise RuntimeError(
        f"Benchmark invariant failed: "
        f"expected 1080 items, found {len(records)}."
    )

if len({r["item_id"] for r in records}) != 1080:
    raise RuntimeError(
        "Benchmark invariant failed: item IDs are not unique."
    )


# ---------------------------------------------------------------------
# PROVIDER CONFIGURATION
# ---------------------------------------------------------------------

MODEL_CONFIG = {
    "openai": {
        "family": "OpenAI GPT",
        "provider": "openai",
        "model_id": "gpt-5.6-sol",
    },

    "gemini": {
        "family": "Google Gemini",
        "provider": "google",
        "model_id": "gemini-3.8-flash",
    },

    "claude": {
        "family": "Anthropic Claude",
        "provider": "anthropic",
        "model_id": "claude-fable-5-1",
    },

    "llama": {
        "family": "Meta Llama",
        "provider": "novita",
        "model_id":
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
}

cfg = MODEL_CONFIG[args.model]


# ---------------------------------------------------------------------
# ACCESS-STATE GUARD
# ---------------------------------------------------------------------

manifest_model = next(
    m for m in MANIFEST["models"]
    if m["model_id"] == cfg["model_id"]
)

access_state = manifest_model.get(
    "production_access"
)

if access_state != "production_ready":

    raise RuntimeError(
        f"ABORT: {args.model} production access state is "
        f"{access_state!r}. "
        "Provider is not authorized for production."
    )


# ---------------------------------------------------------------------
# CREDENTIAL CHECK + LAZY ADAPTER CONSTRUCTION
# ---------------------------------------------------------------------

if args.model == "openai":

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    from providers.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(
        model_id=cfg["model_id"],
        max_output_tokens=1024
    )


elif args.model == "gemini":

    if not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ):
        raise RuntimeError(
            "GEMINI_API_KEY/GOOGLE_API_KEY is not set."
        )

    from providers.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(
        model_id=cfg["model_id"],
        max_output_tokens=1024
    )


elif args.model == "claude":

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set."
        )

    from providers.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter(
        model_id=cfg["model_id"],
        max_tokens=1024
    )


elif args.model == "llama":

    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set."
        )

    from providers.llama_adapter import LlamaAdapter

    adapter = LlamaAdapter(
        model_id=cfg["model_id"],
        provider="novita",
        max_tokens=1024,
        temperature=0
    )


# ---------------------------------------------------------------------
# OUTPUT LOCATIONS
# ---------------------------------------------------------------------

raw_dir = ROOT / "outputs" / "production" / "raw"
error_dir = ROOT / "outputs" / "production" / "errors"

raw_dir.mkdir(
    parents=True,
    exist_ok=True
)

error_dir.mkdir(
    parents=True,
    exist_ok=True
)

out = raw_dir / (
    f"primary_{args.model}.jsonl"
)

errout = error_dir / (
    f"primary_{args.model}_technical_errors.jsonl"
)


# ---------------------------------------------------------------------
# SAFE RESUME
# ---------------------------------------------------------------------

completed = set()

if out.exists():

    for line in out.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            x = json.loads(line)

            if x.get("request_status") == "SUCCESS":
                completed.add(
                    x["item_id"]
                )

        except Exception:
            pass


# ---------------------------------------------------------------------
# ERROR CLASSIFICATION
# ---------------------------------------------------------------------

TERMINAL_HTTP = {
    400,
    401,
    402,
    403,
    404
}

RETRYABLE_HTTP = {
    408,
    409,
    429,
    500,
    502,
    503,
    504
}


def get_http_status(exc):

    for attr in (
        "status_code",
        "status"
    ):
        value = getattr(
            exc,
            attr,
            None
        )

        if isinstance(value, int):
            return value

    response = getattr(
        exc,
        "response",
        None
    )

    if response is not None:
        value = getattr(
            response,
            "status_code",
            None
        )

        if isinstance(value, int):
            return value

    return None


def quota_or_credit_terminal(message):

    text = message.lower()

    indicators = [
        "credit_balance_exhausted",
        "depleted your monthly included credits",
        "payment required",
        "purchase pre-paid credits",
        "insufficient credits",
        "quota exhausted",
        "billing",
        "denied access"
    ]

    return any(
        x in text
        for x in indicators
    )


def should_retry(exc):

    status = get_http_status(exc)
    message = str(exc)

    if quota_or_credit_terminal(message):
        return False

    if status in TERMINAL_HTTP:
        return False

    if status in RETRYABLE_HTTP:
        return True

    # Unknown exceptions are not automatically retried.
    # This prevents uncontrolled duplicate requests.
    return False


# ---------------------------------------------------------------------
# MODEL-OUTCOME CLASSIFICATION
# ---------------------------------------------------------------------

def classify_outcome(ans):

    text = (
        ans.get("raw_response")
        or ""
    ).strip()

    stop_values = [
        ans.get("stop_reason"),
        ans.get("finish_reason"),
        ans.get("status_returned")
    ]

    stop_text = " ".join(
        str(x).lower()
        for x in stop_values
        if x is not None
    )

    if "refusal" in stop_text:
        return "REFUSAL"

    if not text:
        return "EMPTY"

    return "RESPONSE"


# ---------------------------------------------------------------------
# PRODUCTION LOOP
# ---------------------------------------------------------------------

for r in records:

    item_id = r["item_id"]

    if item_id in completed:
        print(
            "SKIP",
            item_id,
            "already recorded"
        )
        continue

    rec = {
        "run_type":
            "PRIMARY_PRODUCTION",

        "production_protocol_version":
            MANIFEST[
                "production_protocol_version"
            ],

        "experiment_id":
            MANIFEST["experiment_id"],

        "benchmark_version":
            MANIFEST["benchmark_version"],

        "benchmark_sha256":
            MANIFEST["benchmark_sha256"],

        "item_id":
            item_id,

        "concept_id":
            r["concept_id"],

        "language":
            r["language"],

        "language_code":
            r["language_code"],

        "task_code":
            r["task_code"],

        "dataset_role":
            r["dataset_role"],

        "condition":
            MANIFEST["primary_condition"],

        "model_family":
            cfg["family"],

        "provider":
            cfg["provider"],

        "model_id":
            cfg["model_id"],

        "system_instruction":
            MANIFEST["system_instruction"],

        "prompt":
            r[
                "FINAL target-language prompt"
            ],

        "request_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "request_status":
            "PENDING",

        "model_outcome":
            None
    }

    last_exception = None

    for attempt in range(1, 4):

        rec["attempt"] = attempt

        try:

            ans = adapter.generate(
                MANIFEST[
                    "system_instruction"
                ],
                rec["prompt"]
            )

            rec.update(ans)

            rec["request_status"] = (
                "SUCCESS"
            )

            rec["model_outcome"] = (
                classify_outcome(ans)
            )

            last_exception = None
            break

        except Exception as exc:

            last_exception = exc

            rec["error_type"] = (
                type(exc).__name__
            )

            rec["error_message"] = (
                str(exc)
            )

            rec["http_status"] = (
                get_http_status(exc)
            )

            retry = should_retry(exc)

            rec["retryable_error"] = (
                retry
            )

            if not retry:
                break

            if attempt < 3:
                time.sleep(
                    2 ** (attempt - 1)
                )


    rec["response_timestamp_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    if rec["request_status"] == "SUCCESS":

        target = out

    else:

        rec["request_status"] = (
            "TECHNICAL_ERROR"
        )

        rec["model_outcome"] = (
            "TECHNICAL_ERROR"
        )

        target = errout


    with target.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                rec,
                ensure_ascii=False,
                default=str
            )
            + "\n"
        )


    print(
        item_id,
        rec["request_status"],
        rec["model_outcome"],
        f"attempt={rec['attempt']}"
    )


    # Terminal infrastructure/access failure:
    # stop the model arm rather than generating hundreds
    # of identical technical-error records.
    if (
        rec["request_status"]
        == "TECHNICAL_ERROR"
        and rec.get(
            "retryable_error"
        ) is False
    ):

        print(
            "PRODUCTION ARM HALTED "
            "AFTER TERMINAL ERROR."
        )

        break