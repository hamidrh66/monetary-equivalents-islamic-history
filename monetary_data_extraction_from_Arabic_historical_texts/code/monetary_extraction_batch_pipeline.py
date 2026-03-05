#!/usr/bin/env python3
"""
Manifest-driven OpenAI Batch pipeline for extracting monetary information from Arabic MIUs.

This script is a CLI equivalent of the dissertation notebook:
- deterministic chunking into JSONL parts
- resumability via manifest.json
- optional auto-retry for failed/cancelled/expired chunks
- per-chunk output downloads
- schema-validated parsing and merge back into the MIU corpus

Typical usage (end-to-end):

    python monetary_extraction_batch_pipeline.py run \
        --miu-json sample_MIU_corpus.json \
        --few-shot few_shot_examples.json \
        --work-dir batch_work

You can also run each stage separately:

    python monetary_extraction_batch_pipeline.py build
    python monetary_extraction_batch_pipeline.py submit
    python monetary_extraction_batch_pipeline.py poll
    python monetary_extraction_batch_pipeline.py retry
    python monetary_extraction_batch_pipeline.py download
    python monetary_extraction_batch_pipeline.py merge

Requirements:
- pip install openai jsonschema
- export OPENAI_API_KEY="..."

Notes:
- This script targets the Responses API endpoint (/v1/responses) in Batch mode.
- Structured outputs are enforced via text.format with a JSON Schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError
from openai import OpenAI


# -----------------------
# Structured output schema and system rules
# -----------------------
# Structured Outputs JSON Schema for EACH record
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "MonetaryInfoResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "monetary_info": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "exact_phrase_in_text_Arabic": {"type": "string"},
                        "amount_of_money_in_digits": {"type": "string"},
                        "currency_Arabic": {"type": "string"},
                        "currency": {"type": "string"},
                        "year_of_transaction_Arabic": {"type": "string"},
                        "year_of_transaction_in_digits": {"type": "string"},
                        "other_years_Arabic": {"type": "array", "items": {"type": "string"}},
                        "other_years_in_digits": {"type": "array", "items": {"type": "string"}},
                        "location_of_transaction_Arabic": {"type": "string"},
                        "amount_of_item_in_transaction_Arabic": {"type": "string"},
                        "payer_or_owner_Arabic": {"type": "string"},
                        "payer_or_owner_categories_controlled": {"type": "array", "items": {"type": "string"}},
                        "payer_or_owner_categories_free": {"type": "array", "items": {"type": "string"}},
                        "payee_Arabic": {"type": "string"},
                        "payee_categories_controlled": {"type": "array", "items": {"type": "string"}},
                        "payee_categories_free": {"type": "array", "items": {"type": "string"}},
                        "money_paid_for_Arabic": {"type": "string"},
                        "reason_of_transaction": {"type": "string"},
                        "reason_categories_controlled": {"type": "array", "items": {"type": "string"}},
                        "reason_categories_free": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "exact_phrase_in_text_Arabic",
                        "amount_of_money_in_digits",
                        "currency_Arabic",
                        "currency",
                        "year_of_transaction_Arabic",
                        "year_of_transaction_in_digits",
                        "other_years_Arabic",
                        "other_years_in_digits",
                        "location_of_transaction_Arabic",
                        "amount_of_item_in_transaction_Arabic",
                        "payer_or_owner_Arabic",
                        "payer_or_owner_categories_controlled",
                        "payer_or_owner_categories_free",
                        "payee_Arabic",
                        "payee_categories_controlled",
                        "payee_categories_free",
                        "money_paid_for_Arabic",
                        "reason_of_transaction",
                        "reason_categories_controlled",
                        "reason_categories_free",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["monetary_info"],
        "additionalProperties": False,
    },
}

SYSTEM_RULES = """You are an expert in extracting structured monetary data from classical Arabic texts.
- Return an object with key "monetary_info": [...]
- Use ONLY phrases present in the text for exact_phrase_in_text_Arabic.
- If something is missing, output "null".
- Categories must be relevant and consistent.


GENERAL INSTRUCTIONS
1. Output format: Always return a list of JSON objects, one per occurrence of monetary information.
2. Extraction rules:
  - Catch every mention of monetary units (Dinar, Dirham, Fals, etc.) as a separate object.
  - Stick to the text for extracting information. Extract only if the evidence is explicitly present in the provided text.
    Do not make up answers, search in other sources, or guess. If information is missing, or you do not know the answer, output "null".
  - Fields ending in "_Arabic": copy the exact Arabic text (no edits).
  - Fields ending in "_in_digits": use English digits as strings.
  - Fields ending in "_controlled": select only from the relevant CONTROLLED LISTs below.
  - Fields ending in "_free": propose items irregardless of the CONTROLLED LISTs.
  - All other fields: use English in lowercase.
3. Field definitions:
  - exact_phrase_in_text_Arabic: Repeat the full Arabic sentence in the text containing the monetary information in verbatim (no paraphrase, no added/removed punctuation or diacritics not present in the source).
  - amount_of_money_in_digits: Numeric value of the amount of money as string.
  - currency_Arabic: Currency name in Arabic, including adjectives (e.g., "درهم ناصرية").
  - currency: Currency name in English.
  - year_of_transaction_Arabic: Year of transaction in Arabic.
  - year_of_transaction_in_digits: Numeric value of the year of transaction as string.
  - other_years_Arabic: List of all other years mentioned in the text in Arabic.
  - other_years_in_digits: List of numeric values of all other years mentioned in the text as string.
  - location_of_transaction_Arabic: The city in which the transaction has taken place. If the city is not mentioned, the name of the country or the historical region (like Sham) is also fine.
  - amount_of_item_in_transaction_Arabic: Amount of item in transaction.
  - payer_or_owner_Arabic: Name of the payer or the owner of the money.
  - payer_or_owner_categories_controlled: List of categories for the job, role, or profession of the payer/owner of the money, assigned from the CONTROLLED LIST below.
  - payer_or_owner_categories_free: Propose a list containing one to three categories for the job, role, or profession of the payer/owner of the money, irregardless of the CONTROLLED LIST.
  - payee_Arabic: Name of the payee.
  - payee_categories_controlled: List of categories for the job, role, or profession of the payee, assigned from the CONTROLLED LIST below.
  - payee_categories_free: Propose a list containing one to three categories for the job, role, or profession of the payee, irregardless of the CONTROLLED LIST.
  - money_paid_for_Arabic: Reason of payment copied from the text.
  - reason_of_transaction: Purpose of the monetary transaction.
  - reason_categories_controlled: List of categories for the reason of transaction, assigned from the CONTROLLED LIST below.
  - reason_categories_free: Propose a list containing one to three categories for the reason of transaction, irregardless of the CONTROLLED LIST.

CATEGORIES

Payer/Owner and payee Roles
- payer_or_owner__categories and payee_categories: job, role, or profession of the payer/owner and payee.
- CONTROLLED LIST:
  - scholar → all types of scholars (religious and non-religious)
  - companion → anyone connected to the Prophet
  - governor → provincial/state rulers, not caliphs or kings
  - judge → all legal judges (qāḍī, etc.)
  - administrative_official → viziers, scribes, tax collectors, bureaucrats
  - public_service_worker → non-military staff providing state services
  - military_personnel → soldiers, warriors, commanders
  - merchant → traders of any scale
  - ascetic → renunciants, mystics living without wealth, Sufis
  - captive → prisoners of war or hostages
  - performer → performers of music/entertainment
  - medical_staff → physicians, healers, surgeons
  - poor → all types of poor people
  - caliph, king, farmer, shopkeeper, poet, slave, scientist → as commonly understood
  - other: use only if none of the above fits.



Reasons for Transaction
- reason_categories: purpose of the monetary transaction.
- CONTROLLED LIST:
  - debt → borrowed/lent money, loans, deposits
  - remuneration → salaries, allowances, rewards, gifts
  - earnings → income from selling, renting, trading
  - wealth → personal assets or stored wealth
  - tribute → money given by one municipal leader or governor to another as submission
  - tax → all formal taxes, including zakat and jizya
  - property → houses, land (including iqta), estates
  - bribe → illicit payments
  - theft → looting, extortion, booty
  - military_expenses → costs tied to war, like weapon or sending troops
  - education_expenses → costs paid to teachers and students for teaching and learning
  - medical_expenses → costs paid for medial services
  - agricultural_expenses → costs tied to agriculture, like gardening
  - living_expenses → the amount of money a person needs to spend a day/month/year
  - party_expenses → costs of a party or a ceremony
  - position → jobs or appointments purchased/assigned
  - charity → voluntary giving, donations, sadaqa
  - waqf → endowments specifically designated as waqf
  - exchange_rate → mention of currency exchange value
  - performance → payment for musical, theatrical, or artistic performances
  - luxury → jewelry, fine goods, expensive items
  - book → manuscripts, copied works, literary purchases
  - treasury → money inside or flowed inside/outside of the treasury
  - settlement → political, legal, or other types of settlements between two individuals/groups
  - price → mention of the price of an item (when this label is used, another label identifying the item whose price is mentioned should also be included)
  - food, animal_feed, grain, animal, clothing, furniture, sweets, oil, dairy, liquid, fruit, dried_fruit, meat, fabric, slave, dowry, travel, transport,
    construction, confiscation, public_service, poetry, trade, inheritance, ransom, penalty → straightforward items
  - other: use only if none of the above fits.

CONSISTENCY RULES
- Always use lowercase for labels.
- Use snake case for compound labels (like military_personnel)
- Always choose the most precise categories possible.
- Maintain strict consistency across all extracted JSON objects.
- Each JSON object must be independent, complete, and schema-compliant.
"""


# -----------------------
# Globals (set from CLI args)
# -----------------------
MODEL: str = "gpt-5.1"
COMPLETION_WINDOW: str = "24h"
POLL_SECONDS: int = 15

WORK_DIR: Path = Path("batch_work")
INPUT_DIR: Path = WORK_DIR / "inputs"
OUTPUT_DIR: Path = WORK_DIR / "outputs"

MAX_RECORDS_PER_CHUNK: int = 2000
MAX_BYTES_PER_CHUNK: int = 90 * 1024 * 1024  # 90MB

REQUIRE_NONEMPTY: bool = True
SKIP_ALREADY_ANNOTATED: bool = True
STREAMING_MERGE: bool = True

RETRY_ON_STATUSES: Tuple[str, ...] = ("failed", "cancelled", "expired")
MAX_RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_BASE_SECONDS: int = 120
RETRY_BACKOFF_MAX_SECONDS: int = 3600

MIU_JSON_PATH: Path = Path("sample_MIU_corpus.json")
FEW_SHOT_PATH: Path = Path("few_shot_examples.json")

# Loaded at runtime
client: OpenAI
miu_data: Dict[str, Dict[str, Any]] = {}
few_shot: List[Dict[str, Any]] = []


def get_client() -> OpenAI:
    """Create and return an authenticated OpenAI client using OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


# -----------------------
# Helpers (manifest-driven)
# -----------------------

def sha256_text(s: str) -> str:
    """Compute a SHA-256 hex digest for a given text string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    """Create required working directories (inputs/outputs) if missing."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (YYYY-MM-DDTHH:MM:SSZ)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def iter_mius_sorted(data: Dict[str, Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Iterate over MIUs in deterministic (sorted) order."""
    for miu_id in sorted(data.keys()):
        yield miu_id, data[miu_id]


def build_messages(miu_text: str, few_shot: List[Dict[str, Any]], system_rules: str) -> List[Dict[str, str]]:
    """Build chat-style messages for a single MIU extraction request."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_rules}]
    for ex in few_shot:
        msgs.append({"role": "user", "content": f"Arabic text:\n{ex['text']}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False, indent=2)})
    msgs.append({"role": "user", "content": f"Arabic text:\n{miu_text}"})
    return msgs


def batch_record(miu_id: str, miu_text: str) -> Dict[str, Any]:
    """Create one Batch JSONL request record for a single MIU."""
    return {
        "custom_id": miu_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": build_messages(miu_text, few_shot, SYSTEM_RULES),
            "text": {"format": RESPONSE_SCHEMA},
        },
    }


def load_manifest() -> Dict[str, Any]:
    """Load manifest.json if present; otherwise return a new empty manifest dict."""
    path = WORK_DIR / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Write the given manifest dict to work_dir/manifest.json."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def init_manifest_if_needed() -> Dict[str, Any]:
    """Create a new manifest.json with configuration fingerprints if it does not already exist."""
    ensure_dirs()
    manifest = load_manifest()
    if manifest:
        return manifest

    manifest = {
        "created_utc": utc_now_iso(),
        "model": MODEL,
        "completion_window": COMPLETION_WINDOW,
        "miu_json_path": str(MIU_JSON_PATH),
        "few_shot_path": str(FEW_SHOT_PATH),
        "system_rules_sha256": sha256_text(SYSTEM_RULES),
        "schema_sha256": sha256_text(json.dumps(RESPONSE_SCHEMA, sort_keys=True)),
        "chunks": [],
    }
    save_manifest(manifest)
    return manifest


def chunk_input_paths() -> List[Path]:
    """Return a sorted list of existing chunk JSONL input paths in INPUT_DIR."""
    if not INPUT_DIR.exists():
        return []
    return sorted(INPUT_DIR.glob("batch_part_*.jsonl"))


def _new_chunk_path(part_idx: int) -> Path:
    """Construct the path for a new input chunk JSONL (batch_part_XXXX.jsonl)."""
    return INPUT_DIR / f"batch_part_{part_idx:04d}.jsonl"


def build_chunks_from_corpus(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Create deterministic chunk JSONLs from the corpus and register them in the manifest."""
    ensure_dirs()

    existing = {p.name: p for p in chunk_input_paths()}
    existing_names = set(existing.keys())
    manifest_names = {Path(ch["input_path"]).name for ch in manifest.get("chunks", [])}

    if existing_names and existing_names == manifest_names:
        print(f"Chunks already exist ({len(existing_names)}). Skipping chunk build.")
        return manifest

    if existing_names:
        print("Found chunk files on disk; syncing manifest without rewriting files.")
        for name in sorted(existing_names):
            if name not in manifest_names:
                manifest.setdefault("chunks", []).append(
                    {
                        "chunk_name": name.replace(".jsonl", ""),
                        "input_path": str(existing[name]),
                        "output_path": None,
                        "batch_id": None,
                        "batch_id_history": [],
                        "status": "not_submitted",
                        "attempt": 0,
                        "submitted_utc": None,
                        "completed_utc": None,
                        "next_retry_utc": None,
                        "last_error": None,
                    }
                )
        save_manifest(manifest)
        return manifest

    part_idx = 0
    record_count = 0
    byte_count = 0
    out_path = _new_chunk_path(part_idx)
    out_f = out_path.open("w", encoding="utf-8")

    def _close_and_register(path: Path, n_records: int) -> None:
        """Close current file and register it in the manifest."""
        manifest.setdefault("chunks", []).append(
            {
                "chunk_name": path.stem,
                "input_path": str(path),
                "output_path": None,
                "batch_id": None,
                "batch_id_history": [],
                "status": "not_submitted",
                "attempt": 0,
                "records": n_records,
                "created_utc": utc_now_iso(),
                "submitted_utc": None,
                "completed_utc": None,
                "next_retry_utc": None,
                "last_error": None,
            }
        )

    current_records = 0

    for miu_id, rec in iter_mius_sorted(miu_data):
        if SKIP_ALREADY_ANNOTATED and isinstance(rec.get("ann"), dict) and "monetary_info" in rec.get("ann", {}):
            continue

        miu_text = rec.get("txt") or ""
        line_obj = batch_record(miu_id, miu_text)
        line = json.dumps(line_obj, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if current_records > 0 and (current_records >= MAX_RECORDS_PER_CHUNK or (byte_count + line_bytes) > MAX_BYTES_PER_CHUNK):
            out_f.close()
            _close_and_register(out_path, current_records)

            part_idx += 1
            out_path = _new_chunk_path(part_idx)
            out_f = out_path.open("w", encoding="utf-8")
            current_records = 0
            byte_count = 0

        out_f.write(line)
        current_records += 1
        byte_count += line_bytes
        record_count += 1

    out_f.close()
    if current_records > 0:
        _close_and_register(out_path, current_records)

    save_manifest(manifest)
    print(f"Wrote {len(manifest.get('chunks', []))} chunks (total requests: {record_count}).")
    return manifest


# -----------------------
# Batch API
# -----------------------

def submit_batch(jsonl_path: Path, completion_window: Optional[str] = None) -> str:
    """Upload a chunk JSONL and create a Batch job, returning the batch_id."""
    completion_window = completion_window or COMPLETION_WINDOW
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    return job.id


def retrieve_batch(batch_id: str) -> Any:
    """Retrieve the latest Batch job object from the API."""
    return client.batches.retrieve(batch_id)


def fetch_batch_output(batch_id: str, out_jsonl: Path) -> None:
    """Download a completed Batch output file and save it as JSONL."""
    job = retrieve_batch(batch_id)
    if job.status != "completed":
        raise RuntimeError(f"Batch not completed (status={job.status}).")
    if not job.output_file_id:
        raise RuntimeError("No output_file_id found on the completed batch job.")
    text = client.files.content(job.output_file_id).text
    out_jsonl.write_text(text, encoding="utf-8")


# -----------------------
# Retry logic
# -----------------------

def _seconds_until_retry(attempt: int) -> int:
    """Compute exponential backoff delay (seconds) for a given attempt number."""
    base = RETRY_BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1))
    jitter = random.randint(0, min(30, max(1, base)))
    return min(RETRY_BACKOFF_MAX_SECONDS, base + jitter)


def mark_terminal_status(manifest: Dict[str, Any], ch: Dict[str, Any], status: str, job: Any = None) -> None:
    """Record terminal job status and store a short error summary in the chunk entry."""
    ch["status"] = status
    ch["completed_utc"] = utc_now_iso()
    if job and getattr(job, "errors", None):
        ch["last_error"] = {"kind": "batch_errors", "errors": job.errors}
    else:
        ch.setdefault("last_error", None)


def eligible_for_retry(ch: Dict[str, Any]) -> bool:
    """Return True if a chunk should be retried under the current retry policy."""
    if ch.get("status") not in RETRY_ON_STATUSES:
        return False
    attempt = int(ch.get("attempt") or 0)
    if attempt >= MAX_RETRY_ATTEMPTS:
        return False

    next_retry_utc = ch.get("next_retry_utc")
    if next_retry_utc:
        try:
            t = time.strptime(next_retry_utc, "%Y-%m-%dT%H:%M:%SZ")
            # NOTE: mktime assumes local time; good enough for gating retries.
            next_epoch = int(time.mktime(t))
            if int(time.time()) < next_epoch:
                return False
        except Exception:
            pass

    return True


def retry_failed_chunks(manifest: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Resubmit chunks that failed/cancelled/expired, respecting max attempts and backoff."""
    retried = 0

    for ch in manifest.get("chunks", []):
        if not eligible_for_retry(ch):
            continue

        input_path = Path(ch["input_path"])
        if not input_path.exists():
            ch["last_error"] = {"kind": "missing_input_path", "path": str(input_path)}
            continue

        prev = ch.get("batch_id")
        if prev:
            ch.setdefault("batch_id_history", []).append(prev)

        new_batch_id = submit_batch(input_path, completion_window=COMPLETION_WINDOW)
        ch["batch_id"] = new_batch_id
        ch["attempt"] = int(ch.get("attempt") or 0) + 1
        ch["status"] = "submitted"
        ch["submitted_utc"] = utc_now_iso()
        ch["completed_utc"] = None
        ch["output_path"] = None
        ch["last_error"] = None

        delay = _seconds_until_retry(ch["attempt"])
        ch["next_retry_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + delay))
        retried += 1
        print(
            f"Retried {ch['chunk_name']} (attempt={ch['attempt']}) -> batch_id={new_batch_id}; "
            f"next_retry_utc={ch['next_retry_utc']}"
        )

    if retried:
        save_manifest(manifest)
    return manifest, retried


# -----------------------
# Output parsing + merge
# -----------------------

def _find_first_json(text: str) -> Optional[str]:
    """Extract the first top-level JSON object substring found in text, if any."""
    start = text.find("{")
    if start == -1:
        return None
    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start : i + 1]
    return None


def iter_jsonl(path: Path) -> Iterable[Tuple[int, str]]:
    """Yield (line_number, line_text) for a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield i, line


def parse_output_line(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse one Batch output JSON line into (miu_id, parsed_json, error_dict)."""
    miu_id = obj.get("custom_id")
    resp = obj.get("response") or {}
    status_code = resp.get("status_code")
    body = resp.get("body") or {}

    if status_code != 200:
        return miu_id, None, {"kind": "http_error", "status_code": status_code, "body": body}

    text_out: Optional[str] = None
    if isinstance(body, dict):
        text_out = body.get("output_text")
        if not text_out and "output" in body and isinstance(body["output"], list):
            parts: List[str] = []
            for item in body["output"]:
                if isinstance(item, dict) and item.get("type") == "message":
                    content = item.get("content") or []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            parts.append(c.get("text") or "")
            if parts:
                text_out = "\n".join(parts)

    if not text_out:
        return miu_id, None, {"kind": "missing_output_text", "body_keys": list(body.keys()) if isinstance(body, dict) else None}

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(text_out)
    except Exception:
        js = _find_first_json(text_out)
        if js:
            try:
                parsed = json.loads(js)
            except Exception as e2:
                return miu_id, None, {"kind": "json_parse_error", "error": str(e2), "text_preview": text_out[:500]}
        else:
            return miu_id, None, {"kind": "json_not_found", "text_preview": text_out[:500]}

    try:
        jsonschema_validate(instance=parsed, schema=RESPONSE_SCHEMA["schema"])
    except ValidationError as ve:
        return miu_id, None, {"kind": "schema_validation_error", "error": str(ve), "parsed_preview": str(parsed)[:500]}

    return miu_id, parsed, None


@dataclass
class ParsedBatch:
    """Container for parsed batch results and errors."""
    results_by_id: Dict[str, Dict[str, Any]]
    errors_by_id: Dict[str, Dict[str, Any]]


def parse_output_jsonl(out_path: Path) -> ParsedBatch:
    """Parse a per-chunk output JSONL file into results and errors keyed by custom_id."""
    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, Dict[str, Any]] = {}
    for line_no, line in iter_jsonl(out_path):
        try:
            obj = json.loads(line)
        except Exception as e:
            errors[f"{out_path.name}::__line_{line_no}__"] = {"kind": "bad_jsonl_line", "error": str(e), "line": line[:500]}
            continue

        miu_id, parsed, err = parse_output_line(obj)
        if miu_id is None:
            errors[f"{out_path.name}::__line_{line_no}__"] = err or {"kind": "unknown_parse_error"}
            continue
        if err:
            errors[miu_id] = {"file": out_path.name, **err}
            continue
        results[miu_id] = parsed

    return ParsedBatch(results_by_id=results, errors_by_id=errors)


def merge_one_result_into_mius(
    miu_data: Dict[str, Dict[str, Any]],
    miu_id: str,
    parsed: Dict[str, Any],
    require_nonempty: bool = True,
) -> bool:
    """Merge one validated parsed result into miu_data[miu_id]['ann']['monetary_info']."""
    if miu_id not in miu_data:
        return False
    info = parsed.get("monetary_info")
    if require_nonempty and (not info):
        return False
    rec = miu_data[miu_id]
    ann = rec.get("ann")
    if not isinstance(ann, dict):
        ann = {}
        rec["ann"] = ann
    ann["monetary_info"] = info
    return True


def stream_merge_outputs_into_mius(
    miu_data: Dict[str, Dict[str, Any]],
    output_paths: List[Path],
    require_nonempty: bool = True,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """Streaming parse and merge of many per-chunk outputs into the corpus (memory-efficient)."""
    errors_by_id: Dict[str, Any] = {}
    empty_hits: List[str] = []
    stats: Dict[str, Any] = {"lines": 0, "merged": 0, "empty": 0, "errors": 0}

    for p in output_paths:
        for line_no, line in iter_jsonl(p):
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                stats["errors"] += 1
                errors_by_id[f"{p.name}::__line_{line_no}__"] = {"kind": "bad_jsonl_line", "error": str(e), "line": line[:500]}
                continue

            miu_id, parsed, err = parse_output_line(obj)
            if miu_id is None:
                stats["errors"] += 1
                errors_by_id[f"{p.name}::__line_{line_no}__"] = err or {"kind": "unknown_parse_error"}
                continue
            if err:
                stats["errors"] += 1
                errors_by_id[miu_id] = {"file": p.name, **err}
                continue

            merged = merge_one_result_into_mius(miu_data, miu_id, parsed, require_nonempty=require_nonempty)
            if merged:
                stats["merged"] += 1
            else:
                stats["empty"] += 1
                empty_hits.append(miu_id)

    return errors_by_id, empty_hits, stats


# -----------------------
# Pipeline steps
# -----------------------

def step_build() -> None:
    """Build deterministic chunk JSONLs and initialize manifest."""
    init_manifest_if_needed()
    manifest = load_manifest()
    build_chunks_from_corpus(manifest)


def step_submit() -> None:
    """Submit any not-submitted chunks as Batch jobs and record batch_ids in the manifest."""
    manifest = load_manifest()
    changed = False

    for ch in manifest.get("chunks", []):
        if ch.get("status") != "not_submitted":
            continue
        input_path = Path(ch["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input chunk file: {input_path}")

        print(f"Submitting {input_path.name} ...")
        batch_id = submit_batch(input_path, completion_window=COMPLETION_WINDOW)
        ch["batch_id"] = batch_id
        ch["status"] = "submitted"
        ch["submitted_utc"] = utc_now_iso()
        changed = True
        print("  batch_id =", batch_id)

    if changed:
        save_manifest(manifest)
        print("Updated manifest with submitted batch_ids.")
    else:
        print("No chunks needed submission.")


def step_poll(once: bool = False) -> None:
    """Poll Batch statuses and update manifest; optionally run only one polling pass."""
    while True:
        manifest = load_manifest()
        changed = False
        active = 0

        for ch in manifest.get("chunks", []):
            batch_id = ch.get("batch_id")
            if not batch_id:
                continue
            if ch.get("status") in ("completed", "failed", "cancelled", "expired"):
                continue

            active += 1
            job = retrieve_batch(batch_id)
            status = job.status
            print(f"{ch['chunk_name']}: {status}")

            if status != ch.get("status"):
                ch["status"] = status
                changed = True
                if status == "completed":
                    ch["completed_utc"] = utc_now_iso()

        if changed:
            save_manifest(manifest)
            print("Manifest updated.")

        if once or active == 0:
            break

        time.sleep(POLL_SECONDS)


def step_retry() -> None:
    """Retry chunks in terminal failure states, respecting max attempts and backoff."""
    manifest = load_manifest()

    changed = False
    for ch in manifest.get("chunks", []):
        batch_id = ch.get("batch_id")
        if not batch_id:
            continue
        if ch.get("status") not in RETRY_ON_STATUSES:
            continue
        try:
            job = retrieve_batch(batch_id)
            if job.status != ch.get("status"):
                ch["status"] = job.status
                changed = True
            if job.status in RETRY_ON_STATUSES:
                mark_terminal_status(manifest, ch, job.status, job)
                changed = True
        except Exception as e:
            ch["last_error"] = {"kind": "retrieve_failed", "error": str(e)}
            changed = True

    if changed:
        save_manifest(manifest)

    _, retried = retry_failed_chunks(manifest)
    print("Retried chunks:", retried)


def step_download() -> None:
    """Download outputs for completed chunks into outputs/ and record paths in manifest."""
    manifest = load_manifest()
    changed = False
    ensure_dirs()

    for ch in manifest.get("chunks", []):
        if ch.get("status") != "completed":
            continue

        out_path = ch.get("output_path")
        if out_path and Path(out_path).exists():
            continue

        batch_id = ch.get("batch_id")
        if not batch_id:
            continue

        target = OUTPUT_DIR / f"{ch['chunk_name']}.out.jsonl"
        print(f"Downloading output for {ch['chunk_name']} -> {target.name}")
        fetch_batch_output(batch_id, target)
        ch["output_path"] = str(target)
        changed = True

    if changed:
        save_manifest(manifest)
        print("Manifest updated with output paths.")
    else:
        print("No outputs needed downloading.")


def step_merge() -> None:
    """Parse completed outputs, merge into MIU JSON, and write merged + logs to work_dir/."""
    manifest = load_manifest()

    completed = [
        ch for ch in manifest.get("chunks", [])
        if ch.get("status") == "completed" and ch.get("output_path") and Path(ch["output_path"]).exists()
    ]
    print(f"Completed chunks with outputs: {len(completed)}")

    output_paths = [Path(ch["output_path"]) for ch in completed]

    if STREAMING_MERGE:
        errors_by_id, empty_hits, stats = stream_merge_outputs_into_mius(
            miu_data,
            output_paths,
            require_nonempty=REQUIRE_NONEMPTY,
        )
    else:
        all_results: Dict[str, Dict[str, Any]] = {}
        all_errors: Dict[str, Dict[str, Any]] = {}
        for ch in completed:
            out_path = Path(ch["output_path"])
            parsed = parse_output_jsonl(out_path)
            all_results.update(parsed.results_by_id)
            for k, v in parsed.errors_by_id.items():
                all_errors[k] = {"chunk": ch["chunk_name"], **v}

        empty_hits = []
        for miu_id, result in all_results.items():
            merged = merge_one_result_into_mius(miu_data, miu_id, result, require_nonempty=REQUIRE_NONEMPTY)
            if not merged:
                empty_hits.append(miu_id)

        errors_by_id = all_errors
        stats = {"lines": len(all_results) + len(all_errors), "merged": len(all_results), "empty": len(empty_hits), "errors": len(all_errors)}

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    merged_path = WORK_DIR / "merged_miu.json"
    merged_path.write_text(json.dumps(miu_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", merged_path)

    errors_path = WORK_DIR / "errors.json"
    errors_path.write_text(json.dumps(errors_by_id, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", errors_path, f"(errors: {len(errors_by_id):,})")

    empty_path = WORK_DIR / "empty_hits.json"
    empty_path.write_text(json.dumps(empty_hits, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", empty_path, f"(empty hits: {len(empty_hits):,})")

    stats_path = WORK_DIR / "merge_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", stats_path)
    print("Stats:", stats)


def step_run() -> None:
    """Run the full pipeline end-to-end (build → submit → poll/retry loop → download → merge)."""
    step_build()
    step_submit()

    while True:
        step_poll(once=True)

        manifest = load_manifest()
        active = [
            ch for ch in manifest.get("chunks", [])
            if ch.get("batch_id") and ch.get("status") not in ("completed", "failed", "cancelled", "expired")
        ]
        if active:
            time.sleep(POLL_SECONDS)
            continue

        if any(eligible_for_retry(ch) for ch in manifest.get("chunks", [])):
            step_retry()
            time.sleep(1)
            continue

        break

    step_download()
    step_merge()


# -----------------------
# CLI
# -----------------------

def _apply_args_to_globals(args: argparse.Namespace) -> None:
    """Apply argparse settings to module-level configuration globals."""
    global MODEL, COMPLETION_WINDOW, POLL_SECONDS
    global WORK_DIR, INPUT_DIR, OUTPUT_DIR
    global MAX_RECORDS_PER_CHUNK, MAX_BYTES_PER_CHUNK
    global REQUIRE_NONEMPTY, SKIP_ALREADY_ANNOTATED, STREAMING_MERGE
    global MAX_RETRY_ATTEMPTS, RETRY_BACKOFF_BASE_SECONDS, RETRY_BACKOFF_MAX_SECONDS
    global MIU_JSON_PATH, FEW_SHOT_PATH

    MODEL = args.model
    COMPLETION_WINDOW = args.completion_window
    POLL_SECONDS = args.poll_seconds

    WORK_DIR = Path(args.work_dir)
    INPUT_DIR = WORK_DIR / "inputs"
    OUTPUT_DIR = WORK_DIR / "outputs"

    MAX_RECORDS_PER_CHUNK = args.max_records_per_chunk
    MAX_BYTES_PER_CHUNK = args.max_bytes_per_chunk

    REQUIRE_NONEMPTY = not args.allow_empty_merges
    SKIP_ALREADY_ANNOTATED = not args.no_skip_annotated
    STREAMING_MERGE = not args.no_streaming_merge

    MAX_RETRY_ATTEMPTS = args.max_retry_attempts
    RETRY_BACKOFF_BASE_SECONDS = args.retry_backoff_base_seconds
    RETRY_BACKOFF_MAX_SECONDS = args.retry_backoff_max_seconds

    MIU_JSON_PATH = Path(args.miu_json)
    FEW_SHOT_PATH = Path(args.few_shot)


def _load_inputs() -> None:
    """Load MIU corpus and few-shot examples into memory."""
    global miu_data, few_shot
    miu_data = json.loads(MIU_JSON_PATH.read_text(encoding="utf-8"))
    few_shot = json.loads(FEW_SHOT_PATH.read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Manifest-driven OpenAI Batch pipeline for extracting monetary information from Arabic MIUs."
    )

    parser.add_argument("--miu-json", default=str(MIU_JSON_PATH), help="Path to corpus JSON (miu_id -> record with 'txt').")
    parser.add_argument("--few-shot", default=str(FEW_SHOT_PATH), help="Path to few-shot examples JSON (list of {text, output}).")
    parser.add_argument("--work-dir", default=str(WORK_DIR), help="Working directory for manifest/inputs/outputs.")
    parser.add_argument("--model", default=MODEL, help="Model name for Responses API.")
    parser.add_argument("--completion-window", default=COMPLETION_WINDOW, help="Batch completion window (e.g., 24h).")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS, help="Seconds between polling passes.")

    parser.add_argument("--max-records-per-chunk", type=int, default=MAX_RECORDS_PER_CHUNK, help="Max MIUs per chunk JSONL.")
    parser.add_argument("--max-bytes-per-chunk", type=int, default=MAX_BYTES_PER_CHUNK, help="Max bytes per chunk JSONL.")

    parser.add_argument("--allow-empty-merges", action="store_true", help="Merge empty monetary_info arrays (default: no).")
    parser.add_argument("--no-skip-annotated", action="store_true", help="Do not skip MIUs already having ann.monetary_info.")
    parser.add_argument("--no-streaming-merge", action="store_true", help="Disable streaming merge (uses in-memory merge).")

    parser.add_argument("--max-retry-attempts", type=int, default=MAX_RETRY_ATTEMPTS, help="Max retry submissions per chunk.")
    parser.add_argument("--retry-backoff-base-seconds", type=int, default=RETRY_BACKOFF_BASE_SECONDS, help="Base seconds for exponential backoff.")
    parser.add_argument("--retry-backoff-max-seconds", type=int, default=RETRY_BACKOFF_MAX_SECONDS, help="Maximum backoff seconds.")

    sub = parser.add_subparsers(dest="cmd", required=False)
    sub.add_parser("build", help="Build chunk JSONLs and initialize manifest.")
    sub.add_parser("submit", help="Submit not-submitted chunks.")
    sub.add_parser("poll", help="Poll job statuses once and update manifest.")
    sub.add_parser("retry", help="Retry failed/cancelled/expired chunks if eligible.")
    sub.add_parser("download", help="Download outputs for completed chunks.")
    sub.add_parser("merge", help="Parse outputs and merge into merged_miu.json + logs.")
    sub.add_parser("run", help="Run the full pipeline end-to-end (default).")

    args = parser.parse_args(argv)
    if not args.cmd:
        args.cmd = "run"

    _apply_args_to_globals(args)

    global client
    client = get_client()
    _load_inputs()

    if args.cmd == "build":
        step_build()
    elif args.cmd == "submit":
        step_submit()
    elif args.cmd == "poll":
        step_poll(once=True)
    elif args.cmd == "retry":
        step_retry()
    elif args.cmd == "download":
        step_download()
    elif args.cmd == "merge":
        step_merge()
    elif args.cmd == "run":
        step_run()
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
