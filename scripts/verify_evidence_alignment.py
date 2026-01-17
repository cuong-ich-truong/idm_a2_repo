#!/usr/bin/env python3
"""
Verify 1-1 alignment between a MedQA-style dataset JSONL (e.g. test.jsonl) and a
precomputed evidence cache JSON (Self-BioRAG-style), by index.

Checks:
  - length match (dataset_n vs evidence_n)
  - for each idx: evidence record exists
  - for each idx: evidence["evidence"] exists and is non-empty (optional strict)
  - optional question-text sanity: dataset question ~= evidence instances.input QUESTION:

Usage:
  python3 scripts/verify_evidence_alignment.py \
    --dataset_jsonl vendor/med_agents/datasets/MedQA/test.jsonl \
    --evidence_json data/retrieved_med_qa_test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


_RE_EXTRACT_QUESTION = re.compile(
    r"QUESTION:\s*(.*?)(?:\nOption\s*[A-E]\s*:|\nOptions?\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00b0", " degrees ")
    s = _RE_NON_ALNUM.sub(" ", s)
    return " ".join(s.split())


def _extract_question_from_instances_input(inst_input: str) -> str:
    m = _RE_EXTRACT_QUESTION.search(inst_input or "")
    if not m:
        return ""
    return (m.group(1) or "").strip().strip('"').strip()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no} of {path}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"Expected JSON object per line; got {type(obj)} on line {line_no} of {path}"
                )
            yield obj


def _load_evidence_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Evidence JSON must be a list; got {type(data)} in {path}")
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(data):
        if isinstance(rec, dict):
            out.append(rec)
        else:
            # keep slot but with empty dict to preserve indexing
            out.append({})
    return out


@dataclass(frozen=True)
class Mismatch:
    idx: int
    ratio: float
    dataset_q: str
    evidence_q: str


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_jsonl",
        type=str,
        default="vendor/med_agents/datasets/MedQA/test.jsonl",
        help="Path to dataset JSONL (index i is treated as qid=i).",
    )
    p.add_argument(
        "--evidence_json",
        type=str,
        default="data/retrieved_med_qa_test.json",
        help="Path to evidence cache JSON (a JSON list aligned by dataset index).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max dataset rows to check (useful for quick spot-check). -1 means all.",
    )
    p.add_argument(
        "--require_nonempty_evidence",
        action="store_true",
        help="If set, treat empty/missing evidence[] as an error condition.",
    )
    p.add_argument(
        "--check_question_text",
        action="store_true",
        help='If set, compare dataset["question"] vs evidence["instances"]["input"] QUESTION: text.',
    )
    p.add_argument(
        "--min_question_ratio",
        type=float,
        default=0.92,
        help="Min normalized similarity ratio to consider questions aligned (only used with --check_question_text).",
    )
    p.add_argument(
        "--report_top",
        type=int,
        default=10,
        help="How many mismatches to print in the JSON summary.",
    )
    args = p.parse_args()

    ds_path = Path(args.dataset_jsonl)
    ev_path = Path(args.evidence_json)
    if not ds_path.exists():
        raise RuntimeError(f"Dataset JSONL not found: {ds_path}")
    if not ev_path.exists():
        raise RuntimeError(f"Evidence JSON not found: {ev_path}")

    dataset = list(_read_jsonl(ds_path))
    evidence = _load_evidence_json(ev_path)

    ds_n = len(dataset)
    ev_n = len(evidence)

    limit = ds_n if args.limit == -1 else min(args.limit, ds_n)

    missing_record = 0
    empty_evidence = 0
    missing_question = 0
    question_mismatches: list[Mismatch] = []

    for idx in range(limit):
        rec = evidence[idx] if idx < ev_n else None
        if not rec:
            missing_record += 1
            continue

        snips = rec.get("evidence", None)
        if (
            not isinstance(snips, list)
            or len([s for s in snips if isinstance(s, str) and s.strip()]) == 0
        ):
            empty_evidence += 1

        if args.check_question_text:
            ds_q = str(dataset[idx].get("question", "") or "")
            inst_input = ""
            inst = rec.get("instances", {})
            if isinstance(inst, dict):
                inst_input = str(inst.get("input", "") or "")
            ev_q = _extract_question_from_instances_input(inst_input)
            if not ds_q.strip() or not ev_q.strip():
                missing_question += 1
            else:
                a = _norm_text(ds_q)
                b = _norm_text(ev_q)
                ratio = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
                if ratio < float(args.min_question_ratio):
                    question_mismatches.append(
                        Mismatch(idx=idx, ratio=ratio, dataset_q=ds_q, evidence_q=ev_q)
                    )

    # Build a summary with a few representative mismatches (worst first)
    question_mismatches.sort(key=lambda m: m.ratio)
    top = question_mismatches[: max(0, int(args.report_top))]

    ok_len = ds_n == ev_n
    ok_missing_record = missing_record == 0
    ok_evidence = (empty_evidence == 0) if args.require_nonempty_evidence else True
    ok_q = (
        (len(question_mismatches) == 0 and missing_question == 0)
        if args.check_question_text
        else True
    )

    summary = {
        "dataset_jsonl": str(ds_path),
        "evidence_json": str(ev_path),
        "dataset_n": ds_n,
        "evidence_n": ev_n,
        "checked_n": limit,
        "length_match": ok_len,
        "missing_evidence_record_count": missing_record,
        "empty_evidence_count": empty_evidence,
        "missing_question_text_count": (
            missing_question if args.check_question_text else None
        ),
        "question_mismatch_count": (
            len(question_mismatches) if args.check_question_text else None
        ),
        "question_mismatch_min_ratio": (
            (top[0].ratio if top else None) if args.check_question_text else None
        ),
        "question_mismatch_samples": (
            [
                {
                    "idx": m.idx,
                    "ratio": round(m.ratio, 4),
                    "dataset_question": m.dataset_q,
                    "evidence_question_extracted": m.evidence_q,
                }
                for m in top
            ]
            if args.check_question_text
            else None
        ),
        "ok": bool(ok_len and ok_missing_record and ok_evidence and ok_q),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Exit code: non-zero if any hard failure conditions
    if not ok_len or not ok_missing_record or not ok_evidence or not ok_q:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
