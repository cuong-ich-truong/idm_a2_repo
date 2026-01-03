#!/usr/bin/env python3
"""
Pre-inference leakage filter for pre-retrieved evidence.

Goal (POC-A):
- Remove obvious QA-formatted artifacts from `evidence[]` snippets before we ever inject them into prompts.
- Keep it deterministic, offline, and auditable.

Notes:
- We do NOT touch `instances.input` (it often contains "Option A:" etc by construction in some evidence dumps).
- We only filter the `evidence` list for each record.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


QA_ARTIFACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("option_label", re.compile(r"\bOption\s*[A-E]\s*:", re.IGNORECASE)),
    ("options_header", re.compile(r"\bOptions?\s*:", re.IGNORECASE)),
    ("answer_header", re.compile(r"\bAnswer\s*:", re.IGNORECASE)),
    ("correct_answer_header", re.compile(r"\bCorrect\s*answer\s*:", re.IGNORECASE)),
    ("explanation_header", re.compile(r"\bExplanation\s*:", re.IGNORECASE)),
    ("paren_choice", re.compile(r"\([A-E]\)")),
    ("choice_line", re.compile(r"^\s*[A-E]\s*[\.\)]\s+", re.IGNORECASE | re.MULTILINE)),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def _load_evidence_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Evidence JSON must be a list; got {type(data)}")
    return data  # type: ignore[return-value]


def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _iter_option_texts(row: dict[str, Any]) -> Iterable[str]:
    opts = row.get("options")
    if not isinstance(opts, dict):
        return []
    out: list[str] = []
    for _, v in opts.items():
        if isinstance(v, str):
            v = _norm(v)
            if len(v) >= 8:
                out.append(v)
    return out


@dataclass(frozen=True)
class FilterConfig:
    mode: str  # "artifact_only" | "strict"
    min_snip_chars: int


def _leak_reasons(snippet: str, row: dict[str, Any], cfg: FilterConfig) -> list[str]:
    s = snippet or ""
    reasons: list[str] = []

    for name, pat in QA_ARTIFACT_PATTERNS:
        if pat.search(s):
            reasons.append(name)

    if cfg.mode == "strict":
        # Strict mode: drop snippets that appear to contain the gold answer text or any option string.
        ans = row.get("answer")
        if isinstance(ans, str):
            ansn = _norm(ans)
            if len(ansn) >= 8 and ansn.lower() in s.lower():
                reasons.append("contains_gold_answer_text")

        for opt in _iter_option_texts(row):
            if opt.lower() in s.lower():
                reasons.append("contains_option_text")
                break

    if len(_norm(s)) < cfg.min_snip_chars:
        reasons.append("too_short")

    return reasons


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_jsonl",
        type=Path,
        default=Path("vendor/med_agents/datasets/MedQA/test.jsonl"),
        help="Used only for strict-mode checks (answer/options text).",
    )
    p.add_argument(
        "--evidence_json",
        type=Path,
        default=Path("data/retrieved_med_qa_test.json"),
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path("data/retrieved_med_qa_test.filtered.json"),
    )
    p.add_argument(
        "--mode",
        choices=["artifact_only", "strict"],
        default="artifact_only",
        help="artifact_only: drop obvious QA-format artifacts only; strict: also drop snippets containing gold/option text.",
    )
    p.add_argument(
        "--disable_filter",
        action="store_true",
        help="If set, do not filter for leakage; just copy evidence through (optionally still applying --topk).",
    )
    p.add_argument(
        "--min_snip_chars",
        type=int,
        default=80,
        help="Drop snippets shorter than this after normalization.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="If set >0, only consider first K evidence snippets per record.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out_json.exists() and not args.overwrite:
        raise RuntimeError(
            f"Refusing to overwrite existing {args.out_json}. Pass --overwrite."
        )

    dataset = _load_jsonl(args.dataset_jsonl)
    evidence = _load_evidence_json(args.evidence_json)
    if len(dataset) != len(evidence):
        print(
            f"[warn] length mismatch: dataset={len(dataset)} evidence={len(evidence)} (filter will use min length)"
        )

    n = min(len(dataset), len(evidence))
    cfg = FilterConfig(mode=args.mode, min_snip_chars=args.min_snip_chars)

    reason_counts: Counter[str] = Counter()
    dropped_total = 0
    kept_total = 0

    out: list[dict[str, Any]] = []
    for i in range(n):
        row = dataset[i]
        ev = dict(evidence[i])  # shallow copy
        snips = ev.get("evidence", [])
        if not isinstance(snips, list):
            snips = []

        snips = [s for s in snips if isinstance(s, str)]
        if args.topk and args.topk > 0:
            snips = snips[: args.topk]

        if args.disable_filter:
            kept = list(snips)
            kept_total += len(kept)
        else:
            kept = []
            for s in snips:
                reasons = _leak_reasons(s, row, cfg)
                if reasons:
                    dropped_total += 1
                    for r in set(reasons):
                        reason_counts[r] += 1
                    continue
                kept.append(s)
                kept_total += 1

        ev["evidence"] = kept
        out.append(ev)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {args.out_json}")
    print(f"[mode] filter={'disabled' if args.disable_filter else cfg.mode}")
    print(f"[stats] kept_snips={kept_total} dropped_snips={dropped_total}")
    if reason_counts:
        print("[drop_reasons]")
        for k, v in reason_counts.most_common():
            print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
