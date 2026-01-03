#!/usr/bin/env python3
"""
Post-run leakage audit for pre-retrieved evidence.

This answers: "how much are we cheating?" for POC-A.

What we check (offline, heuristic):
- Evidence snippets contain QA-format artifacts (Option A:, Answer:, etc.)
- Evidence snippets contain the gold answer text
- Evidence snippets contain any option text

Optionally:
- Restrict audit to indices present in a MedAgents output JSONL (which includes "idx").
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


def _artifact_hit_names(text: str) -> list[str]:
    hits: list[str] = []
    for name, pat in QA_ARTIFACT_PATTERNS:
        if pat.search(text or ""):
            hits.append(name)
    return hits


@dataclass(frozen=True)
class LeakSignals:
    has_artifact: bool
    has_gold_answer_text: bool
    has_any_option_text: bool


def _leak_signals_for_snippets(snips: list[str], row: dict[str, Any]) -> LeakSignals:
    ans = row.get("answer")
    ansn = _norm(ans) if isinstance(ans, str) else ""
    opt_texts = list(_iter_option_texts(row))

    has_artifact = False
    has_gold = False
    has_opt = False

    for s in snips:
        if not isinstance(s, str):
            continue
        if _artifact_hit_names(s):
            has_artifact = True
        if ansn and len(ansn) >= 8 and ansn.lower() in s.lower():
            has_gold = True
        if not has_opt:
            for opt in opt_texts:
                if opt.lower() in s.lower():
                    has_opt = True
                    break

    return LeakSignals(
        has_artifact=has_artifact,
        has_gold_answer_text=has_gold,
        has_any_option_text=has_opt,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_jsonl",
        type=Path,
        default=Path("vendor/med_agents/datasets/MedQA/test.jsonl"),
    )
    p.add_argument(
        "--evidence_json",
        type=Path,
        default=Path("data/retrieved_med_qa_test.json"),
    )
    p.add_argument(
        "--pred_jsonl",
        type=Path,
        default=None,
        help="Optional MedAgents output JSONL; if provided, audit only those idxs.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Check only first K evidence snippets per record.",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Print up to N example idxs per leak type.",
    )
    args = p.parse_args()

    dataset = _load_jsonl(args.dataset_jsonl)
    evidence = _load_evidence_json(args.evidence_json)
    if len(dataset) != len(evidence):
        print(
            f"[warn] length mismatch: dataset={len(dataset)} evidence={len(evidence)} (audit will use min length)"
        )

    idxs: list[int] = []
    pred_by_idx: dict[int, dict[str, Any]] = {}
    if args.pred_jsonl is not None:
        preds = _load_jsonl(args.pred_jsonl)
        for rec in preds:
            idx = rec.get("idx")
            if isinstance(idx, int):
                idxs.append(idx)
                pred_by_idx[idx] = rec
        idxs = sorted(set(idxs))
        if not idxs:
            raise RuntimeError(f"No integer 'idx' found in {args.pred_jsonl}")
        print(f"[scope] auditing n={len(idxs)} indices from pred_jsonl")
    else:
        idxs = list(range(min(len(dataset), len(evidence))))
        print(f"[scope] auditing full overlap n={len(idxs)}")

    counts = Counter()
    examples: dict[str, list[int]] = {
        "artifact": [],
        "gold_answer_text": [],
        "any_option_text": [],
    }

    for idx in idxs:
        if idx < 0 or idx >= len(dataset) or idx >= len(evidence):
            continue

        row = dataset[idx]
        ev = evidence[idx]
        snips = ev.get("evidence", [])
        if not isinstance(snips, list):
            snips = []
        snips = [s for s in snips[: args.topk] if isinstance(s, str)]

        sig = _leak_signals_for_snippets(snips, row)
        counts["n"] += 1

        if sig.has_artifact:
            counts["artifact"] += 1
            if len(examples["artifact"]) < args.max_examples:
                examples["artifact"].append(idx)
        if sig.has_gold_answer_text:
            counts["gold_answer_text"] += 1
            if len(examples["gold_answer_text"]) < args.max_examples:
                examples["gold_answer_text"].append(idx)
        if sig.has_any_option_text:
            counts["any_option_text"] += 1
            if len(examples["any_option_text"]) < args.max_examples:
                examples["any_option_text"].append(idx)

    n = counts["n"]
    if not n:
        print("[done] nothing to audit")
        return 0

    def _rate(k: str) -> float:
        return float(counts[k]) / float(n)

    print("[summary]")
    print(f"  n={n} topk={args.topk}")
    print(f"  artifact_rate={_rate('artifact'):.4f} ({counts['artifact']}/{n})")
    print(
        f"  gold_answer_text_rate={_rate('gold_answer_text'):.4f} ({counts['gold_answer_text']}/{n})"
    )
    print(
        f"  any_option_text_rate={_rate('any_option_text'):.4f} ({counts['any_option_text']}/{n})"
    )

    print("[examples]")
    for k in ["artifact", "gold_answer_text", "any_option_text"]:
        print(f"  {k}: {examples[k]}")

    # Extra warning: many evidence dumps include QA-formatted options inside instances.input by construction.
    # We should never inject instances.input, only evidence snippets.
    if idxs:
        ex0 = evidence[idxs[0]]
        inst = ex0.get("instances")
        inst_in = ""
        if isinstance(inst, dict) and isinstance(inst.get("input"), str):
            inst_in = inst["input"]
        inst_hits = _artifact_hit_names(inst_in) if inst_in else []
        if inst_hits:
            print(
                "[warn] evidence.instances.input contains QA artifacts (expected for some dumps)."
            )
            print(
                "       Do NOT inject instances.input into prompts; only use evidence[]."
            )
            print(f"       example_hits={sorted(set(inst_hits))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
