#!/usr/bin/env python3
"""
Verify that MedAgents MedQA dataset records align with pre-retrieved evidence records.

Checks:
- lengths match
- spot-check selected indices: dataset question should appear in evidence "instances[0].input"
- spot-check evidence snippets for obvious QA artifacts (Option A:, Answer:, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


QA_ARTIFACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bOption\s*[A-E]\s*:", re.IGNORECASE),
    re.compile(r"\bOptions?\s*:", re.IGNORECASE),
    re.compile(r"\bAnswer\s*:", re.IGNORECASE),
    re.compile(r"\bCorrect\s*answer\s*:", re.IGNORECASE),
    re.compile(r"\bExplanation\s*:", re.IGNORECASE),
    re.compile(r"\([A-E]\)"),
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
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _instance_input(evd: dict[str, Any]) -> str:
    """
    Self-BioRAG evidence schema varies; in our file, "instances" is a dict with string fields.
    """
    inst = evd.get("instances")
    if isinstance(inst, dict):
        val = inst.get("input")
        if isinstance(val, str):
            return val
    return ""


def _artifact_hits(snips: Iterable[str]) -> list[tuple[int, str, str]]:
    """
    Returns: list of (snippet_index, pattern, matched_text)
    """
    hits: list[tuple[int, str, str]] = []
    for i, s in enumerate(snips):
        if not isinstance(s, str):
            continue
        for pat in QA_ARTIFACT_PATTERNS:
            m = pat.search(s)
            if m:
                hits.append((i, pat.pattern, m.group(0)))
    return hits


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
    p.add_argument("--indices", type=str, default="0,1,50,200")
    p.add_argument("--evidence_topk_check", type=int, default=5)
    args = p.parse_args()

    dataset = _load_jsonl(args.dataset_jsonl)
    evidence = _load_evidence_json(args.evidence_json)

    print(f"[len] dataset={len(dataset)} evidence={len(evidence)}")
    if len(dataset) != len(evidence):
        print("[len] FAIL: dataset and evidence lengths differ")
    else:
        print("[len] OK")

    idxs = []
    for part in args.indices.split(","):
        part = part.strip()
        if not part:
            continue
        idxs.append(int(part))

    for idx in idxs:
        if idx < 0 or idx >= len(dataset) or idx >= len(evidence):
            print(f"[idx={idx}] SKIP: out of range")
            continue

        q = dataset[idx].get("question", "")
        qn = _norm(q if isinstance(q, str) else "")
        ev_in = _instance_input(evidence[idx])
        evn = _norm(ev_in)

        ok = qn and (qn in evn)
        print(f"[idx={idx}] question_in_evidence_input={'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"  dataset.question={q!r}")
            print(f"  evidence.instances[0].input={ev_in[:400]!r}{'...' if len(ev_in) > 400 else ''}")

        snips = evidence[idx].get("evidence", [])
        if not isinstance(snips, list):
            snips = []
        snips_topk = [s for s in snips[: args.evidence_topk_check] if isinstance(s, str)]
        hits = _artifact_hits(snips_topk)
        print(f"[idx={idx}] qa_artifacts_in_top{args.evidence_topk_check}={'YES' if hits else 'NO'} (hits={len(hits)})")
        for (si, pat, match) in hits[:6]:
            print(f"  - snippet[{si}] match={match!r} via /{pat}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


