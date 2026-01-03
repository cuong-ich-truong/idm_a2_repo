#!/usr/bin/env python3
"""
Evaluate MedAgents output JSONL produced by scripts/run_medagents_baseline.py.

We prefer self-contained evaluation:
- uses `gold_answer` embedded in each record
- optionally uses `meta_info` embedded in each record for breakdown
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_file", type=Path, required=True)
    args = p.parse_args()

    total = 0
    correct = 0
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_correct: dict[str, int] = defaultdict(int)

    with args.pred_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at {args.pred_file}:{line_no}: {e}") from e

            pred = (rec.get("pred_answer") or "").strip()
            gold = (rec.get("gold_answer") or "").strip()
            meta = rec.get("meta_info")
            meta_key = meta if isinstance(meta, str) else "all"

            total += 1
            by_type_total[meta_key] += 1

            is_correct = bool(pred) and (pred == gold or gold in pred)
            if is_correct:
                correct += 1
                by_type_correct[meta_key] += 1

    acc = correct / total if total else 0.0
    print(f"[overall] n={total} acc={acc:.4f}")

    for k in sorted(by_type_total.keys()):
        n = by_type_total[k]
        c = by_type_correct.get(k, 0)
        a = c / n if n else 0.0
        print(f"[{k}] n={n} acc={a:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


