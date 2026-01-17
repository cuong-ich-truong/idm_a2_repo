#!/usr/bin/env python3
"""
Filter a MedQA-style JSONL file down to rows where `meta_info == "step2&3"`.

Default input (vendored): vendor/med_agents/datasets/MedQA/test.jsonl

Usage:
  python3 scripts/filter_medqa_step23.py \
    --input_jsonl vendor/med_agents/datasets/MedQA/test.jsonl \
    --output_jsonl vendor/med_agents/datasets/MedQA/test.step2_3.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _derive_default_output(input_path: Path) -> Path:
    # e.g. test.jsonl -> test.step2_3.jsonl
    if input_path.suffix == ".jsonl":
        return input_path.with_name(f"{input_path.stem}.step2_3.jsonl")
    return input_path.with_name(f"{input_path.name}.step2_3.jsonl")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_jsonl",
        type=str,
        default="vendor/med_agents/datasets/MedQA/test.jsonl",
        help="Path to input JSONL (each line is a JSON object).",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Path to output JSONL (defaults to <input>.step2_3.jsonl).",
    )
    p.add_argument(
        "--meta_value",
        type=str,
        default="step2&3",
        help='Keep rows where obj["meta_info"] == this value.',
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print counts; do not write output.",
    )
    args = p.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else _derive_default_output(in_path)
    )

    total = 0
    kept = 0
    bad = 0

    if not in_path.exists():
        raise SystemExit(f"input_jsonl not found: {in_path}")

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
    else:
        out_f = None

    try:
        with in_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                total += 1
                try:
                    obj = json.loads(s)
                except Exception:
                    bad += 1
                    continue

                if obj.get("meta_info") != args.meta_value:
                    continue

                kept += 1
                if out_f is not None:
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    print(
        json.dumps(
            {
                "input_jsonl": str(in_path),
                "output_jsonl": None if args.dry_run else str(out_path),
                "meta_value": args.meta_value,
                "total_rows_seen": total,
                "rows_kept": kept,
                "rows_dropped": total - kept,
                "bad_json_lines": bad,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
