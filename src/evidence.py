from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QA_ARTIFACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bOption\s*[A-E]\s*:", re.IGNORECASE),
    re.compile(r"\bOptions?\s*:", re.IGNORECASE),
    re.compile(r"\bAnswer\s*:", re.IGNORECASE),
    re.compile(r"\bCorrect\s*answer\s*:", re.IGNORECASE),
    re.compile(r"\bExplanation\s*:", re.IGNORECASE),
    re.compile(r"\([A-E]\)"),
]


def load_evidence_json(path: Path) -> list[dict[str, Any]]:
    """
    Loads Self-BioRAG-style evidence cache.

    Expected shape: a JSON list; each element i corresponds to dataset index i.
    We only ever use `record["evidence"]` (list of strings). We DO NOT use `instances.input`.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Evidence JSON must be a list; got {type(data)}")
    return data  # type: ignore[return-value]


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


@dataclass(frozen=True)
class EvidenceFormatConfig:
    topk: int = 5
    max_chars: int = 2500
    min_snip_chars: int = 80
    filter_mode: str = "artifact_only"  # "off" | "artifact_only"


def _should_drop(snippet: str, cfg: EvidenceFormatConfig) -> bool:
    sn = _norm_ws(snippet)
    if len(sn) < cfg.min_snip_chars:
        return True
    if cfg.filter_mode == "off":
        return False
    for pat in QA_ARTIFACT_PATTERNS:
        if pat.search(snippet or ""):
            return True
    return False


def format_evidence_context(record: dict[str, Any], cfg: EvidenceFormatConfig) -> str:
    """
    Returns a single string, ready to paste into prompts.
    Format:
      [E1] ...
      [E2] ...
    """
    snips = record.get("evidence", [])
    if not isinstance(snips, list):
        return ""
    snips = [s for s in snips if isinstance(s, str)]
    if cfg.topk and cfg.topk > 0:
        snips = snips[: cfg.topk]

    kept: list[str] = []
    for s in snips:
        if _should_drop(s, cfg):
            continue
        kept.append(_norm_ws(s))

    if not kept:
        return ""

    lines: list[str] = []
    total = 0
    for i, s in enumerate(kept, start=1):
        line = f"[E{i}] {s}"
        if total + len(line) + 1 > cfg.max_chars:
            break
        lines.append(line)
        total += len(line) + 1

    return "\n".join(lines).strip()
