#!/usr/bin/env python3
"""
Run vendored MedAgents "as-is" (no RAG) without modifying upstream code.

How it works:
- Adds `vendor/med_agents/upstream/` to sys.path so upstream flat imports work.
- Runs the same loop as upstream run.py and writes JSONL.

Env vars (OpenAI direct):
- OPENAI_API_KEY (required)
- OPENAI_MODEL_NAME (required)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from string import punctuation

import tqdm


LOGGER = logging.getLogger("run_medagents_baseline")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_project_root_to_syspath() -> Path:
    """
    Makes `import src.*` work when running: `python3 scripts/run_medagents_baseline.py`
    """
    root = _project_root()
    sys.path.insert(0, str(root))
    return root


def _load_dotenv_if_present() -> None:
    """
    Loads project-local `.env` (gitignored) so you don't need to export vars every run.
    """
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        # keep script runnable even if dependency is missing
        return
    load_dotenv(env_path)


def _add_upstream_to_syspath() -> Path:
    upstream_dir = _project_root() / "vendor" / "med_agents" / "upstream"
    if not upstream_dir.exists():
        raise RuntimeError(f"Missing vendored upstream dir: {upstream_dir}")
    sys.path.insert(0, str(upstream_dir))
    return upstream_dir


def _configure_openai_direct() -> str:
    import openai  # imported after deps are installed

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    if not model:
        raise RuntimeError("Missing OPENAI_MODEL_NAME")

    # openai==0.27.x expects this for direct OpenAI usage
    openai.api_key = api_key
    return model


class _OpenAIChatHandler:
    """
    Minimal adapter that matches the upstream handler API expected by `utils.fully_decode`.
    """

    def __init__(self, model: str):
        self.model = model
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_total_tokens = 0
        self.total_wall_s = 0.0

    def get_output_multiagent(
        self,
        system_role,
        user_input,
        max_tokens,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    ):
        import openai

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.call_count += 1
                call_no = self.call_count
                stage = getattr(self, "current_stage", None) or ""
                meta = getattr(self, "current_meta", None) or {}
                t0 = time.time()
                sys_len = len(system_role or "")
                usr_len = len(user_input or "")
                LOGGER.debug(
                    "[openai] %s call#%s attempt=%s/%s model=%s max_tokens=%s temp=%s chars(system=%s,user=%s)",
                    stage,
                    call_no,
                    attempt + 1,
                    max_attempts,
                    self.model,
                    max_tokens,
                    temperature,
                    sys_len,
                    usr_len,
                )
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_input},
                    ],
                )
                dt = time.time() - t0
                self.total_wall_s += dt

                usage = getattr(resp, "usage", None)
                pt = ct = tt = None
                if usage:
                    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                    ct = int(getattr(usage, "completion_tokens", 0) or 0)
                    tt = int(getattr(usage, "total_tokens", 0) or 0)
                    self.total_prompt_tokens += pt
                    self.total_completion_tokens += ct
                    self.total_total_tokens += tt
                    LOGGER.debug(
                        "[openai] call#%s done in %.2fs tokens(prompt=%s,completion=%s,total=%s)",
                        call_no,
                        dt,
                        pt,
                        ct,
                        tt,
                    )
                else:
                    LOGGER.debug(
                        "[openai] call#%s done in %.2fs tokens(usage=missing)",
                        call_no,
                        dt,
                    )
                out_text = "ERROR."
                if (
                    resp.choices
                    and resp.choices[0].message
                    and "content" in resp.choices[0].message
                ):
                    out_text = resp.choices[0].message["content"]

                # Full per-call record (prompt + output) goes to the log file only (DEBUG level).
                LOGGER.debug(
                    "[llm_call] %s",
                    json.dumps(
                        {
                            "call_no": call_no,
                            "attempt": attempt + 1,
                            "model": self.model,
                            "stage": stage,
                            "meta": meta,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "frequency_penalty": frequency_penalty,
                            "presence_penalty": presence_penalty,
                            "stop": stop,
                            "duration_s": round(dt, 4),
                            "tokens": {
                                "prompt_tokens": pt,
                                "completion_tokens": ct,
                                "total_tokens": tt,
                            },
                            "prompt": {
                                "system": system_role or "",
                                "user": user_input or "",
                            },
                            "output": out_text,
                        },
                        ensure_ascii=False,
                    ),
                )
                return out_text
            except Exception as e:
                LOGGER.warning(
                    "[warn] OpenAI call failed (attempt %s/%s): %r",
                    attempt + 1,
                    max_attempts,
                    e,
                )
                if attempt == max_attempts - 1:
                    return "ERROR."


def _setup_logging(log_path: Path) -> None:
    """
    File gets full debug logs (per-call details). Console stays minimal.
    """
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    console_h = logging.StreamHandler(stream=sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(logging.Formatter("%(message)s"))

    LOGGER.addHandler(file_h)
    LOGGER.addHandler(console_h)


def main() -> int:
    _load_dotenv_if_present()
    _add_project_root_to_syspath()
    _add_upstream_to_syspath()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--llm_provider",
        default="openai",
        choices=["openai"],
        help="Which API to use for ChatCompletion calls.",
    )
    p.add_argument(
        "--model_name",
        default="chatgpt",
        help="Run label used for output filename (does not have to match provider model id).",
    )
    p.add_argument(
        "--run_tag",
        default="",
        help="Optional short tag appended to output filename (e.g. ablation name).",
    )
    p.add_argument(
        "--dataset_name",
        default="MedQA",
        choices=["MedQA", "PubMedQA", "MedMCQA", "MedicationQA"],
    )
    p.add_argument("--dataset_dir", default="vendor/med_agents/datasets/MedQA/")
    p.add_argument("--start_pos", type=int, default=0)
    p.add_argument("--end_pos", type=int, default=5, help="-1 means full dataset")
    p.add_argument("--output_dir", default="outputs/MedQA/")
    p.add_argument("--max_attempt_vote", type=int, default=3)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="no API calls; writes stub records for wiring checks only",
    )
    p.add_argument(
        "--evidence_json",
        type=str,
        default="",
        help="Optional evidence cache JSON (Self-BioRAG style). If empty, no evidence is used.",
    )
    p.add_argument("--evidence_topk", type=int, default=5)
    p.add_argument("--evidence_max_chars", type=int, default=2500)
    p.add_argument(
        "--evidence_filter_mode",
        choices=["off", "artifact_only"],
        default="artifact_only",
        help="Runtime filtering for evidence[] snippets (you can also pre-filter offline).",
    )
    p.add_argument("--evidence_min_snip_chars", type=int, default=80)
    p.add_argument(
        "--log_evidence",
        action="store_true",
        help="If set, store the formatted evidence context (candidate + used) in each output record.",
    )
    args = p.parse_args()

    output_dir = _project_root() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_pos_label = "all" if args.end_pos == -1 else str(args.end_pos)
    range_part = f"-s{args.start_pos}-e{end_pos_label}"
    tag = (args.run_tag or "").strip()
    safe_tag = ""
    if tag:
        safe_tag = "".join(ch for ch in tag if ch.isalnum() or ch in ("-", "_"))
        safe_tag = safe_tag[:40]
    tag_part = f"-{safe_tag}" if safe_tag else ""
    out_path = output_dir / f"{args.model_name}{tag_part}{range_part}-{run_ts}.jsonl"
    log_path = out_path.with_suffix(".log")
    _setup_logging(log_path)
    LOGGER.info("[log] %s", log_path)

    # upstream imports (flat)
    from data_utils import MyDataset  # type: ignore
    from utils import fully_decode  # type: ignore

    evidence_cache = None
    if args.evidence_json and not args.dry_run:
        # local project import
        from src.evidence import (
            EvidenceFormatConfig,
            format_evidence_context,
            load_evidence_json,
        )

        evidence_path = Path(args.evidence_json)
        if not evidence_path.is_absolute():
            evidence_path = _project_root() / evidence_path
        if not evidence_path.exists():
            raise RuntimeError(
                "Evidence file not found: "
                f"{evidence_path}\n\n"
                "Fix:\n"
                "Point --evidence_json to an existing file (e.g. data/retrieved_med_qa_test.json).\n"
            )
        evidence_cache = load_evidence_json(evidence_path)
        evidence_cfg = EvidenceFormatConfig(
            topk=args.evidence_topk,
            max_chars=args.evidence_max_chars,
            min_snip_chars=args.evidence_min_snip_chars,
            filter_mode=args.evidence_filter_mode,
        )

    if args.dry_run:
        handler = None
    else:
        model = _configure_openai_direct()
        handler = _OpenAIChatHandler(model)

    dataobj = MyDataset("test", args, traindata_obj=None)
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)
    with out_path.open("a", encoding="utf-8") as f:
        for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
            raw_sample = dataobj.get_by_idx(idx)
            question_raw = raw_sample["question"]
            question = (
                question_raw
                if question_raw and question_raw[-1] in punctuation
                else question_raw + "?"
            )

            if args.dataset_name in ["MedQA", "MedMCQA"] or "MMLU" in args.dataset_name:
                options = raw_sample["options"]
                gold_answer = raw_sample["answer_idx"]
            elif args.dataset_name == "PubMedQA":
                question = raw_sample["context"] + " " + question
                options = raw_sample["options"]
                gold_answer = raw_sample["answer_idx"]
            elif args.dataset_name in ["MedicationQA"]:
                options = ""
                gold_answer = raw_sample["answer_idx"]
            else:
                raise ValueError(f"Unsupported dataset_name={args.dataset_name}")

            if args.dry_run:
                data_info = {
                    "idx": idx,
                    "question": question,
                    "options": options,
                    "pred_answer": "",
                    "gold_answer": gold_answer,
                    "raw_output": "DRY_RUN",
                }
            else:
                assert handler is not None
                evidence_context = None
                candidate_ctx = None
                if evidence_cache is not None and 0 <= idx < len(evidence_cache):
                    from src.evidence import format_evidence_context

                    candidate_ctx = format_evidence_context(
                        evidence_cache[idx], evidence_cfg
                    )
                    evidence_context = candidate_ctx or None

                data_info = fully_decode(
                    idx,
                    idx,
                    question,
                    options,
                    gold_answer,
                    handler,
                    args,
                    dataobj,
                    evidence_context=evidence_context,
                )
                # add extra audit-friendly fields without changing upstream logic/prompts
                data_info["idx"] = idx
                if isinstance(raw_sample, dict) and "meta_info" in raw_sample:
                    data_info["meta_info"] = raw_sample["meta_info"]
                if evidence_cache is not None:
                    data_info["evidence_enabled"] = bool(args.evidence_json)
                    data_info["evidence_injected"] = bool(evidence_context)
                    if args.log_evidence:
                        data_info["evidence_json"] = args.evidence_json
                        data_info["evidence_params"] = {
                            "topk": args.evidence_topk,
                            "max_chars": args.evidence_max_chars,
                            "min_snip_chars": args.evidence_min_snip_chars,
                            "filter_mode": args.evidence_filter_mode,
                        }
                        # candidate_ctx: what we *would* inject for this idx (after filtering/truncation)
                        data_info["evidence_candidate_context"] = candidate_ctx or ""
                        # evidence_context: what we *did* inject (empty if gate said NO)
                        data_info["evidence_used_context"] = evidence_context or ""

            f.write(json.dumps(data_info) + "\n")

    if isinstance(handler, _OpenAIChatHandler):
        LOGGER.info(
            "[openai] run summary: calls=%s tokens(prompt=%s,completion=%s,total=%s) wall_s=%.2f",
            handler.call_count,
            handler.total_prompt_tokens,
            handler.total_completion_tokens,
            handler.total_total_tokens,
            handler.total_wall_s,
        )
    LOGGER.info("[done] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
