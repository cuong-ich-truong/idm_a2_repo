#!/usr/bin/env python3
"""
Quick OpenAI connectivity check (direct OpenAI, not Azure).

Reads:
- OPENAI_API_KEY
- OPENAI_MODEL_NAME

Loads project-local `.env` automatically if present.
Then makes a tiny ChatCompletion call (max_tokens=1) to validate:
- API key is accepted
- model name is valid/accessible
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dotenv_if_present() -> None:
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(env_path)


def main() -> int:
    _load_dotenv_if_present()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME")

    if not api_key:
        print("[fail] OPENAI_API_KEY is not set")
        return 2
    if not model:
        print("[fail] OPENAI_MODEL_NAME is not set")
        return 2

    try:
        import openai
    except Exception as e:
        print(f"[fail] openai import failed: {e}")
        print("Did you install deps? `python -m pip install -r requirements.txt`")
        return 2

    openai.api_key = api_key

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a connectivity test."},
                {"role": "user", "content": "Reply with one character."},
            ],
            temperature=0,
            max_tokens=1,
        )
    except Exception as e:
        print("[fail] OpenAI ChatCompletion call failed")
        print(f"  model={model!r}")
        print(f"  error={e!r}")
        return 1

    content = ""
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        pass

    print("[ok] OpenAI connection good")
    print(f"  model={model!r}")
    if getattr(resp, "id", None):
        print(f"  response_id={resp.id}")
    if content:
        print(f"  sample={content!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


