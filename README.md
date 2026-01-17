# IDM A2 Project

Medical QA project comparing RAG and MedAgent approaches.

## Setup

```bash
# Create virtual environment
python3.10 -m venv .venv_py310
source .venv_py310/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# NLTK data (required)
# Upstream MedAgents uses `sent_tokenize`, which requires the `punkt` tokenizer data.
#
# If you hit SSL cert errors when downloading, use certifi's CA bundle:
python -m pip install -U certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
python -c "import nltk; nltk.download('punkt')"
```

## Baseline: run MedAgents as-is (vendored, no RAG)

We vendor MedAgents code under `vendor/med_agents/upstream/` and keep it unmodified.
Run it via a wrapper script that configures the LLM provider via env vars at runtime (we do **not** edit upstream `api_utils.py`):

```bash
cd /Users/cuongtruong/dev/personal/study/c3009_idm/idm_a2/idm_a2_repo
source .venv_py310/bin/activate

# Optional (recommended): project-local secrets (gitignored)
cp env.example .env
# edit .env and set OPENAI_API_KEY / OPENAI_MODEL_NAME once

# OpenAI (direct) (required if using --llm_provider openai)
export OPENAI_API_KEY="<your-key>"
export OPENAI_MODEL_NAME="gpt-3.5-turbo"   # or your preferred ChatCompletion model

```

### Check OpenAI connection (recommended)

```bash
python3 scripts/check_openai.py
```

### Smoke test (no API calls)

```bash
python scripts/run_medagents_baseline.py \
  --dry_run \
  --start_pos 0 \
  --end_pos 2 \
  --output_dir outputs/MedQA/
```

### List all args / help

```bash
# Standard argparse help
python3 scripts/run_medagents_baseline.py -h
python3 scripts/run_medagents_baseline.py --help
```

### Real run (small slice) — OpenAI direct

```bash
python3 scripts/run_medagents_baseline.py \
  --llm_provider openai \
  --model_name openai \
  --dataset_name MedQA \
  --dataset_jsonl vendor/med_agents/datasets/MedQA/test.jsonl \
  --max_attempt_vote 3 \
  --start_pos 0 \
  --end_pos 100 \
  --run_tag v1 \
  --output_dir outputs/MedQA/raw
```

### Real run (small slice) — evidence injection (POC-A)

Always inject evidence (from cache):

```bash
python3 scripts/run_medagents_baseline.py \
  --llm_provider openai \
  --model_name openai_evd \
  --dataset_name MedQA \
  --dataset_jsonl vendor/med_agents/datasets/MedQA/test.jsonl \
  --max_attempt_vote 3 \
  --start_pos 0 \
  --end_pos 100 \
  --output_dir outputs/MedQA/raw \
  --evidence_json data/retrieved_med_qa_test.json \
  --evidence_topk 5 \
  --evidence_max_chars 2500 \
  --run_tag v1 \
  --log_evidence
```

### Evaluate

```bash
python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/base/openai-v1-s0-e300.jsonl

python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/base/openai-v1-s450-e750.jsonl

python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/base/openai-v1-s900-e1200.jsonl

python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/rag/openai_evd-v1-s0-e300.jsonl

python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/rag/openai_evd-v1-s450-e750.jsonl

python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/rag/openai_evd-v1-s900-e1200.jsonl
```

### Other useful scripts

```bash
python3 scripts/filter_medqa_step23.py \
  --input_jsonl vendor/med_agents/datasets/MedQA/test.jsonl \
  --output_jsonl vendor/med_agents/datasets/MedQA/test.step2_3.jsonl

# Verify evidence cache aligns 1-1 with dataset test.jsonl (by index + question text)
python3 scripts/verify_evidence_alignment.py \
  --dataset_jsonl vendor/med_agents/datasets/MedQA/test.jsonl \
  --evidence_json data/retrieved_med_qa_test.json \
  --check_question_text \
  --require_nonempty_evidence
```

## Project Structure

- `src/` - Main source code
- `data/` - Data files
- `notebooks/` - Jupyter notebooks for experimentation
- `scripts/` - Utility scripts

## References

- `../med_agents_repo/` - MedAgent reference implementation
- `../self_biorag_repo/` - BioRAG reference implementation
