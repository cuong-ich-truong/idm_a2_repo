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

### Real run (small slice) — OpenAI direct

```bash
python3 scripts/run_medagents_baseline.py \
  --llm_provider openai \
  --model_name openai \
  --dataset_name MedQA \
  --dataset_dir vendor/med_agents/datasets/MedQA/ \
  --method syn_verif \
  --max_attempt_vote 3 \
  --start_pos 0 \
  --end_pos 1 \
  --output_dir outputs/MedQA/
```

### Real run (small slice) — evidence injection (POC-A)

Always inject evidence (from cache):

```bash
python3 scripts/run_medagents_baseline.py \
  --llm_provider openai \
  --model_name openai_evd \
  --dataset_name MedQA \
  --dataset_dir vendor/med_agents/datasets/MedQA/ \
  --method syn_verif \
  --max_attempt_vote 3 \
  --start_pos 0 \
  --end_pos 1 \
  --output_dir outputs/MedQA/ \
  --evidence_json data/retrieved_med_qa_test.json \
  --evidence_mode always \
  --evidence_topk 5 \
  --evidence_max_chars 2500 \
  --run_tag t0_1 \
  --log_evidence
```

### Evaluate

```bash
python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/openai-syn_verif-t0_10-20260103_153310_597275.jsonl

python3 scripts/eval_medagents_outputs.py \
  --pred_file outputs/MedQA/openai_evd-syn_verif-t0_10-20260103_154359_810591.jsonl

```

## Project Structure

- `src/` - Main source code
- `data/` - Data files
- `notebooks/` - Jupyter notebooks for experimentation
- `scripts/` - Utility scripts

## References

- `../med_agents_repo/` - MedAgent reference implementation
- `../self_biorag_repo/` - BioRAG reference implementation
