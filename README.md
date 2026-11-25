# Do LLMs Split Couples' Chores Fairly?

IMT 598G final project (Jimin Keel, Mei Fang Wu). We ask open-weight LLMs how a married couple should split
11 household and child-care tasks, varying both partners' gender and race, and compare the answers with a
baseline that has no names or roles.

## Files

| File | What it does | Output |
| --- | --- | --- |
| `make_prompts.py` | Builds all prompts (Appendices A–C) | `prompts.jsonl` (10,000 main + 100 baseline) |
| `run.py` | Runs one model with vLLM, saving every 256 prompts; rerun the same command to resume | `outputs_<model>.jsonl`, `run_meta_<model>.json` |
| `validate.py` | Parses and checks answers, builds the dataset, prints a summary | `responses.csv`, `dataset.csv` |

## Running on Google Colab

Open `colab_run.ipynb` in Colab (File → Open notebook → GitHub), choose an L4 GPU, add `HF_TOKEN` to Secrets,
and run the cells in order. Outputs are saved to `MyDrive/chores_llm_outputs`; after a disconnect, rerun
cells 1–5 and the run cell, and finished prompts are skipped.

## Setup (any other GPU machine)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

## Run

```
python make_prompts.py                         # rebuild prompts (master seed 598)
python run.py --model gemma --sample           # 240-prompt sample
python validate.py outputs_gemma.jsonl         # check the sample
python run.py --model gemma                    # full run, resumes after the sample
python run.py --model qwen
python validate.py outputs_gemma.jsonl outputs_qwen.jsonl
```

## Settings

- Models: `google/gemma-4-12B-it` (FP8) and `Qwen/Qwen3.5-9B` (bf16), one NVIDIA L4, vLLM, no system prompt.
- Thinking off for both. Gemma uses its default sampling (temperature 1.0, top_p 0.95, top_k 64). Qwen runs in
  instruct (non-thinking) mode with the model card's settings (temperature 0.7, top_p 0.8, top_k 20,
  presence_penalty 1.5).
- Each prompt has its own seed; model commit hash, vLLM version and GPU are saved in `run_meta_<model>.json`.
