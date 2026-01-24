# Target Language Dataset Construction

AutoIF-based pipeline for high-quality target-language instruction datasets with automated verification.

## Pipeline

1. **Prompt Sourcing:** Real-user prompts (WildChat, translated) + native prompts
2. **Generation & Filtering:** Teacher model (Qwen3-235B) with rejection sampling (≥7/10)
3. **Augmentation:** Cross-lingual constraints + system prompt mixing

## Scripts

- **`000_autoif_pipeline_step1-5.py`** — Generate constraints + verification functions
- **`001_autoif_pipeline_step6-8.py`** — Combine prompts + generate verified responses

## Setup

```bash
pip install openai transformers torch jsonlines numpy tqdm fasttext huggingface-hub
export OPENAI_API_KEY="your-key"
```

## Usage

**Step 1: Generate Constraints**
```bash
python 000_autoif_pipeline_step1-5.py \
    --seed_path sample_data/seed_if_thai.txt \
    --output_path dataset/back_trans_filter.jsonl \
    --openai_model "gpt-4o-2024-08-06" \
    --num_workers 16
```

**Step 2: Generate SFT Dataset**
```bash
python 001_autoif_pipeline_step6-8.py \
    --back_trans_filter dataset/back_trans_filter.jsonl \
    --question_prompt dataset/question_prompt.jsonl \
    --output dataset/final_sft.jsonl \
    --openai_model "gpt-4o-2024-08-06" \
    --max_items 1000 \
    --num_workers 16
```