# AutoIF Data Processing Pipeline

Re-implementation of the [AutoIF pipeline](https://github.com/QwenLM/AutoIF) for generating instruction-following datasets with automated verification functions.

## Overview

The pipeline consists of two main scripts that implement the complete AutoIF workflow:

1. **`000_autoif_pipeline_step1-5.py`**: Generates constraint-based instructions with evaluation functions
2. **`001_autoif_pipeline_step6-8.py`**: Combines constraints with question prompts to create SFT training data

## Pipeline Steps

### Steps 1-5: Constraint Generation
1. Generate Instructions (RFT with KD)
2. Generate Verification Functions
3. Cross Validation (Function Cleaning/Processing)
4. Back Translation
5. Filtering (NLI)

### Steps 6-8: Dataset Creation
6. Combine constraints with question prompts
7. Generate responses with constraint verification
8. Filter and format as SFT dataset

## Requirements

```bash
pip install openai transformers torch jsonlines numpy tqdm fasttext huggingface-hub
```

Additional models downloaded automatically:
- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (NLI filtering)
- `facebook/fasttext-language-identification` (language detection)

## Quick Start

### Step 1: Generate Constrained Instructions

```bash
python 000_autoif_pipeline_step1-5.py \
    --seed_path sample_data/seed_if_thai.txt \
    --output_path dataset/back_trans_filter.jsonl \
    --openai_model "gpt-4o-2024-08-06" \
    --count 50 \
    --k_generations 1 \
    --n_choices 10 \
    --num_workers 16
```

**Parameters:**
- `--seed_path`: Seed instructions text file (one instruction per line)
- `--output_path`: Output JSONL file with constraint instructions
- `--openai_model`: OpenAI model for generation (default: `gpt-4o-mini`)
- `--count`: Number of instructions per generation prompt (default: 50)
- `--k_generations`: Number of generation calls (default: 1)
- `--n_choices`: Number of evaluation function choices per request (default: 5)
- `--num_workers`: Parallel workers (default: 8)
- `--base_url`: OpenAI API base URL (default: `https://api.openai.com/v1`)
- `--no_cache`: Disable caching (cache stored in `cache_step1to5/`)

### Step 2: Generate SFT Dataset

```bash
python 001_autoif_pipeline_step6-8.py \
    --back_trans_filter dataset/back_trans_filter.jsonl \
    --question_prompt dataset/question_prompt.jsonl \
    --output dataset/final_sft.jsonl \
    --openai_model "gpt-4o-2024-08-06" \
    --max_items 1000 \
    --num_workers 16
```

**Parameters:**
- `--back_trans_filter`: Input constraint instructions from step 1
- `--question_prompt`: JSONL file with question prompts to combine with constraints
- `--output`: Output SFT dataset in conversational format
- `--openai_model`: OpenAI model for response generation
- `--max_items`: Maximum number of samples to generate (default: 500)
- `--quality_threshold`: Minimum quality score 0-1 (default: 0.8)
- `--enable_translation`: Enable Thai translation (default: False)
- `--translation_model`: Model for translation (default: `gpt-4o-mini`)
- `--num_workers`: Parallel workers (default: 8)
- `--force_thai`: Force Thai responses
- `--no_cache`: Disable caching (cache stored in `cache/`)
- `--clear_cache`: Clear existing cache before running

## Output Formats

### Step 1 Output (`back_trans_filter.jsonl`)
```json
{
  "instruction": "Constraint instruction text",
  "evaluation_function": "def verify_output(response: str) -> bool: ...",
  "back_translation": "Back-translated instruction"
}
```

### Step 2 Output (`final_sft.jsonl`)
```json
{
  "conversations": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User instruction with constraints"},
    {"role": "assistant", "content": "Generated response"}
  ]
}
```

## Running the Full Pipeline

See [example.sh](example.sh) for a complete example:

```bash
bash example.sh
```

## Environment Variables

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it via command line:
```bash
--openai_api_key "your-api-key-here"
```

## Caching

Both scripts implement automatic caching to avoid redundant API calls:
- Step 1 cache: `cache_step1to5/`
- Step 2 cache: `cache/`

Clear cache:
```bash
rm -rf cache_step1to5/ cache/
```

## Notes

- The pipeline requires GPU for NLI model inference (automatic CPU fallback available)
- Generation costs depend on OpenAI model pricing and `count`/`max_items` parameters
- Increase `num_workers` for faster parallel processing
- Use `--no_cache` during development to force regeneration
