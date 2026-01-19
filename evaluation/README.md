# Evaluation Suite

Evaluation scripts for benchmarking language models on Thai sovereign capability datasets.

## Requirements

```bash
pip install pandas pyarrow httpx openai tqdm huggingface-hub
```

For agent mode (nitibench with RAG tools):
```bash
pip install fastapi uvicorn datasets faiss-cpu sentence-transformers
```

## Dataset Download

Download the evaluation dataset from HuggingFace:

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download the dataset
huggingface-cli download kunato/typhoon-s-sovereign-capability-dataset nitibench_test.parquet  --repo-type dataset --local-dir ./data
```

## Quick Start

### Evaluation with Agent Mode (RAG Tools)

This evaluation requires running models with tool-calling capabilities. You need to start both a vLLM server for model inference and the Nitibench RAG server.

**Step 1: Start vLLM Server**

In a separate terminal, start the vLLM server with your model:

```bash
# Example: Start vLLM with Typhoon-S Legal model
vllm serve kunato/typhoon-s-4b-nitibench-ccl-legal-research-preview \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.4 \
  --max_num_seqs 128
```

The vLLM server will be available at `http://localhost:8000/v1` (OpenAI-compatible API).

**Step 2: Start the Nitibench RAG Server**

In another terminal:

```bash
# Default port 8932
python nitibench_server.py

# Or specify a custom port
PORT=34431 python nitibench_server.py
```

The server will:
- Download and cache Thai legal documents
- Build a FAISS index for fast semantic search
- Provide `/search` and `/read` endpoints for the agent

**Step 3: Run Evaluation with Agent**

In a third terminal:

```bash
export OPENAI_API_KEY=...

python evaluate_parquet_dataset.py \
    --dataset_path ./data/nitibench_test.parquet \
    --output_file typhoon-s-nitibench-agent.jsonl \
    --model kunato/typhoon-s-4b-nitibench-ccl-legal-research-preview \
    --base_url http://localhost:8000/v1 \
    --judge_model gpt-5-nano --judge_base_url https://api.openai.com/v1 \
    --n_samples 1 --max_concurrent 64 \
    --use_agent --agent_url http://localhost:8932
```

For the judge model, you can use OpenAI API (set `OPENAI_API_KEY` properly) or another vLLM instance.

## Command Options

### Dataset Options

- `--dataset_path`: Path to local parquet dataset file (required)

### Model Options

- `--model`: Model for generating responses (default: `gpt-4o-mini`)
- `--judge_model`: Model for judging correctness (default: `gpt-4o-mini`)
- `--base_url`: Base URL for generation client (e.g., `https://api.openai.com/v1`)
- `--judge_base_url`: Base URL for judge client (optional, defaults to `base_url`)

### Evaluation Options

- `--n_samples`: Number of responses to generate per question (default: 8)
- `--max_concurrent`: Maximum concurrent API calls (default: 8)
- `--limit`: Limit number of items to evaluate (useful for testing)
- `--output_file`: Output JSONL file path (default: `evaluation_report.jsonl`)

### Agent Options

- `--use_agent`: Enable agent mode with tool use
- `--agent_url`: URL of the tool server (default: `http://localhost:8932`)
- `--max_tool_response_length`: Maximum length for tool responses (default: 4000)
- `--system_prompt`: Custom system prompt (optional)

### Other Options

- `--use_search`: Use web search tool (for GPT-5 models with web search capability)

## Additional Examples

### Using OpenAI Models

If you want to evaluate using OpenAI models instead of local vLLM:

```bash
export OPENAI_API_KEY=your_api_key_here

python evaluate_parquet_dataset.py \
  --dataset_path ./data/nitibench_test.parquet \
  --output_file gpt4o-nitibench-agent.jsonl \
  --model gpt-4o-mini \
  --judge_model gpt-5-nano \
  --base_url https://api.openai.com/v1 \
  --n_samples 1 \
  --max_concurrent 8 \
  --use_agent \
  --agent_url http://localhost:8932
```

## Output Format

The evaluation produces a JSONL file where each line contains:

```json
{
  "index": 0,
  "prompt": [...],
  "generated_responses": [...],
  "judgments": [true, false, true, ...],
  "scores": [1.0, 0.0, 1.0, ...],
  "n_correct": 5,
  "n_total": 8,
  "pass_rate": 0.625,
  "pass_at_1": 0.625,
  "pass_at_8": 0.996,
  ...
}
```

At the end of evaluation, aggregate metrics are printed:
- **pass_rate**: Average accuracy across all samples
- **pass@k**: Probability of getting at least one correct answer in k samples

## Notes

- For local model evaluation: Start vLLM server first, then nitibench server, then run evaluation
- For OpenAI models: Set `OPENAI_API_KEY` environment variable, start nitibench server, then run evaluation  
- Agent mode (`--use_agent`) is required for Nitibench evaluation as questions need legal document retrieval
- First run will download and cache Thai legal documents and build search indices (nitibench server)
- The evaluation script automatically resumes from where it left off if interrupted
- Use `--max_concurrent` to control API rate limits and costs
- Ensure your model supports tool/function calling for agent mode to work properly
