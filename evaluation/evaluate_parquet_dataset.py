#!/usr/bin/env python3
"""
Evaluation script for Parquet dataset with reward_model guideline.
Based on score_wangchan_instruct.py
"""

import json
import httpx
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm as sync_tqdm

@dataclass
class ScoringConfig:
    """Configuration for the scoring process"""
    model_name: str = "gpt-4o-mini"  # Model for generation
    judge_model: str = "gpt-5-nano"  # Model for judging correctness
    n_samples: int = 8  # Number of samples per instruction
    temperature: float = 0.7
    max_tokens: int = 8192
    max_concurrent: int = 8  # Max concurrent API calls
    dataset_path: str = ""  # Path to parquet dataset file
    output_file: str = "evaluation_report.jsonl"
    resume: bool = True  # Resume from existing output file
    base_url: Optional[str] = None  # Base URL for generation client
    judge_base_url: Optional[str] = None  # Base URL for judge client
    system_prompt: Optional[str] = None  # System prompt to append
    use_search: bool = False  # Use web search tool
    use_agent: bool = False  # Use agent loop
    agent_url: str = "http://localhost:8932"  # Tool server URL
    max_tool_response_length: int = 2048  # Max length for tool response
    tool_response_truncate_side: str = "center"  # Truncation side: left, right, center
    task: str = "nitibench"  # Task name for tool selection

# --- Tool Implementations ---

async def execute_search(arguments: Dict[str, Any], config: ScoringConfig, client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{config.agent_url}/search", json=arguments)
    resp.raise_for_status()
    
    raw_results = resp.json().get("result", [])
    pretty_results = []
    for retrieval in raw_results:
        formatted = _passages2string(retrieval)
        pretty_results.append(formatted)
    
    final_result = "\n---\n".join(pretty_results)
    if not final_result:
        final_result = "No search results found."
    return json.dumps({"result": final_result}, ensure_ascii=False)

async def execute_read_law(arguments: Dict[str, Any], config: ScoringConfig, client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{config.agent_url}/read", json=arguments)
    if resp.status_code == 404:
        try:
            detail = resp.json().get("detail", "Law not found.")
            return json.dumps({"result": detail}, ensure_ascii=False)
        except:
            return json.dumps({"result": "Law not found."}, ensure_ascii=False)
    else:
        resp.raise_for_status()
        return resp.json().get("text", "")

# --- Tool Registry ---

TOOL_REGISTRY = {
    "nitibench": {
        "schemas": [
            {
                "type": "function",
                "function": {
                    "name": "search_law",
                    "description": "Searches for relevant Thai laws based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["queries"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_law",
                    "description": "Read the full content of a specific law section.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "law_name": {
                                "type": "string",
                                "description": "Name of the law."
                            },
                            "section_num": {
                                "type": "string",
                                "description": "Section number (optional)."
                            }
                        },
                        "required": ["law_name"]
                    }
                }
            }
        ],
        "system_prompt": (
            "You are a legal expert in Thai law. You are given a legal question and you need to answer it. "
            "You have access to a `search_law` tool that can search for relevant Thai laws, "
            "and a `read_law` tool that can read the full content of a specific law section. "
            "You should use the `search_law` tool to find relevant laws, and then use `read_law` to get the full text if needed. "
            "Reason step by step before using the tool. "
            "After gathering information, provide your final answer."
        ),
        "implementations": {
            "search_law": execute_search,
            "read_law": execute_read_law
        }
    }
}

def _clean_json(response: str) -> str:
    """Clean response string to extract valid JSON."""
    try:
        response = response.replace("```json", "").replace("```", "").strip()
        # Attempt to find the first and last curly braces
        first_brace = response.index("{")
        last_brace = response.rindex("}") + 1
        json_str = response[first_brace:last_brace]
        return json_str
    except ValueError:
        # If no braces found, return the original response
        return response

def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()

def calculate_pass_at_k(n, c, k):
    """
    Calculate pass@k
    n: total samples
    c: correct samples
    k: k in pass@k
    """
    if n < k:
        return 0.0 # Should not happen if we sample enough
    if c == 0:
        return 0.0
    
    # pass@k = 1 - comb(n-c, k) / comb(n, k)
    # Using log space for stability if needed, but for small n, direct calculation is fine.
    # However, simpler approximation or direct simulation:
    # If we have c correct out of n, what is prob that taking k samples has at least 1 correct?
    
    # If c > n-k, then we must have picked at least one correct.
    if c > n - k:
        return 1.0
        
    # prob of picking k failures:
    # (n-c)/n * (n-c-1)/(n-1) * ... * (n-c-k+1)/(n-k+1)
    
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
        
    return 1.0 - prob_all_fail

def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

class ParquetScorer:
    """Handles scoring of parquet dataset"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        
        # Initialize generation client
        gen_client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if config.base_url:
            gen_client_kwargs["base_url"] = config.base_url
        self.client = AsyncOpenAI(**gen_client_kwargs)
        
        # Initialize judge client (separate from generation client)
        judge_client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if config.judge_base_url:
            judge_client_kwargs["base_url"] = config.judge_base_url
        elif config.base_url:  # Fallback to generation base_url if judge not specified
            judge_client_kwargs["base_url"] = config.base_url
        self.judge_client = AsyncOpenAI(**judge_client_kwargs)
        
        self.processed_indices = set()
        
        # Load already processed items if resume is enabled
        if config.resume and os.path.exists(config.output_file):
            self._load_processed_indices()
    
    def _load_processed_indices(self):
        """Load indices of already processed items"""
        try:
            with open(self.config.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if 'index' in item:
                            self.processed_indices.add(item['index'])
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(self.processed_indices)} already processed items")
        except Exception as e:
            print(f"Error loading processed items: {e}")
    
    async def generate_responses(self, instruction: str) -> Tuple[List[str], List[List[Dict]]]:
        """
        Generate n samples for a given instruction using parallel API calls
        """
        tasks = [self._generate_single_response(instruction) for _ in range(self.config.n_samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_responses = []
        valid_messages = []
        for resp in results:
            if isinstance(resp, Exception):
                print(f"Error generating response: {resp}")
            elif resp:
                content, msgs = resp
                if content:
                    valid_responses.append(content)
                    valid_messages.append(msgs)
        
        return valid_responses, valid_messages
    
    async def _generate_agent_response(self, instruction: str) -> Tuple[Optional[str], List[Dict]]:
        """Generate response using agent loop with tools"""
        messages = []
        
        task_config = TOOL_REGISTRY.get(self.config.task)
        if not task_config:
            print(f"Warning: No tool configuration found for task '{self.config.task}'. Running without tools.")
            tools = []
            system_prompt = self.config.system_prompt
        else:
            tools = task_config["schemas"]
            system_prompt = self.config.system_prompt or task_config["system_prompt"]
            tool_implementations = task_config["implementations"]

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": instruction})

        max_turns = 12
        current_turn = 0
        
        kwargs = {}
        if 'gpt-5' in self.config.model_name:
            kwargs['reasoning_effort'] = 'low'
        else:
            kwargs["temperature"] = self.config.temperature
            # Initial max tokens
            kwargs["max_completion_tokens"] = self.config.max_tokens

        async with httpx.AsyncClient() as client:
            while current_turn < max_turns:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    **kwargs,
                )
                
                message = response.choices[0].message
                messages.append(message)
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        tool_result = ""
                        try:
                            if task_config and function_name in tool_implementations:
                                tool_result = await tool_implementations[function_name](arguments, self.config, client)
                            else:
                                tool_result = json.dumps({"result": f"Unknown tool: {function_name}"}, ensure_ascii=False)
                        except Exception as e:
                            tool_result = json.dumps({"result": f"Error executing tool {function_name}: {str(e)}"}, ensure_ascii=False)
                        
                        # Truncate tool result
                        if len(tool_result) > self.config.max_tool_response_length:
                            if self.config.tool_response_truncate_side == "left":
                                tool_result = tool_result[:self.config.max_tool_response_length] + "...(truncated)"
                            elif self.config.tool_response_truncate_side == "right":
                                tool_result = "(truncated)..." + tool_result[-self.config.max_tool_response_length:]
                            else: # center
                                length = self.config.max_tool_response_length // 2
                                tool_result = tool_result[:length] + "...(truncated)..." + tool_result[-length:]

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                    current_turn += 1
                else:
                    content = message.content
                    for tag in ['</thinking>', '</think>']:
                        if tag in content:
                            content = content.split(tag)[-1].strip()
                    
                    # Convert messages to dicts for serialization
                    serializable_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            serializable_messages.append(msg)
                        else:
                            serializable_messages.append(msg.model_dump())
                    return content, serializable_messages
            
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get('content', '')
            else:
                content = last_msg.content
            
            # Convert messages to dicts for serialization
            serializable_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    serializable_messages.append(msg)
                else:
                    serializable_messages.append(msg.model_dump())
            return content, serializable_messages

    async def _generate_single_response(self, instruction: str) -> Tuple[Optional[str], List[Dict]]:
        """Generate a single response for an instruction"""
        try:
            if self.config.use_agent:
                return await self._generate_agent_response(instruction)

            if self.config.use_search:
                input_text = instruction
                assert 'gpt-5' in self.config.model_name, "Web search tool is only supported with gpt-5 models"
                if self.config.system_prompt:
                    input_text = f"{self.config.system_prompt}\n\n{instruction}"
                
                response = await self.client.responses.create(
                    model=self.config.model_name,
                    tools=[{"type": "web_search"}],
                    input=input_text,
                    reasoning={
                        "effort": "low"
                    }
                )
                return response.output_text, [{"role": "user", "content": input_text}, {"role": "assistant", "content": response.output_text}]

            kwargs = {}
            if 'gpt-5' in self.config.model_name:
                kwargs['reasoning_effort'] = 'low'
            else:
                kwargs["temperature"] = self.config.temperature
                kwargs["max_completion_tokens"] = self.config.max_tokens
            
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": instruction})
            
            kwargs = {}
            if 'gpt-5' in self.config.model_name:
                kwargs['reasoning_effort'] = 'low'
            else:
                kwargs["temperature"] = self.config.temperature
                kwargs["max_completion_tokens"] = self.config.max_tokens

            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                **kwargs,
            )
            content = response.choices[0].message.content
            for tag in ['</thinking>', '</think>']:
                if tag in content:
                    content = content.split(tag)[-1].strip()
            
            full_messages = messages + [response.choices[0].message.model_dump()]
            return content, full_messages
        except Exception as e:
            print(f"Error in single generation: {e}")
            return None, []
    
    async def judge_correctness(
        self, 
        instruction: str, 
        generated_response: str, 
        guideline: str
    ) -> tuple[float, str, str]:
        """
        Use LLM to judge if generated response is correct given guideline
        """
        # Extract content after </thinking> tag if present
        response_to_judge = generated_response
        if "</thinking>" in generated_response:
            parts = generated_response.split("</thinking>")
            if len(parts) > 1:
                response_to_judge = parts[-1].strip()

        judge_prompt = f"""You are an expert evaluator. Your task is to evaluate how well a response addresses a given prompt according to specific evaluation criteria.

# Task
Evaluate the response below using a 3-level scoring system:
- **Score 0**: The response is incorrect, irrelevant, or does not address the requirements
- **Score 1**: The response partially addresses the requirements but has significant gaps, errors, or missing information
- **Score 2**: The response fully addresses all requirements correctly and completely

# Evaluation Criteria
{guideline}

# User Prompt
{instruction}

# Response to Evaluate
{response_to_judge}

# Instructions
1. Carefully check if the response meets ALL requirements specified in the evaluation criteria
2. Assign a score of 0, 1, or 2 based on how well it meets the criteria
3. Provide a brief explanation justifying your score
4. Return your evaluation in the following JSON format:

{{
    "score": <0, 1, or 2>,
    "explanation": "<Brief explanation of why you gave this score>"
}}
"""

        try:
            kwargs = {}
            if 'gpt-5' in self.config.judge_model:
                kwargs['reasoning_effort'] = 'low'
            else:
                kwargs["temperature"] = 0.0
                kwargs['max_completion_tokens'] = 4096
            
            response = await self.judge_client.chat.completions.create(
                model=self.config.judge_model,
                messages=[
                    {"role": "user", "content": judge_prompt}
                ],
                **kwargs,
            )
            full_response = response.choices[0].message.content
            if '</think>' in full_response:
                full_response = full_response.split('</think>')[-1].strip()
            if '<response>' in full_response:
                full_response = full_response.split('<response>')[-1].strip().replace('</response>', '').strip()
            result_text = _clean_json(full_response)
            try:
                result = json.loads(result_text)
                score = float(result.get("score", 0)) / 2.0  # Normalize to [0, 1]
                explanation = result.get("explanation", "")
                score = max(0.0, min(1.0, score))
            except json.JSONDecodeError:
                score = 0.0
                explanation = f"Failed to parse JSON: {result_text}"
            
            return (score, explanation, judge_prompt)
        
        except Exception as e:
            print(f"Error in judging: {e}")
            return (0.0, f"Error: {str(e)}", judge_prompt)
    
    def _extract_prompt_only(self, instruction) -> str:
        if isinstance(instruction, list):
            messages = instruction
            return messages[-1]['content']  # Last message
        return str(instruction)
    
    async def score_single_item(self, item: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Score a single instruction item
        """
        instruction = self._extract_prompt_only(item.get('prompt', ''))
        
        # Extract ground truth from reward_model
        reward_model_data = item.get('reward_model', {})
        # Handle if reward_model is a dict or string
        if isinstance(reward_model_data, str):
             # Try to parse if it looks like a dict string, though pandas usually handles it
             pass 
        
        ground_truth = ""
        if isinstance(reward_model_data, dict):
            ground_truth = reward_model_data.get('ground_truth', '')
        elif hasattr(reward_model_data, 'get'): # In case it's some other dict-like object
            ground_truth = reward_model_data.get('ground_truth', '')
            
        if not instruction or not ground_truth:
            print(f"Skipping item {index} with missing instruction or output")
            return None
        
        # Generate responses
        generated_responses, generated_messages = await self.generate_responses(instruction)
        
        if not generated_responses:
            print(f"No valid responses generated for instruction {index}")
            return None
        
        # Judge each response
        judgment_tasks = [
            self.judge_correctness(instruction, response, ground_truth)
            for response in generated_responses
        ]
        
        judgment_results = await asyncio.gather(*judgment_tasks, return_exceptions=True)
        
        # Process judgment results
        judgments = []
        scores = []
        judge_reasonings = []
        judge_prompts = []
        
        for result in judgment_results:
            if isinstance(result, Exception):
                judgments.append(False)
                scores.append(0.0)
                judge_reasonings.append(f"Error: {str(result)}")
                judge_prompts.append("")
            else:
                score, reasoning, prompt = result
                is_correct = (score == 1.0)
                judgments.append(is_correct)
                scores.append(score)
                judge_reasonings.append(reasoning)
                judge_prompts.append(prompt)
        
        # Count correct responses
        n_correct = sum(1 for j in judgments if j is True)
        n_total = len(generated_responses)
        pass_rate = n_correct / n_total if n_total > 0 else 0.0
        
        # Calculate pass@k
        pass_at_1 = calculate_pass_at_k(n_total, n_correct, 1)
        pass_at_n = calculate_pass_at_k(n_total, n_correct, self.config.n_samples)
        
        # Add scoring results to item
        # Convert numpy types to python types for json serialization
        scored_item = {k: v for k, v in item.items() if k != 'reward_model'} # Exclude complex object if needed, or keep it
        scored_item['reward_model_ground_truth'] = ground_truth
        scored_item['index'] = index
        scored_item['generated_responses'] = generated_responses
        scored_item['generated_messages'] = generated_messages
        scored_item['judgments'] = judgments
        scored_item['scores'] = scores
        scored_item['judge_reasonings'] = judge_reasonings
        scored_item['n_correct'] = n_correct
        scored_item['n_total'] = n_total
        scored_item['pass_rate'] = pass_rate
        scored_item['pass_at_1'] = pass_at_1
        scored_item[f'pass_at_{self.config.n_samples}'] = pass_at_n
        
        return scored_item
    
    def save_item(self, item: Dict[str, Any]):
        """Save a single scored item to JSONL file"""
        try:
            item = convert_numpy_types(item)
            with open(self.config.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error saving item: {e}")
    
    async def process_dataset(self, limit: Optional[int] = None):
        """
        Process the entire dataset
        """
        print(f"Loading dataset: {self.config.dataset_path}")
        df = pd.read_parquet(self.config.dataset_path)
        
        if limit:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            df = df.head(limit)
        
        do_remove_system_cnt = 0
        def remove_system_turn(conv):
            if conv[0]['role'] == 'system':
                nonlocal do_remove_system_cnt
                do_remove_system_cnt += 1
                return conv[1:]  # Remove system turn
            return conv
        
        df = df.copy()
        df['prompt'] = df['prompt'].apply(remove_system_turn)
        print('Total items remove system turn:', do_remove_system_cnt)
        print(f"Total items to process: {len(df)}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(row, index):
            if index in self.processed_indices:
                return None
                
            async with semaphore:
                # Convert row to dict
                item = row.to_dict()
                result = await self.score_single_item(item, index)
                if result:
                    self.save_item(result)
                    return result
                return None

        tasks = []
        for index, row in df.iterrows():
            tasks.append(process_with_semaphore(row, index))
            
        # Run tasks with progress bar
        results = []
        for f in tqdm_asyncio.as_completed(tasks):
            result = await f
            if result:
                results.append(result)
                
        # Calculate aggregate metrics
        if results:
            avg_pass_rate = sum(r['pass_rate'] for r in results) / len(results)
            avg_pass_at_1 = sum(r['pass_at_1'] for r in results) / len(results)
            n_samples = self.config.n_samples
            avg_pass_at_n = sum(r[f'pass_at_{n_samples}'] for r in results) / len(results)
            
            # Calculate avg@k (average accuracy for items with k samples)
            avg_at_k_values = {}
            for k in [1, n_samples]:
                items_with_k = [r for r in results if r['n_total'] == k]
                if items_with_k:
                    avg_at_k_values[k] = sum(r['pass_rate'] for r in items_with_k) / len(items_with_k)
                else:
                    avg_at_k_values[k] = 0.0
            
            print("\n" + "="*50)
            print(f"Evaluation Complete")
            print(f"Total items processed: {len(results)}")
            print(f"Average pass_rate (accuracy): {avg_pass_rate:.4f}")
            print(f"pass@1 (prob of >=1 correct in 1 sample): {avg_pass_at_1:.4f}")
            print(f"pass@{n_samples} (prob of >=1 correct in {n_samples} samples): {avg_pass_at_n:.4f}")
            if 1 in avg_at_k_values:
                print(f"avg@1 (average accuracy for single runs): {avg_at_k_values[1]:.4f}")
            if n_samples in avg_at_k_values:
                print(f"avg@{n_samples} (average accuracy for {n_samples} runs): {avg_at_k_values[n_samples]:.4f}")
            print("="*50 + "\n")

async def main():
    parser = argparse.ArgumentParser(description="Evaluate Parquet Dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to parquet dataset")
    parser.add_argument("--output_file", type=str, default="evaluation_report.jsonl", help="Output JSONL file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model for generation")
    parser.add_argument("--judge_model", type=str, default="gpt-5-nano", help="Model for judging")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per instruction")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--max_concurrent", type=int, default=8, help="Max concurrent items")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for generation client")
    parser.add_argument("--judge_base_url", type=str, default=None, help="Base URL for judge client")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to append")
    parser.add_argument("--use_search", action="store_true", help="Use web search tool")
    parser.add_argument("--use_agent", action="store_true", help="Use agent loop")
    parser.add_argument("--agent_url", type=str, default="http://localhost:8932", help="Tool server URL")
    parser.add_argument("--max_tool_response_length", type=int, default=4000, help="Max length for tool response")
    
    args = parser.parse_args()
    
    task = 'nitibench'
    if 'nitibench' in args.dataset_path.lower():
        task = 'nitibench'
    elif 'wangchan' in args.dataset_path.lower():
        task = 'wangchaninstruct'
    
    agent_url = args.agent_url
    if args.agent_url == "http://localhost:8932" and task == 'wangchaninstruct':
         agent_url = "http://localhost:8933"

    config = ScoringConfig(
        dataset_path=args.dataset_path,
        output_file=args.output_file,
        model_name=args.model,
        judge_model=args.judge_model,
        n_samples=args.n_samples,
        max_concurrent=args.max_concurrent,
        base_url=args.base_url,
        judge_base_url=args.judge_base_url,
        system_prompt=args.system_prompt,
        use_search=args.use_search,
        use_agent=args.use_agent,
        agent_url=agent_url,
        max_tool_response_length=args.max_tool_response_length,
        task=task
    )
    if args.n_samples == 1:
        print("Warning: n_samples is set to 1, using temperature=0.0 for deterministic output.")
        config.temperature = 0.0  # Force deterministic if only 1 sample
    
    scorer = ParquetScorer(config)
    await scorer.process_dataset(limit=args.limit)

if __name__ == "__main__":
    asyncio.run(main())
