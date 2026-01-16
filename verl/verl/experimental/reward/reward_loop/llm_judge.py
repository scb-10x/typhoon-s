# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import re
from typing import Any

import numpy as np
import torch
from openai import AsyncOpenAI

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


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


def _extract_last_assistant_turn(response_ids: torch.Tensor, tokenizer, response_mask: torch.Tensor) -> str:
    """Extract only the last assistant turn from a multi-turn response using the response mask.
    
    The response_mask indicates which tokens belong to assistant turns (1) vs system/tool/other (0).
    This function finds the last contiguous segment of 1s and decodes only that portion.
    
    If response_mask is None or invalid, it falls back to string splitting using common assistant headers.
    
    Args:
        response_ids: Token IDs for the full response
        tokenizer: Tokenizer for decoding
        response_mask: Binary mask where 1 = assistant token, 0 = other
        
    Returns:
        Decoded string of only the last assistant turn
    """
    # Try using response_mask if available
    if response_mask is not None and response_mask.sum() > 0:
        # Find all segments where mask is 1
        mask_np = response_mask.cpu().numpy() if isinstance(response_mask, torch.Tensor) else response_mask
        
        # Find transitions: 0->1 (start) and 1->0 (end)
        padded = np.concatenate([[0], mask_np, [0]])
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]  # Indices where assistant turn begins
        ends = np.where(diff == -1)[0]   # Indices where assistant turn ends
        
        if len(starts) > 0 and len(ends) > 0:
            # Take the last segment
            last_start = starts[-1]
            last_end = ends[-1]
            
            last_assistant_ids = response_ids[last_start:last_end]
            return tokenizer.decode(last_assistant_ids, skip_special_tokens=True)

    # Fallback: string splitting if no mask or mask processing failed
    # Decode with special tokens to see the headers
    full_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    
    # Common assistant markers (ordered by specificity)
    markers = [
        "<|start_header_id|>assistant<|end_header_id|>", # Llama 3
        "<|im_start|>assistant", # ChatML (Qwen, etc.)
        "<|Assistant|>", # DeepSeek sometimes
        "### Response:", # Alpaca/DeepSeek-Coder
        "Assistant:", # Generic
        "[/INST]", # Llama 2
    ]
    
    last_marker_pos = -1
    used_marker = ""
    
    for marker in markers:
        pos = full_text.rfind(marker)
        if pos > last_marker_pos:
            last_marker_pos = pos
            used_marker = marker
    
    if last_marker_pos != -1:
        content = full_text[last_marker_pos + len(used_marker):]
        
        # Clean up common EOS tokens from the end
        eos_tokens = ["<|im_end|>", "<|eot_id|>", "</s>"]
        for eos in eos_tokens:
            if content.strip().endswith(eos):
                content = content.strip()[:-len(eos)]
        
        return content.strip()
        
    return tokenizer.decode(response_ids, skip_special_tokens=True)


@register("llm_judge")
class LLMJudgeRewardLoopManager(RewardLoopManagerBase):
    """Reward manager using LLM as a judge with guideline-based evaluation and format checking."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        
        self.llm_judge_api_key = reward_kwargs.get("llm_judge_api_key")
        self.llm_judge_base_url = reward_kwargs.get("llm_judge_base_url", "https://api.openai.com/v1")
        self.llm_judge_model = reward_kwargs.get("llm_judge_model", "gpt-4o-mini")
        self.max_concurrent = reward_kwargs.get("max_concurrent", 64)
        self.format_reward_weight = reward_kwargs.get("format_reward_weight", 0.1)
        self.guideline_reward_weight = reward_kwargs.get("guideline_reward_weight", 0.9)
        self.score_type = reward_kwargs.get("score_type", "3-level")

        # DAPO Reward Config
        overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = reward_kwargs.get("max_resp_len", None)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.llm_judge_api_key, base_url=self.llm_judge_base_url)

        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def check_format(self, response: str, expected_format: dict) -> tuple[float, dict]:
        """
        Check if response matches expected format.
        """
        format_details = {}

        if not expected_format:
            return 1.0, {"format_check": "no_format_required"}

        format_type = expected_format.get("type", "none")

        if format_type == "none":
            return 0.0, {"format_check": "no_format_required"}
        elif format_type == "think_tag":
            """
            Score the response based on <thinking>...</thinking> formatting.
            Returns 0.5 if exactly one <thinking>...</thinking> pair is present, starts with <thinking>,
            and has proper structure, otherwise returns 0.0.
            """
            INTERMEDIATE_PATTERN = r"<thinking>.*?</thinking>"
            
            if response is None:
                format_details["valid_think_format"] = False
                format_details["has_think_tags"] = False
                return 0.0, format_details

            intermediate_matches = re.findall(INTERMEDIATE_PATTERN, response, re.DOTALL)
            
            if (
                len(intermediate_matches) == 1
                and response.strip().startswith("<thinking>")
                and response.count("<thinking>") == 1
                and response.count("</thinking>") == 1
            ):
                format_details["valid_think_format"] = True
                format_details["has_think_tags"] = True
                return 0.5, format_details
            else:
                format_details["valid_think_format"] = False
                format_details["has_think_tags"] = False
                return 0.0, format_details

        raise NotImplementedError(f"Format type '{format_type}' not implemented.")

    async def call_llm_judge(self, prompt: str, response: str, guideline: str) -> tuple[float, dict]:
        """
        Call LLM judge API to evaluate response based on guideline.
        
        Note: This method now receives the already-extracted response (last assistant turn).
        We keep the </thinking> tag extraction logic for backward compatibility.
        """
        # Extract content after </thinking> tag if present
        response_to_judge = response
        has_think_tag = "</thinking>" in response
        
        if has_think_tag:
            parts = response.split("</thinking>")
            if len(parts) > 1:
                response_to_judge = parts[-1].strip()
            
        if self.score_type == "binary":
            score_prompt = """Evaluate the response below using a binary scoring system:
- **Score 0**: The response is incorrect, irrelevant, or does not address the requirements
- **Score 1**: The response fully addresses all requirements correctly and completely"""
            score_options = "0 or 1"
        else:
            score_prompt = """Evaluate the response below using a 3-level scoring system:
- **Score 0**: The response is incorrect, irrelevant, or does not address the requirements
- **Score 1**: The response partially addresses the requirements but has significant gaps, errors, or missing information
- **Score 2**: The response fully addresses all requirements correctly and completely"""
            score_options = "0, 1, or 2"

        judge_prompt = f"""You are an expert evaluator. Your task is to evaluate how well a response addresses a given prompt according to specific evaluation criteria.

# Task
{score_prompt}

# Evaluation Criteria
{guideline}

# User Prompt
{prompt}

# Response to Evaluate
{response_to_judge}

# Instructions
1. Carefully check if the response meets ALL requirements specified in the evaluation criteria
2. Assign a score of {score_options} based on how well it meets the criteria
3. Provide a brief explanation justifying your score
4. Return your evaluation in the following JSON format:

{{
    "score": <{score_options}>,
    "explanation": "<Brief explanation of why you gave this score>"
}}
"""

        async with self.semaphore:
            kwargs = {}
            if 'gpt-5' in self.llm_judge_model:
                kwargs['reasoning_effort'] = 'low'
            else:
                kwargs["temperature"] = 0.0
            try:
                completion = await self.client.chat.completions.create(
                    model=self.llm_judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    **kwargs,
                )

                result_text =  _clean_json(completion.choices[0].message.content)

                result = json.loads(result_text)

                raw_score = float(result.get("score", 0.0))
                if self.score_type == "binary":
                    score = raw_score
                else:
                    score = raw_score / 2.0  # Normalize to [0, 1]
                
                explanation = result.get("explanation", "")

                # Clip score to [0, 1]
                score = max(0.0, min(1.0, score))

                return score, {
                    "llm_explanation": explanation,
                    "raw_llm_response": result_text,
                    "judged_after_think": has_think_tag,
                    "judge_prompt": judge_prompt,
                }

            except Exception as e:
                print(f"Error calling LLM judge: {e}")
                return 0.0, {"error": str(e), "judge_prompt": judge_prompt}

    async def run_single(self, data: DataProto) -> dict:
        """
        Evaluate a single data item.
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Get response mask to identify assistant-only tokens
        response_mask = data_item.batch.get("response_mask")
        if response_mask is not None:
            response_mask = response_mask[:valid_response_length]

        # Decode prompt
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        
        # Decode full response (for format checking)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        
        # Extract only last assistant turn for LLM judge evaluation
        last_assistant_str = _extract_last_assistant_turn(valid_response_ids, self.tokenizer, response_mask)

        # Get evaluation metadata
        reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
        guideline = reward_model_info.get("guideline")
        if not guideline:
            guideline = reward_model_info.get("ground_truth", None)
        if guideline is None:
            print('Warning: No guideline found in reward_model info; using default.')
            guideline = "Evaluate the quality and correctness of the response."
        expected_format = reward_model_info.get("expected_format", {"type": "none"})

        details = {}
        # Check format on full response
        format_score, format_details = self.check_format(response_str, expected_format)
        details["format_score"] = format_score
        # print('eval', json.dumps({
        #     "prompt_str": prompt_str, 
        #     "last_assistant_str": last_assistant_str
        # }, ensure_ascii=False))
        # if False:
        #     prompt_str_full = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
        #     response_str_full = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        #     with open('lm_judge.jsonl', 'a') as w:
        #         w.write(f"{json.dumps({'prompt': prompt_str_full, 'response': response_str_full, 'last_assistant': last_assistant_str, 'guideline': guideline}, ensure_ascii=False)}\n")
        # Call LLM judge for guideline-based evaluation on LAST assistant turn only
        guideline_score, judge_details = await self.call_llm_judge(prompt_str, last_assistant_str, guideline)
        details["guideline_score"] = guideline_score

        # Compute total score
        total_score = (
            format_score * self.format_reward_weight + guideline_score * self.guideline_reward_weight
        )

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            val_len = valid_response_length.item()
            exceed_len = val_len - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            total_score += overlong_reward
            if self.overlong_buffer_cfg.log:
                details["overlong_reward"] = overlong_reward
                details["overlong"] = overlong_reward < 0

        details["total_score"] = total_score

        # Construct return dictionary
        return {
            "reward_score": total_score,
            "reward_extra_info": details
        }
