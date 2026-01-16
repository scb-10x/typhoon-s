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

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from openai import OpenAI

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


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

@register("llm_judge")
class LLMJudgeRewardManager(AbstractRewardManager):
    """Reward manager using LLM as a judge with guideline-based evaluation and format checking."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        llm_judge_api_key=None,
        llm_judge_base_url=None,
        llm_judge_model="gpt-4o-mini",
        max_concurrent=10,
        format_reward_weight=0.1,
        guideline_reward_weight=0.9,
        max_resp_len=None,
        overlong_buffer_cfg=None,
        score_type="3-level",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LLMJudgeRewardManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of samples to print for debugging.
            compute_score: Optional custom scoring function (not used here).
            reward_fn_key: Key for accessing data source.
            llm_judge_api_key: API key for LLM judge.
            llm_judge_base_url: Base URL for LLM judge API.
            llm_judge_model: Model name for LLM judge.
            max_concurrent: Maximum concurrent API calls.
            format_reward_weight: Weight for format reward (default 0.1).
            guideline_reward_weight: Weight for guideline reward (default 0.9).
            max_resp_len: Maximum response length.
            overlong_buffer_cfg: Configuration for overlong buffer.
            score_type: Type of scoring system ("3-level" or "binary").
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # LLM judge configuration
        self.llm_judge_api_key = llm_judge_api_key
        self.llm_judge_base_url = llm_judge_base_url or "https://api.openai.com/v1"
        self.llm_judge_model = llm_judge_model
        self.max_concurrent = max_concurrent
        self.score_type = score_type

        # Reward weights
        self.format_reward_weight = format_reward_weight
        self.guideline_reward_weight = guideline_reward_weight

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

        # Initialize sync client
        self.client = OpenAI(api_key=self.llm_judge_api_key, base_url=self.llm_judge_base_url)

    def check_format(self, response: str, expected_format: dict) -> tuple[float, dict]:
        """
        Check if response matches expected format.

        Args:
            response: The response string to check.
            expected_format: Dictionary specifying format requirements.
                Example: {"type": "think_tag"}  # Check for <think>...</think> format

        Returns:
            Tuple of (format_score, format_details)
        """
        format_details = {}

        if not expected_format:
            return 1.0, {"format_check": "no_format_required"}

        format_type = expected_format.get("type", "think_tag")

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
                return 0.0, format_details

            intermediate_matches = re.findall(INTERMEDIATE_PATTERN, response, re.DOTALL)
            
            # Check all conditions:
            # 1. Exactly one <thinking>...</thinking> pair
            # 2. Response starts with <thinking>
            # 3. Exactly one opening and closing tag each
            if (
                len(intermediate_matches) == 1
                and response.strip().startswith("<thinking>")
                and response.count("<thinking>") == 1
                and response.count("</thinking>") == 1
            ):
                format_details["valid_think_format"] = True
                return 0.5, format_details  # Return 0.5 as per reference implementation
            else:
                format_details["valid_think_format"] = False
                return 0.0, format_details

        else:
            raise NotImplementedError(f"Format type '{format_type}' not implemented.")

        return 1.0, {"format_check": "unknown_type"}

    def call_llm_judge(self, prompt: str, response: str, guideline: str) -> tuple[float, dict]:
        """
        Call LLM judge API to evaluate response based on guideline.
        If response contains <think>...</think>, only evaluate the content after </think>.

        Args:
            prompt: The original prompt.
            response: The model's response.
            guideline: Evaluation guideline.

        Returns:
            Tuple of (score, evaluation_details)
        """
        # Extract content after </thinking> tag if present
        response_to_judge = response
        has_think_tag = "</thinking>" in response
        
        if has_think_tag:
            # Only evaluate the content after </thinking>
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

        try:
            kwargs = {}
            if 'gpt-5' in self.llm_judge_model:
                kwargs['reasoning_effort'] = 'low'
            else:
                kwargs["temperature"] = 0.0
            
            completion = self.client.chat.completions.create(
                model=self.llm_judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                **kwargs,
            )

            result_text = _clean_json(completion.choices[0].message.content)
            result = json.loads(result_text)

            raw_score = float(result.get("score", 0))
            if self.score_type == "binary":
                score = raw_score  # Already [0, 1]
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

    def evaluate_batch(self, data: DataProto) -> tuple[torch.Tensor, dict]:
        """
        Evaluate a batch of responses using ThreadPoolExecutor.

        Args:
            data: DataProto containing prompts and responses.

        Returns:
            Tuple of (reward_tensor, reward_extra_info)
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        tasks = []
        indices = []

        # Prepare tasks
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Get evaluation metadata
            reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
            guideline = reward_model_info.get("guideline")
            if not guideline:
                guideline = reward_model_info.get("ground_truth", None)
            if guideline is None:
                print('Warning: No guideline found in reward_model info; using default.')
                guideline = "Evaluate the quality and correctness of the response."
            expected_format = reward_model_info.get("expected_format", {"type": "think_tag"})

            tasks.append((prompt_str, response_str, guideline, expected_format, valid_response_length))
            indices.append(i)

        # Execute tasks concurrently
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            results = list(executor.map(lambda args: self._evaluate_single(*args), tasks))

        # Process results
        for i, (idx, (total_score, details)) in enumerate(zip(indices, results)):
            valid_response_length = details.pop("valid_response_length")
            reward_tensor[idx, valid_response_length - 1] = total_score

            # Store details
            for key, value in details.items():
                reward_extra_info[key].append(value)

            # Print examination samples
            if i < self.num_examine:
                print(f"\n=== Sample {i + 1} ===")
                prompt_str = tasks[i][0]
                response_str = tasks[i][1]
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:200]}...{response_str[-200:]}")
                print(f"[score] {total_score:.3f}")
                for key, value in details.items():
                    if key != "valid_response_length":
                        print(f"[{key}] {value}")

        return reward_tensor, dict(reward_extra_info)

    def _evaluate_single(
        self, prompt: str, response: str, guideline: str, expected_format: dict, valid_response_length: int
    ) -> tuple[float, dict]:
        """
        Evaluate a single response.

        Args:
            prompt: The prompt string.
            response: The response string.
            guideline: Evaluation guideline.
            expected_format: Expected format specification.
            valid_response_length: Length of valid response tokens.

        Returns:
            Tuple of (total_score, details)
        """
        details = {"valid_response_length": valid_response_length}

        # Check format
        format_score, format_details = self.check_format(response, expected_format)
        details["format_score"] = format_score

        # Call LLM judge for guideline-based evaluation
        guideline_score, judge_details = self.call_llm_judge(prompt, response, guideline)
        details["guideline_score"] = guideline_score

        # Compute total score
        total_score = (
            format_score * self.format_reward_weight + guideline_score * self.guideline_reward_weight
        )

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            # Ensure valid_response_length is a number
            if isinstance(valid_response_length, torch.Tensor):
                val_len = valid_response_length.item()
            else:
                val_len = valid_response_length

            exceed_len = val_len - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            total_score += overlong_reward
            if self.overlong_buffer_cfg.log:
                details["overlong_reward"] = overlong_reward
                details["overlong"] = overlong_reward < 0

        details["total_score"] = total_score

        return total_score, details

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Evaluate the data batch.

        Args:
            data: DataProto containing prompts and responses.
            return_dict: Whether to return dictionary with extra info.

        Returns:
            Reward tensor or dict with reward_tensor and reward_extra_info.
        """
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Run evaluation using thread pool
        reward_tensor, reward_extra_info = self.evaluate_batch(data)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
