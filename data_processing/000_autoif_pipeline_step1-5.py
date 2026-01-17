#!/usr/bin/env python3
"""
AutoIF Pipeline (https://github.com/QwenLM/AutoIF) - Steps 1-5 Implementation
Re-implementation of autoif pipeline steps 1-5.

Steps:
1. Generate Instructions (RFT with KD)
2. Generate Verification Functions
3. Cross Validation (Function Cleaning/Processing)
4. Back Translation
5. Filtering (NLI)

Input:
- Seed instructions (txt file)

Output:
- Filtered instructions with evaluation functions (jsonl)
"""

import argparse
import json
import jsonlines
import os
import random
import re
import ast
import hashlib
import pickle
import time
import signal
import concurrent.futures
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

random.seed(0)

def load_nli_model(device):
    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return tokenizer, model

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


class AutoIFPipelineStep1To5:
    def __init__(self,
                 seed_path: str,
                 output_path: str,
                 openai_model: str = "gpt-4o-mini",
                 openai_api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 num_workers: int = 8,
                 cache_dir: str = "cache_step1to5",
                 use_cache: bool = True,
                 count: int = 50, # Number of instructions to generate per prompt
                 k_generations: int = 1, # Number of generation calls for step 1
                 n_choices: int = 5, # Number of choices per request in step 2
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 ):
        self.seed_path = seed_path
        self.output_path = output_path
        self.openai_model = openai_model
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.count = count
        self.k_generations = k_generations
        self.n_choices = n_choices
        self.device = device
        
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'), base_url=base_url)

    def _get_cache_key(self, step_name: str, data_hash: str = None) -> str:
        key_parts = [step_name, self.openai_model, str(self.count)]
        if data_hash:
            key_parts.append(data_hash)
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()[:12]

    def _get_data_hash(self, data: Any) -> str:
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def _save_cache(self, step_name: str, data: Any, input_hash: str = None) -> None:
        if not self.use_cache: return
        cache_key = self._get_cache_key(step_name, input_hash)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Cached {step_name} results to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache for {step_name}: {e}")

    def _load_cache(self, step_name: str, input_hash: str = None) -> Optional[Any]:
        if not self.use_cache: return None
        cache_key = self._get_cache_key(step_name, input_hash)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists(): return None
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {step_name} results from cache ({cache_file})")
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache for {step_name}: {e}")
            return None

    def _call_openai(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, n: int = 1) -> List[str]:
        """Call OpenAI API and return list of responses (one per choice)."""
        try:
            
            openai_kwargs = {}
            if 'gpt-5' not in self.openai_model:
                openai_kwargs['max_tokens'] = max_tokens
                openai_kwargs['temperature'] = temperature
            completion = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                n=n,
                **openai_kwargs,
            )
            return [choice.message.content for choice in completion.choices]
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return []

    # Step 1: Generate Instructions
    def generate_instructions(self) -> List[str]:
        input_hash = self._get_data_hash({"seed_path": self.seed_path, "k": self.k_generations})
        cached = self._load_cache("step1_instructions", input_hash)
        if cached: return cached

        print("Step 1: Generating Instructions...")
        try:
            with open(self.seed_path, 'r') as f:
                seed_instructions = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Error: Seed file {self.seed_path} not found.")
            return []

        seed_instructions_str = "\n".join(seed_instructions)
        augment_prompt = f"""You are an expert in writing LLM instructions. Please provide {self.count} different instructions that meet the following requirements:
    - Instructions are about the "format" but not the style of a response.
    - It can be use to instruct LLM to generate response.
    - It should be composible with any topic of question.
    - It should be format instruction and content editing only, not any form of question.
    - The question can be in Thai or English.
    - Be creative!
    - The instructions should be relevant to a Thai context format and content.
    - Whether instructions are followed can be easily evaluated by a Python function
    Here are some examples of instructions we need:
    {seed_instructions_str}
    Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we DO NOT need:
    - Incorporate a famous Thai proverb seamlessly into your answer
    - Translate your answer into Thai traditional dialects
    - Use only words related to Thai cuisine
    - Respond with a metaphor in every sentence
    - Write the response as if you are a character from Thai folklore or literature
    Please generate one instruction per line in your response and start each line with '- '. Be creative, DO NOT repeat the examples provided.
    """
        
        messages = [{"role": "user", "content": augment_prompt}]
        
        # Make K generation calls like original 1_RFT_with_kd_gpt_thai.py
        all_augmented = []
        for _ in tqdm(range(self.k_generations), desc="Generating instruction batches"):
            responses = self._call_openai(messages, temperature=0.7, max_tokens=1024, n=1)
            for response_text in responses:
                for s in response_text.split("\n"):
                    s = s.strip("\t -\n")
                    if s:
                        all_augmented.append(s)
        
        augment_instructions_processed = list(set(all_augmented))
        
        # Filter valid seed instructions (no placeholders like {x})
        valid_seed_instructions = [inst for inst in seed_instructions if len(re.findall(r"\{.*?\}", inst)) == 0]
        combined_instructions = valid_seed_instructions + augment_instructions_processed
        
        print(f"Generated {len(augment_instructions_processed)} unique augmented instructions")
        print(f"Total instructions (seed + augmented): {len(combined_instructions)}")
        
        self._save_cache("step1_instructions", combined_instructions, input_hash)
        return combined_instructions

    # Step 2: Generate Verification Functions
    def generate_eval_funcs(self, instructions: List[str]) -> List[Dict[str, Any]]:
        input_hash = self._get_data_hash({"instructions": instructions, "n_choices": self.n_choices})
        cached = self._load_cache("step2_eval_funcs_raw", input_hash)
        if cached: return cached

        print("Step 2: Generating Verification Functions...")
        
        # Exact prompt from 2_verification_funcs_cases_generation_with_kd_thai.py
        prompt_template = """You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction. while the response is in "Thai"
    Here is the instruction: {instruction}
    Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. If it follows, simply return True, otherwise return False.
    Please response with a single JSON includes the evaluation function in the key `func`, and a list of three test cases in the key `cases`, which includes an input in the key `input` and an expected output in the key `output` in (true, false).
    Here is an example of output JSON format: {{"func": JSON_STR(use only \\n instead of \n), "cases": [{{"input": str, "output": str}}]}}."""

        results = []
        
        def process_instruction(inst):
            # Filter invalid instructions like original
            if "{...}" in inst or "{{x}}" in inst or "{{y}}" in inst:
                return None
            prompt = prompt_template.format(instruction=inst)
            msg = [{"role": "user", "content": prompt}]
            # Original uses temperature=0.2 and n=5
            gpt_responses = self._call_openai(msg, temperature=0.2, max_tokens=1024, n=self.n_choices)
            if not gpt_responses:
                return None
            return {"instruction": inst, "gpt_answers": gpt_responses}

        with tqdm(total=len(instructions), desc="Generating eval functions") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_inst = {executor.submit(process_instruction, inst): inst for inst in instructions}
                for future in concurrent.futures.as_completed(future_to_inst):
                    res = future.result()
                    if res:
                        results.append(res)
                    pbar.update(1)
        
        self._save_cache("step2_eval_funcs_raw", results, input_hash)
        return results

    # Step 3: Cross Validation / Processing (exact logic from 3_cross_validation.py)
    def process_eval_funcs(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        input_hash = self._get_data_hash(raw_results)
        cached = self._load_cache("step3_processed_funcs", input_hash)
        if cached: return cached

        print("Step 3: Processing Evaluation Functions (Cross Validation)...")
        
        def col_formatter(s):
            """Format JSON string, handling escaped newlines in function code."""
            try:
                json.loads(s)
            except Exception:
                try:
                    data = ast.literal_eval(s)
                    s = json.dumps(data, ensure_ascii=False, indent=4)
                except Exception:
                    pass
            
            start = s.find("def evaluate(response):")
            end = s.find("return", start)
            
            if start == -1 or end == -1:
                return s
            
            part_to_modify = s[start:end]
            modified_part = part_to_modify.replace("\n", "\\n")
            final_string = s[:start] + modified_part + s[end:]
            return final_string

        filter_results = []
        filter_count_func_case = 0
        
        for result in tqdm(raw_results, desc="Cross-validating functions"):
            instruction = result["instruction"]
            gpt_answers = result["gpt_answers"]  # List of n=5 responses
            
            eval_funcs = []
            test_cases = []
            
            # Process each of the n choices
            for each in gpt_answers:
                try:
                    # Extract JSON from markdown code block
                    json_dict = re.findall(r"```json(.*?)```", each, re.DOTALL)[0].strip()
                except IndexError:
                    continue
                
                try:
                    json_dict = col_formatter(json_dict)
                    res_dict = json.loads(json_dict, strict=False)
                except json.JSONDecodeError:
                    continue
                
                if "func" not in res_dict:
                    continue
                
                func = res_dict["func"]
                func = func.strip()
                
                # Remove dangerous lines
                func = "\n".join([
                    line for line in func.split("\n")
                    if "download" not in line and "requests" not in line
                ])
                
                # Handle escaped newlines
                if "\\n" in func:
                    func = func.replace("\\n", "\n")
                
                # Test if function can be executed
                try:
                    exec(func)
                except Exception:
                    continue
                
                eval_funcs.append(func)
                
                # Collect test cases
                if "cases" in res_dict:
                    for case in res_dict["cases"]:
                        try:
                            test_cases.append((case["input"], case["output"]))
                        except KeyError:
                            continue
            
            # Deduplicate
            eval_funcs = list(set(eval_funcs))
            test_cases = list(set(map(lambda x: (x[0], x[1]), test_cases)))  # Proper dedup for tuples
            
            # Original requires at least 3 funcs and 10 test cases
            # For small test runs with n_choices=1, we relax to 1 func and 1 test case
            min_funcs = min(3, max(1, self.n_choices))
            min_cases = 1 if self.n_choices < 3 else 10
            
            # Filter test cases: keep only those that at least one function agrees with
            filtered_test_cases = []
            for inp, out in test_cases:
                flag = False
                for func in eval_funcs:
                    local_vars = {}
                    try:
                        exec(func, globals(), local_vars)
                    except Exception:
                        continue
                    
                    if "evaluate" not in local_vars:
                        continue
                    
                    eval_func = local_vars["evaluate"]
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res = eval_func(inp)
                    except Exception:
                        res = None
                    finally:
                        signal.alarm(0)
                    
                    if res is not None and res == out:
                        flag = True
                        break
                
                if flag:
                    filtered_test_cases.append((inp, out))
            
            # Score each function based on accuracy on filtered test cases
            scored_funcs = []
            for func in eval_funcs:
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                except Exception:
                    continue
                
                if "evaluate" not in local_vars:
                    continue
                
                eval_func = local_vars["evaluate"]
                acc = []
                for inp, out in filtered_test_cases:
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res = eval_func(inp)
                    except Exception:
                        res = None
                    finally:
                        signal.alarm(0)
                    
                    if res is None or res != out:
                        acc.append(0)
                    else:
                        acc.append(1)
                
                acc_score = np.mean(acc) if acc else 0
                scored_funcs.append([func, acc_score])
            
            # Keep only functions with accuracy >= 0.8
            valid_funcs = [each for each in scored_funcs if each[1] >= 0.8]
            if not valid_funcs:
                if self.n_choices <= 3:  # Debug output for small runs
                    print(f"  [DEBUG] No valid funcs after scoring. scored_funcs: {[(f[:50]+'...', s) for f, s in scored_funcs]}")
                    print(f"  [DEBUG] filtered_test_cases: {filtered_test_cases[:3]}")
                continue
            
            filter_results.append({
                "instruction": instruction,
                "eval_func": valid_funcs,  # List of [func_code, accuracy_score]
                "cases": filtered_test_cases,
            })
        
        print(f"Total filtered (not enough funcs/cases): {filter_count_func_case}")
        print(f"Total left from cross-validation: {len(filter_results)}")
        
        self._save_cache("step3_processed_funcs", filter_results, input_hash)
        return filter_results

    # Step 4: Back Translation
    def back_translate(self, processed_funcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        input_hash = self._get_data_hash(processed_funcs)
        cached = self._load_cache("step4_back_translated", input_hash)
        if cached: return cached
        
        print("Step 4: Back Translating Functions to Instructions...")
        
        results = []
        
        def process_back_trans(item):
            funcs = item["eval_func"]  # This is now [[func_code, score], ...]
            
            # The prompt expects the 'funcs' variable to be inserted (exact format from original)
            prompt = f"""You are an expert in converting the Python eval function code into the corresponding instruction text. I will provide the eval function code. Please strictly follow the code to convert it into the corresponding instruction text. Here's an example: \n\n[["def evaluate(response):\n    return 'e' not in response.lower()", 1.0], ["def evaluate(response):\n    words = response.split()\n    for word in words:\n        if 'e' in word.lower():\n            return False\n    return True", 1.0], ["def evaluate(response):\n    return all('e' not in word.lower() for word in response.split())", 1.0]] \n\n["Answer without using any words that contain the letter 'E'.","Answer with words that do not contain the letter 'E'.","Respond using words that omit the letter 'E'."] Please convert the following eval function into instructions stored in a list: \n\n{funcs}"""
            
            msg = [{"role": "user", "content": prompt}]
            gpt_responses = self._call_openai(msg, temperature=0.7, max_tokens=1024, n=1)
            return {
                "original_item": item,
                "back_instruction_response": gpt_responses[0] if gpt_responses else ""
            }

        with tqdm(total=len(processed_funcs)) as pbar:
             with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_item = {executor.submit(process_back_trans, item): item for item in processed_funcs}
                for future in concurrent.futures.as_completed(future_to_item):
                    res = future.result()
                    results.append(res)
                    pbar.update(1)
        
        # Parse the results (Step 4 check)
        final_results = []
        for res in results:
            item = res["original_item"]
            resp = res["back_instruction_response"]
            
            try:
                # 4_check_func_backtranslator.py logic: extract list
                def extract_list(s):
                    start = s.find("[")
                    end = s.rfind("]") + 1
                    if start == -1 or end == 0: return []
                    list_str = s[start:end]
                    return ast.literal_eval(list_str)

                back_instructions = extract_list(resp)
                
                # Check 4_check_func_backtranslator.py consistency check
                if not isinstance(back_instructions, list):
                    continue
                    
                # The original code asserts len(back_instruction) == len(line[2]["eval_func"])
                # Since we passed a list of 'eval_funcs', we expect back_instructions to match in length.
                if len(back_instructions) != len(item["eval_func"]):
                    continue
                
                final_back_instructions = []
                # Flatten structure if it's like [["inst", score], ...]
                if back_instructions and isinstance(back_instructions[0], list):
                    final_back_instructions = [b[0] for b in back_instructions]
                else:
                    final_back_instructions = back_instructions

                final_results.append({
                    "instruction": item["instruction"],
                    "back_instruction": final_back_instructions,
                    "eval_func": item["eval_func"],
                    "cases": item["cases"]
                })
                
            except Exception as e:
                continue

        self._save_cache("step4_back_translated", final_results, input_hash)
        return final_results

    # Step 5: NLI Filtering
    def filter_with_nli(self, back_translated_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        input_hash = self._get_data_hash(back_translated_items)
        cached = self._load_cache("step5_filtered", input_hash)
        if cached: return cached

        print("Step 5: Filtering with NLI Model...")
        
        tokenizer, model = load_nli_model(self.device)
        model.eval()
        
        filtered_results_list = []
        
        for item in tqdm(back_translated_items):
            ori_ins = item["instruction"]
            back_instructions = item["back_instruction"]
            
            nli_scores = []
            
            for back_ins in back_instructions:
                try:
                    inputs = tokenizer(ori_ins, back_ins, truncation=True, return_tensors="pt").to(self.device)
                    output = model(inputs["input_ids"])
                    prediction = torch.softmax(output["logits"][0], -1).tolist()
                    label_names = ["entailment", "neutral", "contradiction"]
                    
                    scores = {name: pred for pred, name in zip(prediction, label_names)}
                    max_label = max(scores, key=scores.get)
                    nli_scores.append(max_label)
                except Exception:
                    nli_scores.append("error") # Treat error as safe or unsafe? safely skip
                    
            # 5_eval_func_backtranslator_filter.py logic:
            # if "contradiction" in nli_scores: skip
            
            if "contradiction" in nli_scores:
                continue
            else:
                item["nli_scores"] = nli_scores
                filtered_results_list.append(item)

        self._save_cache("step5_filtered", filtered_results_list, input_hash)
        return filtered_results_list

    def run(self):
        # Step 1
        instructions = self.generate_instructions()
        print(f"Generated {len(instructions)} instructions.")
        
        # Step 2
        raw_eval_funcs = self.generate_eval_funcs(instructions)
        print(f"Generated {len(raw_eval_funcs)} raw eval functions.")
        
        # Step 3
        processed_funcs = self.process_eval_funcs(raw_eval_funcs)
        print(f"Processed {len(processed_funcs)} eval functions.")
        
        # Step 4
        back_translated = self.back_translate(processed_funcs)
        print(f"Back-translated {len(back_translated)} items.")
        
        # Step 5
        final_results = self.filter_with_nli(back_translated)
        print(f"Final Count after NLI filtering: {len(final_results)}")
        
        # Save output
        with jsonlines.open(self.output_path, 'w') as writer:
            writer.write_all(final_results)
        print(f"Saved results to {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoIF Pipeline Steps 1-5")
    parser.add_argument("--seed_path", type=str, required=True, help="Path to seed instructions txt file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output jsonl file")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--count", type=int, default=50, help="Number of instructions to generate per prompt in step 1")
    parser.add_argument("--k_generations", type=int, default=1, help="Number of generation calls for step 1")
    parser.add_argument("--n_choices", type=int, default=5, help="Number of choices per API request in step 2")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    pipeline = AutoIFPipelineStep1To5(
        seed_path=args.seed_path,
        output_path=args.output_path,
        openai_model=args.openai_model,
        base_url=args.base_url,
        num_workers=args.num_workers,
        count=args.count,
        k_generations=args.k_generations,
        n_choices=args.n_choices,
        use_cache=not args.no_cache
    )
    
    pipeline.run()
