#!/usr/bin/env python3
"""
AutoIF Pipeline (https://github.com/QwenLM/AutoIF) - Steps 6-9 Implementation
Re-implementation of autoif pipeline steps 6-9.

Input: 
- back_trans_filter.jsonl (constraint instructions with evaluation functions)
- question_prompts (List[str]) - instruction queries to combine with constraints

Output:
- instruction dataset in jsonl format: {'conversations': [{'role': '...', 'content': '...'}, ...]}
"""

import argparse
import json
import jsonlines
import argparse
from pathlib import Path
import hashlib
import pickle
import copy
import random
import re
import os
import signal
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import TimeoutError
import requests
import fasttext
from huggingface_hub import hf_hub_download
import concurrent.futures
from openai import OpenAI
from transformers import set_seed
import time

def load_language_detector():
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", filename="model.bin"
    )
    return fasttext.load_model(model_path)

def detect_lang(ld_model, text: str) -> str:
    line = " ".join(text.split("\n"))
    lang = ld_model.predict(line)[0][0]
    lang = lang.replace("__label__", "")
    return lang

# Utility functions from query_vertification_utils.py
def count_chinese_chars(text: str) -> int:
    """Count the number of Chinese characters in a string."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
        
    chinese_ranges = [
        ('\u4e00', '\u9fff'),     # CJK Unified Ideographs
        ('\u3400', '\u4dbf'),     # Extension A
        ('\U00020000', '\U0002a6df'),  # Extension B
        ('\U0002a700', '\U0002b73f'),  # Extension C
        ('\U0002b740', '\U0002b81f'),  # Extension D
        ('\U0002b820', '\U0002ceaf'),  # Extension E
        ('\U0002ceb0', '\U0002ebef'),  # Extension F
        ('\U00030000', '\U0003134f'),  # Extension G
        ('\U00031350', '\U00031427'),  # Extension H
        ('\uf900', '\ufaff'),     # CJK Compatibility Ideographs
        ('\U0002f800', '\U0002fa1f')   # CJK Compatibility Supplement
    ]
    
    count = 0
    for char in text:
        if any(start <= char <= end for start, end in chinese_ranges):
            count += 1  
            
    return count

def timeout_handler(signum, frame):
    """Handle timeout for evaluation function execution."""
    raise TimeoutError("Function execution timed out")

class InstructionTranslator:
    """Handles translation between English and Thai for instructions."""
    
    def __init__(self, openai_client: OpenAI, translation_model: str = "gpt-4o-mini"):
        try:
            self.ld_model = load_language_detector()
        except Exception as e:
            print(f"Warning: Could not load language detection model: {e}")
            self.ld_model = None
        self.openai_client = openai_client
        self.translation_model = translation_model

    def _translate(self, text: str) -> str:
        """Translate text from English to Thai."""
        def send_translate_request(text: str):
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given English text to Thai. Do not translate words or phrases inside quotes (\"xxxx\"). Only output the translated text without any explanations."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.translation_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048
                )
                if response.choices[0].finish_reason == 'length':
                    raise Exception("Translation response too long")
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI translation error: {e}")
                raise

        def replace_quotes(pre_translate, post_translate):
            """Replace quoted text in translation with original quoted text."""
            quote_pairs = [("'", "'"), ('"', '"'), (''', '''), ('"', '"')]
            patterns = []
            for oq, cq in quote_pairs:
                pattern = f'({re.escape(oq)})(.*?){re.escape(cq)}'
                patterns.append(pattern)
            full_pattern = '|'.join(patterns)
            quote_pattern = re.compile(full_pattern)
            
            pre_quotes = []
            for match in quote_pattern.finditer(pre_translate):
                for i in range(2, len(match.groups()) + 1, 2):
                    if match.group(i):
                        pre_quotes.append(match.group(i))
                        break
            
            pre_quote_iter = iter(pre_quotes)
            def replace_match(match):
                try:
                    original_text = next(pre_quote_iter)
                    for i in range(1, len(match.groups()) + 1, 2):
                        if match.group(i):
                            opening_quote = match.group(i)
                            closing_quote = match.group(i)
                            break
                    return f"{opening_quote}{original_text}{closing_quote}"
                except StopIteration:
                    return match.group(0)
            
            result = quote_pattern.sub(replace_match, post_translate)
            return result

        try:
            translate_text = send_translate_request(text)
            translate_text = replace_quotes(text, translate_text)
            return translate_text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def predict_and_translate(self, text: str, instruction: str) -> str:
        """Predict language and translate instruction if needed."""
        if self.ld_model is None:
            return instruction
            
        try:
            lang = detect_lang(self.ld_model, text)
            if "eng" in lang:
                return instruction
            elif "tha" in lang:
                return self._translate(instruction)
            else:
                if random.random() > 0.5:
                    return instruction
                else:
                    return self._translate(instruction)
        except Exception:
            return instruction

class AutoIFPipeline:
    """Main pipeline class that implements steps 6-9 of the AutoIF process."""
    
    def __init__(self, 
                 back_trans_filter_path: str,
                 question_prompt_path: str,
                 output_path: str,
                 openai_model: str = "gpt-4o-mini",
                 quality_threshold: float = 8.0,
                 enable_translation: bool = False,
                 translation_model: str = "gpt-4o-mini",
                 openai_api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 num_workers: int = 8,
                 max_items: int = None,
                 cache_dir: str = "cache",
                 use_cache: bool = True,
                 final_split: str = None,
                 no_skip_special_tokens: bool = False,
                 force_thai: bool = False,
                 ):
        """
        Initialize the AutoIF pipeline.
        
        Args:
            back_trans_filter_path: Path to the back_trans_filter.jsonl file
            question_prompt_path: Path to the question_prompt.jsonl file
            output_path: Path for final output file
            openai_model: OpenAI model name for generation
            quality_threshold: Minimum quality score for filtering
            enable_translation: Whether to enable translation functionality
            translation_model: OpenAI model to use for translation (default: gpt-4o-mini)
            openai_api_key: OpenAI API key
            num_workers: Number of parallel workers for API calls
            max_items: Maximum number of items to process (None for all)
            cache_dir: Directory to store cache files
            use_cache: Whether to use caching functionality
            final_split: Final split for evaluation
            no_skip_special_tokens: Whether to disable skipping special tokens
            force_thai: Whether to force thai responses
        """
        self.back_trans_filter_path = back_trans_filter_path
        self.question_prompt_path = question_prompt_path
        self.output_path = output_path
        self.openai_model = openai_model
        self.quality_threshold = quality_threshold
        self.enable_translation = enable_translation
        self.num_workers = num_workers
        self.max_items = max_items
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.final_split = final_split
        self.no_skip_special_tokens = no_skip_special_tokens
        self.force_thai = force_thai
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'), base_url=base_url)
        self.ld_model = load_language_detector()

        # Initialize translator if enabled
        if self.enable_translation:
            self.translator = InstructionTranslator(self.openai_client, translation_model)
        else:
            self.translator = None
    
    def _get_cache_key(self, step_name: str, data_hash: str = None) -> str:
        """Generate a cache key for a given step."""
        key_parts = [
            step_name,
            self.openai_model,
            str(self.quality_threshold),
            str(self.max_items),
            str(self.enable_translation)
        ]
        if data_hash:
            key_parts.append(data_hash)
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def _get_data_hash(self, data: Any) -> str:
        """Generate a hash for input data to detect changes."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def _save_cache(self, step_name: str, data: Any, input_hash: str = None) -> None:
        """Save data to cache."""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(step_name, input_hash)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Cached {step_name} results to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache for {step_name}: {e}")
    
    def _load_cache(self, step_name: str, input_hash: str = None) -> Optional[Any]:
        """Load data from cache if available."""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(step_name, input_hash)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {step_name} results from cache ({cache_file})")
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache for {step_name}: {e}")
            return None
    
    def _clear_cache(self, step_name: str = None) -> None:
        """Clear cache files. If step_name is None, clear all cache."""
        if not self.use_cache or not self.cache_dir.exists():
            return
        
        if step_name:
            # Clear cache for specific step
            pattern = f"*{step_name}*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                print(f"Cleared cache file: {cache_file}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print(f"Cleared all cache files from {self.cache_dir}")

    def load_questions(self) -> List[str]:
        """Load question prompts from question_prompt.jsonl."""
        questions = []
        with open(self.question_prompt_path, "r") as f:
            for l in f:
                row = json.loads(l)
                if row['conversations'][0]['role'] == 'user':
                    questions.append(row['conversations'][0]['content'])
        return questions

    def load_constraints(self) -> List[Dict[str, Any]]:
        """Load constraint instructions from back_trans_filter.jsonl and expand back_instruction field."""
        constraints = []
        with jsonlines.open(self.back_trans_filter_path, "r") as f:
            for item in f:
                # If instruction is empty but back_instruction exists, expand it
                if item.get('back_instruction'):
                    # Create a separate constraint for each back_instruction
                    for back_inst in item['back_instruction']:
                        if back_inst.strip() == '':
                            continue
                        new_item = copy.deepcopy(item)
                        new_item['instruction'] = back_inst
                        constraints.append(new_item)
                elif item.get('instruction', '').strip() != '':
                    # Use the item as-is if instruction is not empty
                    constraints.append(item)
        return constraints

    def step6_concat_instruct_query(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 6: Concatenate instruction constraints with queries to create prompts.
        """
        print("Step 6: Concatenating instruction constraints with queries...")
        
        # Load question prompts
        question_prompts = self.load_questions()
        random.shuffle(question_prompts)
        question_prompts = question_prompts[:len(constraints)]

        if self.max_items:
            question_prompts = question_prompts[:self.max_items]
        
        print(f"Loaded {len(question_prompts)} question prompts")
        print(f"Using {len(constraints)} constraints")
        
        # Check cache first
        input_data = {"constraints": constraints, "questions": question_prompts}
        input_hash = self._get_data_hash(input_data)
        cached_result = self._load_cache("step6", input_hash)
        if cached_result is not None:
            return cached_result
        
        inputs = []
        for idx, query in enumerate(tqdm(question_prompts, desc="Processing queries")):
            constraint = constraints[idx]
            # Create prompt combining constraint and query
            prompt = f"Please answer the query strictly following the instruction.\n[instruction] {constraint['instruction']}\n[Query] {query}"
            
            lang = detect_lang(self.ld_model, query)
            if self.force_thai:
                if "eng" in lang:
                    prompt = prompt + '\nPLEASE ANSWER IN THAI.'
                else:
                    continue

            item = copy.deepcopy(constraint)
            item["prompt"] = prompt
            item["query"] = query
            item["language"] = lang
            inputs.append(item)
    
        print(f"Generated {len(inputs)} instruction-query combinations")
        random.shuffle(inputs)
        if self.max_items:
            inputs = inputs[:self.max_items]
            print(f"Limited to {len(inputs)} instruction-query combinations to stay within max_items={self.max_items}")

        # Save to cache
        self._save_cache("step6", inputs, input_hash)
        return inputs

    def call_openai_api(self, messages: List[Dict[str, str]], n: int = 2, max_tokens: int = 1024, temperature: float = 0.7) -> List[str]:
        """
        Call OpenAI API to generate responses.
        """
        openai_kwargs = {}
        if 'gpt' not in self.openai_model:
            openai_kwargs['extra_body'] = {
                "skip_special_tokens": not self.no_skip_special_tokens
            }
        if 'gpt-5' not in self.openai_model:
            openai_kwargs['max_tokens'] = max_tokens
            openai_kwargs['temperature'] = temperature
        else:
            openai_kwargs['reasoning_effort'] = 'low'
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                n=n,
                **openai_kwargs,
            )
            out = []
            for choice in response.choices:
                if choice.finish_reason == 'length':
                    pass
                else:
                    out.append(choice.message.content)
            return out
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Retry once after a delay
            time.sleep(2)
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    n=n,
                    **openai_kwargs,
                )
                out = []
                for choice in response.choices:
                    if choice.finish_reason == 'length':
                        pass
                    else:
                        out.append(choice.message.content)
                return out
            except Exception as e2:
                print(f"OpenAI API retry failed: {e2}")
                return []

    def step7_query_verification(self, prompted_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Step 7: Verify GPT responses using evaluation functions and create quality scoring prompts.
        """
        print("Step 7: Verifying GPT responses...")
        
        # Check cache first
        input_hash = self._get_data_hash(prompted_data)
        cached_result = self._load_cache("step7", input_hash)
        if cached_result is not None:
            return cached_result
        
        # Call OpenAI API to generate responses for each prompt (parallel processing)
        def generate_response_for_item(item):
            """Generate response for a single item."""
            messages = [{"role": "user", "content": item["prompt"]}]
            original_gpt_responses = self.call_openai_api(messages, n=2, max_tokens=4096, temperature=0.7)
            if self.final_split:
                gpt_responses = [response.split(self.final_split)[-1] for response in original_gpt_responses]
            else:
                gpt_responses = original_gpt_responses
            if gpt_responses:  # Only add if we got responses
                return {
                    "prompt": item["prompt"],
                    "instruction": item["instruction"],
                    "eval_func": item["eval_func"],
                    'original-gpt-answer': original_gpt_responses,
                    "gpt-answer": gpt_responses,
                    "query": item["query"]
                }
            return None
        
        # Use ThreadPoolExecutor for parallel API calls
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(generate_response_for_item, item): item for item in prompted_data}
            
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                             total=len(prompted_data), desc="Generating responses"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        filter_samples = []
        
        for result in tqdm(results, desc="Verifying responses"):
            eval_funcs = []
            
            # Prepare evaluation functions (exactly like original)
            for func, score in result["eval_func"]:
                # add original version of eval
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                    eval_funcs.append(local_vars["evaluate"])
                except Exception as e:
                    print(f"Error compiling eval function: {e}")
                    continue
                
                # add thai version of eval
                local_vars = {}
                try:
                    func_thai = func.replace('。', '.')  # normalize what gpt think it is full stop in Thai
                    func_thai = func_thai.replace("sentences = response.split(' ')", "sentences = response.split('.')")
                    func_thai = func_thai.replace("paragraphs = response.strip().split('\n')", "paragraphs = response.strip().split('\n\n')")
                    func_thai = func_thai.replace("paragraphs = response.split('\n')", "paragraphs = response.split('\n\n')")
                    exec(func_thai, globals(), local_vars)
                    eval_funcs.append(local_vars["evaluate"])
                except Exception as e:
                    print(f"Error compiling Thai eval function: {e}")
                    continue
            
            filter_responses = []
            filter_full_responses = []
            
            for response, full_response in zip(result["gpt-answer"], result['original-gpt-answer']):
                acc = []
                # if any of Thai / original version is corrected then, ok
                response_original = copy.copy(response)
                full_response_original = copy.copy(full_response)
                # Try to import Thai tokenization (fallback to simple splitting if not available)
                try:
                    from pythainlp import word_tokenize, sent_tokenize
                    response_sent_tokenized = '.'.join([res for res in sent_tokenize(response)]).strip().strip('.').strip()
                    response_word_tokenized = '.'.join([' '.join(word_tokenize(res)) for res in sent_tokenize(response)]).strip().strip('.').strip()
                except ImportError:
                    # Fallback if pythainlp is not available
                    response_sent_tokenized = response.replace('\n', '. ').strip()
                    response_word_tokenized = ' '.join(response.split())
                
                # Debug: Print first response details
                if len(filter_samples) == 0:  # Only for first few samples
                    print(f"\n=== DEBUG FIRST SAMPLE ===")
                    print(f"Instruction: {result['instruction'][:100]}...")
                    print(f'Func: {result["eval_func"]}')
                    print(f"Response: {response_original[:100]}...")
                    print(f"Number of eval functions: {len(eval_funcs)}")
                
                for i, eval_func in enumerate(eval_funcs):
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                        res1 = eval_func(response_sent_tokenized)  # eval three times if one is pass then ok
                        res2 = eval_func(response_word_tokenized)
                        res3 = eval_func(response_original)
                        res = res1 or res2 or res3
                        
                        # Debug: Print evaluation results for first sample
                        if len(filter_samples) == 0:
                            print(f"Eval func {i}: res1={res1}, res2={res2}, res3={res3}, final={res}")
                            
                    except Exception as e:
                        if len(filter_samples) == 0:
                            print(f"Evaluation error for func {i}: {e}")
                        res = None
                    finally:
                        signal.alarm(0)
                    
                    if res is not None:
                        try:
                            acc.append(int(res))
                        except Exception as e:
                            if len(filter_samples) == 0:
                                print(f"Error converting result to int: {e}, res={res}")
                            continue
                
                acc_score = np.mean(acc) if acc else 0
                chinese_count = count_chinese_chars(response_original)
                language_corrected = True
                if self.force_thai:
                    lang = detect_lang(self.ld_model, response_original)
                    if "tha" not in lang:
                        language_corrected = False

                # Debug: Print scoring details for first sample
                if len(filter_samples) == 0:
                    print(f"acc: {acc}, acc_score: {acc_score}, chinese_count: {chinese_count}")
                    print(f"Passes filter: {acc_score > 0 and chinese_count == 0}")
                    print(f"=== END DEBUG ===")
                
                # qwen mostly code-switching chinese character, so we don't want any
                if acc_score > 0 and chinese_count == 0 and language_corrected:
                    filter_responses.append(response_original)
                    filter_full_responses.append(full_response_original)
            
            # Debug: Track acceptance/rejection
            if len(filter_responses) == 0:
                print(f"REJECTED: No responses passed evaluation for instruction: {result['instruction'][:50]}...")
            else:
                print(f"ACCEPTED: {len(filter_responses)} responses passed for instruction: {result['instruction'][:50]}...")
            
            # Add successful responses to filter samples
            for response, full_response in zip(filter_responses, filter_full_responses):
                try:
                    query_match = re.findall(r"\[Query\](.*)$", result["prompt"], re.DOTALL)
                    query = query_match[0].strip() if query_match else result.get("query", "")
                    
                    filter_samples.append({
                        "instruction": result["instruction"],
                        "query": query,
                        "response": response,
                        "full_response": full_response,
                    })
                except Exception as e:
                    print(f"Error processing response: {e}")
                    print(f"Problematic prompt: {result['prompt']}")
        
        # Remove duplicates
        filter_samples = list(map(json.loads, set(map(json.dumps, filter_samples))))
        print(f"Total samples left for quality scoring: {len(filter_samples)}")
        
        # Create quality scoring prompts
        prompt_template = """You are an expert that is good at judging whether a response is following the instruction and query.
[Instruction] {instruction}
[Query] {query}
[Response] {response}
Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.
You need to judge whether the response answers the query. Also, you need to judge whether the query is a clear instruction. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.
Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.
Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."""

        quality_prompts = []
        for sample in filter_samples:
            sample_copy = copy.deepcopy(sample)
            sample_copy["validate_prompt"] = prompt_template.format(
                instruction=sample["instruction"],
                query=sample["query"],
                response=sample["response"]
            )
            quality_prompts.append(sample_copy)
        
        # Save to cache
        result = (filter_samples, quality_prompts)
        self._save_cache("step7", result, input_hash)
        return result

    def get_quality_score_response(self, prompt: str) -> str:
        """Get quality scoring response from OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        responses = self.call_openai_api(messages, n=1, max_tokens=1024, temperature=0.7)
        return responses[0] if responses else ""

    def step8_query_score_filter(self, quality_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 8: Filter responses based on quality scores.
        """
        print("Step 8: Filtering by quality scores...")
        
        # Check cache first
        input_hash = self._get_data_hash(quality_prompts)
        cached_result = self._load_cache("step8", input_hash)
        if cached_result is not None:
            return cached_result
        
        # Call OpenAI API for quality scoring (parallel processing)
        def get_quality_score_for_item(item):
            """Get quality score for a single item."""
            quality_response = self.get_quality_score_response(item["validate_prompt"])
            if quality_response:  # Only add if we got a response
                return {
                    "instruction": item["instruction"],
                    "query": item["query"],
                    "response": item["response"],
                    "full_response": item["full_response"],
                    "quality_gen": [quality_response]
                }
            return None
        
        # Use ThreadPoolExecutor for parallel quality scoring
        scored_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(get_quality_score_for_item, item): item for item in quality_prompts}
            
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                             total=len(quality_prompts), desc="Getting quality scores"):
                result = future.result()
                if result is not None:
                    scored_results.append(result)
        
        filter_results = []
        for result in tqdm(scored_results, desc="Filtering by quality"):
            scores = []
            for gen_response in result["quality_gen"]:
                score_match = re.findall(r"Score: (\d+?)$", gen_response)
                if score_match:
                    scores.append(int(score_match[0]))
            
            avg_score = np.mean(scores) if scores else 0
            if avg_score > self.quality_threshold:
                filter_results.append(result)
        
        print(f"Total samples left after quality filter: {len(filter_results)}")
        
        # Save to cache
        self._save_cache("step8", filter_results, input_hash)
        return filter_results

    def step9_sft_data_construction(self, filtered_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 9: Construct final SFT data in conversation format.
        """
        print("Step 9: Constructing SFT data...")
        
        # Check cache first
        input_hash = self._get_data_hash(filtered_data)
        # this one is not cached; due to change of pipeline on prompt change;
        # cached_result = self._load_cache("step9", input_hash)
        # if cached_result is not None:
        #     return cached_result
        
        def _concat_instruct(item: Dict[str, Any]) -> List[Dict[str, str]]:
            """Concatenate instruction with query as user input using diverse patterns."""
            query = item["query"][0].upper() + item["query"][1:] if item["query"] else ""
            instruction = item["instruction"][0].upper() + item["instruction"][1:].strip(".") if item["instruction"] else ""
            
            # Apply translation if enabled
            if self.translator:
                instruction = self.translator.predict_and_translate(query, instruction)
            
            # Define diverse patterns for combining instruction and query
            patterns = [
                # Pattern 1: Direct concatenation (original style) ~ 50% of total randomness
                lambda q, i: f"{q} {i}",
                lambda q, i: f"{i} {q}",
                lambda q, i: f"{q} {i}",
                lambda q, i: f"{i} {q}",
                lambda q, i: f"{q} {i}",
                lambda q, i: f"{i} {q}",
                lambda q, i: f"{q} {i}",
                lambda q, i: f"{i} {q}",
                
                # Pattern 2: Instruction-first format
                lambda q, i: f"Following the instruction '{i}', please {q.lower()}",
                
                # Pattern 3: Explicit instruction format
                lambda q, i: f"Please answer this question while strictly adhering to the following constraint: {i}\n\nQuestion: {q}",
                
                # Pattern 4: Task-oriented format
                lambda q, i: f"Task: {q}\nConstraint: {i}\nPlease complete the task following the constraint.",
                
                # Pattern 5: Imperative format
                lambda q, i: f"{i} Answer: {q}",
                
                # Pattern 6: Structured format
                lambda q, i: f"Query: {q}\nInstruction: {i}\nResponse:",
                
                # Pattern 7: Natural language format
                lambda q, i: f"I need you to {q} Make sure to {i}",
                
                # Pattern 8: Formal request format
                lambda q, i: f"Request: {q}\nPlease ensure your response {i}",
                
                # Pattern 9: Conditional format
                lambda q, i: f"When answering '{q}', remember that your response must {i}",
                
                # Pattern 10: Step-by-step format
                lambda q, i: f"1. Question: {q}\n2. Requirement: {i}\n3. Please provide your answer:",
            ]
            
            # Randomly select a pattern
            pattern = random.choice(patterns)
            
            # Handle punctuation more intelligently
            query_clean = query.rstrip('.?!')
            instruction_clean = instruction.rstrip('.?!')
            
            try:
                inputs = pattern(query_clean, instruction_clean)
            except Exception:
                # Fallback to simple concatenation if pattern fails
                inputs = f"{query} {instruction}."
            
            return [
                {"role": "user", "content": inputs.strip()},
                {"role": "assistant", "content": item["full_response"].strip()},
            ]

        def _concat_as_system(item: Dict[str, Any]) -> List[Dict[str, str]]:
            """Use instruction as system message with diverse patterns."""
            query = item["query"][0].upper() + item["query"][1:] if item["query"] else ""
            instruction = item["instruction"][0].upper() + item["instruction"][1:] if item["instruction"] else ""
            
            # Apply translation if enabled
            if self.translator and random.random() > 0.5:
                instruction = self.translator.predict_and_translate(query, instruction)
            
            # Define diverse system message patterns
            system_patterns = [
                # Pattern 1: Direct instruction (original style) (50% of randomness)
                lambda i: i,
                lambda i: i,
                lambda i: i,
                lambda i: i,
                lambda i: i,
                lambda i: i,
                lambda i: i,
                lambda i: i,
                
                # Pattern 2: Role-based instruction
                lambda i: f"You are an AI assistant. {i}",
                
                # Pattern 3: Constraint-focused instruction
                lambda i: f"When responding to user queries, ensure that you {i}",
                
                # Pattern 4: Behavioral instruction
                lambda i: f"Your responses must always {i}",
                
                # Pattern 5: Task-oriented system message
                lambda i: f"Task guidelines: {i}",
                
                # Pattern 6: Imperative system message
                lambda i: f"Important: {i}",
                
                # Pattern 7: Rule-based system message
                lambda i: f"Follow this rule when answering: {i}",
                
                # Pattern 8: Context-setting system message
                lambda i: f"Context: You should {i} in all your responses.",
                
                # Pattern 9: Formal instruction
                lambda i: f"System directive: {i}",
                
                # Pattern 10: Conversational system message
                lambda i: f"Please remember to {i} when providing your answer."
            ]
            
            # Randomly select a system pattern
            system_pattern = random.choice(system_patterns)
            
            # Clean instruction for better formatting
            instruction_clean = instruction.rstrip('.?!')
            
            try:
                system_content = system_pattern(instruction_clean)
            except Exception:
                # Fallback to original instruction if pattern fails
                system_content = instruction
            
            return [
                {"role": "system", "content": system_content.strip()},
                {"role": "user", "content": query.strip()},
                {"role": "assistant", "content": item["full_response"].strip()},
            ]

        sft_data = []
        for item in tqdm(filtered_data, desc="Constructing conversations"):
            try:
                if random.random() > 0.5:
                    conversations = _concat_instruct(item)
                else:
                    conversations = _concat_as_system(item)
                
                meta_data = {
                    "instruction": item["instruction"],
                    "query": item["query"],
                    "full_response": item["full_response"],
                    "response": item["response"],
                    "quality_gen": item["quality_gen"],
                }
                sft_data.append({"conversations": conversations, "meta": meta_data})
            except Exception as e:
                print(f"Error constructing conversation: {e}")
                continue
        
        print(f"Constructed {len(sft_data)} SFT samples")
        
        # Save to cache
        self._save_cache("step9", sft_data, input_hash)
        return sft_data

    def run_pipeline(self) -> List[Dict[str, Any]]:
        """Run the complete pipeline from steps 6-9."""
        print("Starting AutoIF Pipeline (Steps 6-9)...")
        
        # Load constraints
        constraints = self.load_constraints()
        print(f"Loaded {len(constraints)} constraints")
        
        # Step 6: Concatenate instructions with queries
        prompted_data = self.step6_concat_instruct_query(constraints)
        
        # Step 7: Verify responses and create quality scoring prompts
        verified_samples, quality_prompts = self.step7_query_verification(prompted_data)
        
        # Step 8: Filter by quality scores
        filtered_data = self.step8_query_score_filter(quality_prompts)
        
        # Step 9: Construct final SFT data
        sft_data = self.step9_sft_data_construction(filtered_data)
        if os.path.dirname(self.output_path) != "":
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # Save output
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Pipeline completed. Output saved to: {self.output_path}")
        return sft_data

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="AutoIF Pipeline - Steps 6-9")
    parser.add_argument(
        "--back_trans_filter",
        type=str,
        required=True,
        help="Path to back_trans_filter.jsonl file"
    )
    parser.add_argument(
        "--question_prompt",
        type=str,
        required=True,
        help="Path to question prompts file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for final SFT dataset"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=8.0,
        help="Minimum quality score for filtering"
    )
    parser.add_argument(
        "--enable_translation",
        action="store_true",
        help="Enable translation functionality"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel processing"
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for translation"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (can also use OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Base URL for OpenAI API"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process (None for all)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to store cache files"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching functionality"
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear all cache files before running"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=422,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--final_split",
        type=str,
        default="",
        help="Final split for evaluation"
    )
    parser.add_argument(
        "--no_skip_special_tokens",
        action="store_true",
        help="Disable skipping special tokens"
    )
    parser.add_argument(
        "--force_thai",
        action="store_true",
        help="Force thai responses"
    )
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Initialize and run pipeline
    pipeline = AutoIFPipeline(
        back_trans_filter_path=args.back_trans_filter,
        question_prompt_path=args.question_prompt,
        output_path=args.output,
        openai_model=args.openai_model,
        quality_threshold=args.quality_threshold,
        enable_translation=args.enable_translation,
        translation_model=args.translation_model,
        openai_api_key=args.openai_api_key,
        base_url=args.base_url,
        max_items=args.max_items,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        use_cache=not args.no_cache,
        final_split=args.final_split,
        no_skip_special_tokens=args.no_skip_special_tokens,
        force_thai=args.force_thai
    )
    
    # Clear cache if requested
    if args.clear_cache:
        pipeline._clear_cache()
        print("Cache cleared.")
    
    sft_data = pipeline.run_pipeline()
    print(f"Generated {len(sft_data)} SFT samples")

if __name__ == "__main__":
    main()
