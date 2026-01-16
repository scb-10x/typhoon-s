#!/usr/bin/env python3
"""
Augment Mirage Bench dataset for RL and SFT training with VERL
"""

import argparse
import os
import re
import pandas as pd
from datasets import load_dataset, Dataset
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm

def check_is_self_contained(client, model, question):
    """
    Check if the question is self-contained using OpenAI API.
    """
    try:
        kwargs = {}
        if 'gpt-5' in model:
            pass
        else:
            kwargs['temperature'] = 0.0
            kwargs['max_tokens'] = 10
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data quality assistant. Check if the following question is self-contained. A self-contained question explicitly names the entity or subject it refers to and does not use unresolved pronouns or references like 'the text', 'the document', 'this paper', 'mentioned above', 'he', 'she', 'it' (without antecedent). It must be understandable without any accompanying context document.\n\nRespond with exactly 'YES' if it is self-contained, or 'NO' if it depends on external context."},
                {"role": "user", "content": f"Question: {question}"}
            ],
            **kwargs
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"Error checking question '{question}': {e}")
        return False

def parse_mirage_prompt(prompt_text):
    """
    Parse the Mirage Bench prompt to extract question and contexts.
    Format:
    Question: {question}
    
    Contexts:
    {contexts}
    
    Instruction:
    ...
    """
    # Extract Question
    question_match = re.search(r"Question:\s*(.*?)\s*\n\nContext", prompt_text, re.DOTALL)
    if not question_match:
        # Fallback for some formats: "Question: ...\n\nContext:"
        question_match = re.search(r"Question:\s*(.*?)\s*Context", prompt_text, re.DOTALL)
        
    question = question_match.group(1).strip() if question_match else ""
    
    # Extract Contexts section
    # Usually between "Context:\n" or "Contexts: " and "\n\nInstruction:"
    context_section_match = re.search(r"Contexts?:\s*(\[.*?\] .*?)\s*\n\nInstruction:", prompt_text, re.DOTALL)
    context_section = context_section_match.group(1).strip() if context_section_match else ""
    
    contexts = []
    if context_section:
        # Split by [ID] usually starting a new line or just appearing
        # Regex to find [ID] pattern
        # doc_pattern = re.compile(r"\[(.*?)\] (.*?)(?: - (.*))?$")
        # It seems context items are often separated by newlines, but sometimes they might be dense.
        # Let's split by the pattern `\n[` or just `[` if at start.
        
        # We can try to split by the [ID] markers
        # But we need to keep the content.
        
        # Simple approach: split by newline, check if line starts with [ID]
        # Or better: use re.finditer to locate all [ID] blocks
        
        # Pattern: [alphanum#digit] or [digit]
        # Based on example: [377253#0], [12146#1]
        
        pattern = r"(\[.*?\])\s*(.*?)(?=\n\[|$)"
        # Note: This assumes contexts are separated by newlines and [
        
        # Let's clean the section first
        # normalize newlines
        
        matches = re.finditer(r"(\[[^\]]+\])\s+(.*?)(?=\n\[|$)", context_section, re.DOTALL)
        
        # If the above regex is too strict, we can iterate lines.
        # But contexts can be multi-line?
        # In the example: [377253#0] Title - Content ... \n[378056#2] ...
        # So yes, they are separated by `\n[`.
        
        # Let's try to split by `\n[` but keep the `[` of the next item.
        # Or just split by `\n` and group?
        
        # Let's use re.split with positive lookahead
        parts = re.split(r'\n(?=\[)', context_section)
        
        for part in parts:
            part = part.strip()
            if not part.startswith('['):
                continue
                
            # Parse single context entry
            # Expect: [ID] Title - Content
            # Or: [ID] Content (if no title separator)
            
            # Match [ID]
            m_id = re.match(r"^(\[.*?\])\s*(.*)$", part, re.DOTALL)
            if not m_id:
                continue
                
            doc_id = m_id.group(1)
            rest = m_id.group(2)
            
            # Try to extract title
            # Look for " - "
            # care for " - " inside content? Usually Title - Content
            # Assume first " - " is separator
            
            title = ""
            content = rest
            
            if " - " in rest:
                split_rest = rest.split(" - ", 1)
                title = split_rest[0].strip()
                content = split_rest[1].strip()
            
            contexts.append({
                'id': doc_id,
                'title': title,
                'content': content, 
                'full_text': part
            })
            
    return question, contexts

def process_rl(example, idx, system_prompt=None, openai_client=None, openai_model="gpt-4o"):
    """
    Process a single example for RL dataset.
    """
    prompt_text = example.get('prompt', '')
    if not prompt_text:
        return None
        
    question, _ = parse_mirage_prompt(prompt_text)
    
    if openai_client:
        is_self_contained = check_is_self_contained(openai_client, openai_model, question)
        if not is_self_contained:
            return None

    # Get answers
    answers = example.get('tydiqa_answer', [])
    if not answers:
        return None
        
    # Format answer string
    if isinstance(answers, list):
        answer_str = " or ".join(answers)
    else:
        answer_str = str(answers)
    
    if not question or not answer_str:
        return None
        
    # Create guideline
    guideline = f"To correctly answer this question, the response must match the following answer:\n\n{answer_str}"
    
    # Build prompt
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": question})
    
    return {
        "data_source": "mirage-bench",
        "prompt": prompt,
        "ability": "qa",
        "reward_model": {
            "style": "guideline",
            "ground_truth": guideline
        },
        "extra_info": {
            "index": idx,
            "query_id": example.get('query_id', str(idx)),
            "original_answers": answers
        }
    }

def create_sft_dataset(dataset, omit_user_turn=False, lang='th'):
    """
    Create SFT dataset from contexts found in prompts.
    """
    contexts_map = {} # Key by ID to deduplicate
    
    print("Extracting contexts for SFT...")
    for item in dataset:
        prompt_text = item.get('prompt', '')
        if not prompt_text:
            continue
            
        _, item_contexts = parse_mirage_prompt(prompt_text)
        
        for ctx in item_contexts:
            doc_id = ctx['id']
            if doc_id not in contexts_map:
                contexts_map[doc_id] = ctx
    
    print(f"Found {len(contexts_map)} unique contexts.")
    
    sft_data = []
    for doc_id, ctx in contexts_map.items():
        title = ctx['title']
        content = ctx['content']
        
        if not content:
            continue
            
        # Create instruction
        # Adapt instruction based on language if simple heuristic
        if lang == 'th':
            if title:
                instruction = f"ขอข้อมูลเกี่ยวกับ {title}"
            else:
                instruction = f"ขอข้อมูลเกี่ยวกับเอกสาร {doc_id}"
        else: # Default to English
            if title:
                instruction = f"Provide information about {title}"
            else:
                instruction = f"Provide information about document {doc_id}"
        
        # Build messages
        messages = []
        if not omit_user_turn:
            messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": content})
        
        sft_data.append({
            "messages": messages,
            "extra_info": {
                "doc_id": doc_id,
                "title": title
            }
        })
        
    return sft_data

def main():
    parser = argparse.ArgumentParser(description="Augment Mirage Bench dataset for RL and SFT")
    parser.add_argument("--local_dir", default="dataset/verl/mirage2", help="Base directory to save datasets")
    parser.add_argument("--dataset", default="nthakur/mirage-bench-instruct", help="Dataset name")
    parser.add_argument("--config", default="th", help="Dataset configuration (language code)")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--add-system_prompt", action="store_true", help="Use system prompt to add to RL dataset")
    parser.add_argument("--omit_user_turn", action="store_true", help="Omit user turn in SFT dataset")
    parser.add_argument("--check-self-contained", action="store_true", help="Filter for self-contained questions using OpenAI")
    parser.add_argument("--openai_model", default="gpt-5-nano", help="OpenAI model to use for check")
    parser.add_argument("--max_workers", type=int, default=64, help="Number of workers for parallel processing")
    
    args = parser.parse_args()
    
    openai_client = None
    if args.check_self_contained:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
             print("Error: OPENAI_API_KEY environment variable is not set.")
             return
             
        openai_client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
        print(f"Using OpenAI model {args.openai_model} to filter self-contained questions.")
    
    if args.add_system_prompt:
        args.system_prompt = "You are a reasoning assistant. First, think through the reasoning internally, then present the reasoning within <thinking>...</thinking>. After thinking, state a response that addresses the user's request."
    else:
        args.system_prompt = None
        
    print(f"Loading dataset: {args.dataset} ({args.config}, {args.split})")
    try:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Total items: {len(dataset)}")
    
    # Process RL
    print("\nProcessing RL dataset...")
    rl_all_items = []
    
    if openai_client and args.max_workers > 1:
        print(f"Using {args.max_workers} workers for parallel processing...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Create a partial function or lambda if needed, but simple list comp works
            # We need to pass i (index) as well.
            futures = [
                executor.submit(process_rl, item, i, system_prompt=args.system_prompt, openai_client=openai_client, openai_model=args.openai_model)
                for i, item in enumerate(dataset)
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset), desc="Processing"):
                try:
                    processed = future.result()
                    if processed:
                        rl_all_items.append(processed)
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
    else:
        # Sequential processing
        for i, item in enumerate(tqdm(dataset, desc="Processing")):
            processed = process_rl(item, i, system_prompt=args.system_prompt, openai_client=openai_client, openai_model=args.openai_model)
            if processed:
                rl_all_items.append(processed)
            
    print(f"Generated {len(rl_all_items)} RL examples")
    
    if len(rl_all_items) > 0:
        # Split into train (90%) and test (10%)
        rl_dataset = Dataset.from_list(rl_all_items)
        # Handle small datasets
        if len(rl_dataset) < 10:
             rl_train_items = rl_dataset.to_list()
             rl_test_items = rl_dataset.to_list()
        else:
            split_dataset = rl_dataset.train_test_split(test_size=0.1, seed=42)
            rl_train_items = split_dataset['train'].to_list()
            rl_test_items = split_dataset['test'].to_list()
            
        # Save RL
        rl_save_dir = os.path.join(args.local_dir, args.config, "rl")
        os.makedirs(rl_save_dir, exist_ok=True)
        
        rl_train_path = os.path.join(rl_save_dir, "train.parquet")
        rl_test_path = os.path.join(rl_save_dir, "test.parquet")
        
        Dataset.from_list(rl_train_items).to_parquet(rl_train_path)
        Dataset.from_list(rl_test_items).to_parquet(rl_test_path)
        
        print(f"Saved RL train dataset ({len(rl_train_items)} examples) to {rl_train_path}")
        print(f"Saved RL test dataset ({len(rl_test_items)} examples) to {rl_test_path}")

        if len(rl_train_items) > 0:
            print("\nExample RL item:")
            print(rl_train_items[0])

    # Process SFT
    print("\nProcessing SFT dataset...")
    sft_items = create_sft_dataset(dataset, omit_user_turn=args.omit_user_turn, lang=args.config)
    
    if sft_items:
        # Save SFT
        sft_save_dir = os.path.join(args.local_dir, args.config, "sft")
        os.makedirs(sft_save_dir, exist_ok=True)
        sft_dataset = Dataset.from_list(sft_items)
        print(f"Saving SFT dataset with {len(sft_dataset)} items...")
        
        if len(sft_dataset) > 0:
            print("\nExample SFT item:")
            print(sft_dataset[0])

        sft_output_path = os.path.join(sft_save_dir, "train.parquet")
        sft_dataset.to_parquet(sft_output_path)
        print(f"Saved SFT dataset to {sft_output_path}")
    else:
        print("No SFT items generated.")

if __name__ == "__main__":
    main()
