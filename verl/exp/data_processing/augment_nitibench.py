#!/usr/bin/env python3
"""
Augment Nitibench dataset for RL and SFT training with VERL
"""

import argparse
import os
import pandas as pd
from datasets import load_dataset, Dataset

def normalize_wangchan_item(item):
    relevant_laws = []
    if item.get('positive_contexts'):
        for ctx in item['positive_contexts']:
            metadata = ctx.get('metadata', {})
            relevant_laws.append({
                'law_name': metadata.get('law_title', ''),
                'section_num': metadata.get('section', ''),
                'section_content': ctx.get('context', '')
            })
            
    return {
        'question': item.get('question', ''),
        'answer': item.get('positive_answer', ''),
        'relevant_laws': relevant_laws,
        'reference_laws': [],
        # Add these for process_rl if it uses them, though original nitibench doesn't have them at top level
        'law_name': relevant_laws[0]['law_name'] if relevant_laws else '',
        'section_num': relevant_laws[0]['section_num'] if relevant_laws else '',
        'id': item.get('id', '') 
    }

def process_rl(example, idx, system_prompt=None):
    """
    Process a single example for RL dataset.
    """
    question = example.get('question', '')
    answer = example.get('answer', '')
    
    if not question or not answer:
        return None
        
    # Create guideline that specifies the answer must match
    guideline = f"To correctly answer this question, the response must match the following answer:\n\n{answer}"
    
    # Build prompt with optional system message
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": question})
    
    return {
        "data_source": "VISAI-AI/nitibench",
        "prompt": prompt,
        "ability": "legal",
        "reward_model": {
            "style": "guideline",
            "ground_truth": guideline
        },
        "extra_info": {
            "index": idx,
            "law_name": example.get('law_name', ''),
            "section_num": example.get('section_num', ''),
            "id": example.get('id', idx),
            "original_answer": answer
        }
    }

def format_law_content(content):
    if isinstance(content, list):
        return "\n".join([str(c) for c in content])
    return str(content)

def create_sft_dataset(dataset, omit_user_turn=False):
    """
    Create SFT dataset from law information, deduplicated by law_name + section_num.
    """
    import json
    laws = []
    
    for item in dataset:
        relevant = item.get('relevant_laws', [])
        reference = item.get('reference_laws', [])
        
        # Parse if strings
        if isinstance(relevant, str):
            try:
                relevant = json.loads(relevant)
            except:
                relevant = []
        if isinstance(reference, str):
            try:
                reference = json.loads(reference)
            except:
                reference = []
        
        # Combine both law sources
        all_laws = []
        if isinstance(relevant, list):
            all_laws.extend(relevant)
        if isinstance(reference, list):
            all_laws.extend(reference)
            
        # Extract unique laws
        for law_entry in all_laws:
            if not isinstance(law_entry, dict):
                continue
                
            law_name = law_entry.get('law_name', '')
            section_num = law_entry.get('section_num', '')
            section_content = law_entry.get('section_content', '')
            
            if not law_name or not section_content:
                continue
            
            assert law_name
            assert section_content
            assert section_num
                
            laws.append({
                'law_name': str(law_name).strip(),
                'section_num': str(section_num).strip() if section_num else '',
                'content': str(section_content).strip()
            })
    
    if not laws:
        return []

    df = pd.DataFrame(laws)
    # Deduplicate by law_name and section_num
    df = df.drop_duplicates(subset=['law_name', 'section_num'])
    
    sft_data = []
    for idx, row in df.iterrows():
        content = row['content']
        
        if not content:
            continue
            
        # Create instruction
        # Using Thai instruction since dataset is Thai
        if row['section_num']:
            instruction = f"ขอข้อมูลเกี่ยวกับ {row['law_name']} มาตรา {row['section_num']}"
        else:
            instruction = f"ขอข้อมูลเกี่ยวกับ {row['law_name']}"
        
        # Build messages with optional user turn
        messages = []
        if not omit_user_turn:
            messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": content})
        
        sft_data.append({
            "messages": messages,
            "extra_info": {
                "law_name": row['law_name'],
                "section_num": row['section_num'],
                "index": idx
            }
        })
        
    return sft_data

def main():
    parser = argparse.ArgumentParser(description="Augment Nitibench dataset for RL and SFT")
    parser.add_argument("--local_dir", default="dataset/verl/nitibench", help="Base directory to save datasets")
    parser.add_argument("--dataset", default="VISAI-AI/nitibench", help="Dataset name")
    parser.add_argument("--split", default="ccl", help="Dataset split")
    parser.add_argument("--add-system_prompt", action="store_true", help="Use system prompt to add to RL dataset")
    parser.add_argument("--omit_user_turn", action="store_true", help="Omit user turn in SFT dataset")
    
    args = parser.parse_args()
    
    # Set system_prompt based on flag
    if args.add_system_prompt:
        args.system_prompt = "You are a reasoning assistant. First, think through the reasoning internally, then present the reasoning within <thinking>...</thinking>. After thinking, state a response that addresses the user's request."
    else:
        args.system_prompt = None
    
    if args.dataset == "airesearch/WangchanX-Legal-ThaiCCL-RAG":
        print("Using WangchanX configuration (WangchanX train + Nitibench test)")
        try:
            train_dataset = load_dataset(args.dataset, split="train")
            test_dataset = load_dataset("VISAI-AI/nitibench", split="ccl")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return
            
        # Normalize and process train (WangchanX)
        print("Processing WangchanX train set...")
        sft_source_items = []
        rl_train_items = []
        for i, item in enumerate(train_dataset):
            norm_item = normalize_wangchan_item(item)
            sft_source_items.append(norm_item)
            processed = process_rl(norm_item, i, system_prompt=args.system_prompt)
            if processed:
                rl_train_items.append(processed)
                
        # Process test (Nitibench)
        print("Processing Nitibench test set...")
        rl_test_items = []
        for i, item in enumerate(test_dataset):
            processed = process_rl(item, i, system_prompt=args.system_prompt)
            if processed:
                rl_test_items.append(processed)
                
    else:
        print(f"Loading dataset: {args.dataset} ({args.split})")
        try:
            dataset = load_dataset(args.dataset, split=args.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        print(f"Total items: {len(dataset)}")
        sft_source_items = list(dataset)
        
        # Process RL
        print("\nProcessing RL dataset...")
        if args.system_prompt:
            print(f"Using system prompt: {args.system_prompt}")
        rl_all_items = []
        for i, item in enumerate(dataset):
            processed = process_rl(item, i, system_prompt=args.system_prompt)
            if processed:
                rl_all_items.append(processed)
                
        print(f"Generated {len(rl_all_items)} RL examples")
        
        # Split into train (90%) and test (10%)
        rl_dataset = Dataset.from_list(rl_all_items)
        split_dataset = rl_dataset.train_test_split(test_size=0.1, seed=42)
        rl_train_items = split_dataset['train'].to_list()
        rl_test_items = split_dataset['test'].to_list()
    
    # Process SFT
    print("\nProcessing SFT dataset...")
    if args.omit_user_turn:
        print("Omitting user turn in SFT dataset")
    sft_items = create_sft_dataset(sft_source_items, omit_user_turn=args.omit_user_turn)
    print(f"Generated {len(sft_items)} SFT examples (deduplicated laws)")
    
    # Save RL
    rl_save_dir = os.path.join(args.local_dir, "rl")
    os.makedirs(rl_save_dir, exist_ok=True)
    
    rl_train_path = os.path.join(rl_save_dir, "train.parquet")
    rl_test_path = os.path.join(rl_save_dir, "test.parquet")
    
    Dataset.from_list(rl_train_items).to_parquet(rl_train_path)
    Dataset.from_list(rl_test_items).to_parquet(rl_test_path)
    
    print(f"Saved RL train dataset ({len(rl_train_items)} examples) to {rl_train_path}")
    print(f"Saved RL test dataset ({len(rl_test_items)} examples) to {rl_test_path}")
    
    # Save SFT
    sft_save_dir = os.path.join(args.local_dir, "sft")
    os.makedirs(sft_save_dir, exist_ok=True)
    sft_dataset = Dataset.from_list(sft_items)
    print(f"Saving SFT dataset with {len(sft_dataset)} items...")
    sft_output_path = os.path.join(sft_save_dir, "train.parquet")
    sft_dataset.to_parquet(sft_output_path)
    print(f"Saved SFT dataset to {sft_output_path}")

if __name__ == "__main__":
    # python verl/scripts/augment_nitibench.py --add-system_prompt --omit_user_turn --local_dir dataset/verl/wangchan_nitibench --dataset airesearch/WangchanX-Legal-ThaiCCL-RAG
    main()
