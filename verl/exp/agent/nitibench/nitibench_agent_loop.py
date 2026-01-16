import argparse
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs

def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("question")
        answer = example.pop("answer")
        
        # System prompt to encourage tool use
        system_prompt = (
            "You are a legal expert in Thai law. You are given a legal question and you need to answer it. "
            "You have access to a `search_law` tool that can search for relevant Thai laws, "
            "and a `read_law` tool that can read the full content of a specific law section. "
            "You should use the `search_law` tool to find relevant laws, and then use `read_law` to get the full text if needed. "
            "Reason step by step before using the tool. "
            "After gathering information, provide your final answer."
        )

        data = {
            "data_source": "nitibench",
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "law",
            "reward_model": {
                "style": "llm_judge",
                "ground_truth": answer,
                "expected_format": {"type": "none"},
                "guideline": (
                    "Decide whether the response correctly answers the legal question in Thai law. "
                    "Use the reference answer as the gold standard. "
                    "A correct answer must match the reference answer in meaning, including key legal consequences "
                    "(e.g., penalty, conditions, legal basis), even if wording differs.\n\n"
                    f"Reference answer (gold): {answer}"
                ),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "search_law": {"_dummy": 0},
                    "read_law": {"_dummy": 0}
                },
            },
        }
        return data
    return process_fn

def make_rag_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("question")
        # RAG dataset uses 'positive_answer'
        answer = example.pop("positive_answer")
        
        # System prompt to encourage tool use
        system_prompt = (
            "You are a legal expert in Thai law. You are given a legal question and you need to answer it. "
            "You have access to a `search_law` tool that can search for relevant Thai laws, "
            "and a `read_law` tool that can read the full content of a specific law section. "
            "You should use the `search_law` tool to find relevant laws, and then use `read_law` to get the full text if needed. "
            "Reason step by step before using the tool. "
            "After gathering information, provide your final answer."
        )

        data = {
            "data_source": "wangchan_rag",
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "law",
            "reward_model": {
                "style": "llm_judge",
                "ground_truth": answer,
                "expected_format": {"type": "none"},
                "guideline": (
                    "Decide whether the response correctly answers the legal question in Thai law. "
                    "Use the reference answer as the gold standard. "
                    "A correct answer must match the reference answer in meaning, including key legal consequences "
                    "(e.g., penalty, conditions, legal basis), even if wording differs.\n\n"
                    f"Reference answer (gold): {answer}"
                ),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "search_law": {"_dummy": 0},
                    "read_law": {"_dummy": 0}
                },
            },
        }
        return data
    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load dataset
    train_dataset = datasets.load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG", split="train")
    test_dataset = datasets.load_dataset("VISAI-AI/nitibench", split="ccl")

    # Process datasets
    train_dataset = train_dataset.map(make_rag_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True)

    # Save to parquet
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    
    if args.hdfs_dir:
        copy(os.path.join(args.local_dir, "train.parquet"), os.path.join(args.hdfs_dir, "train.parquet"))
        copy(os.path.join(args.local_dir, "test.parquet"), os.path.join(args.hdfs_dir, "test.parquet"))
