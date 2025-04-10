
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import load_dataset
import json

def load_and_prepare_data():
    # Load dataset from HuggingFace
    dataset = load_dataset("SeaEval/CRAFT-Singapore-Instruction-GPT4")
    
    # Convert to Alpaca format
    def convert_to_alpaca_format(example):
        return {
            "instruction": example["instruction"],
            "input": example.get("input", ""),  # if input exists
            "output": example["output"]
        }

    dataset = dataset['train'].select(range(5))

    # Convert and split dataset
    train_data = dataset.map(convert_to_alpaca_format)
    eval_data = dataset.map(convert_to_alpaca_format)
    
    return train_data, eval_data

def format_prompt(example):
    """Format prompt for Llama instruction model"""
    if "input" in example and example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return prompt

if __name__ == "__main__":
    train_data, eval_data = load_and_prepare_data()
    print(f"Train size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")