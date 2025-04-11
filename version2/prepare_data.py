
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import load_dataset
import json
from utils.tools import convert_to_alpaca_format

def load_and_prepare_data():
    # Load dataset from HuggingFace
    dataset = load_dataset("SeaEval/CRAFT-Singapore-Instruction-GPT4")

    dataset = dataset['train'].select(range(6))

    # Convert and split dataset
    train_data = dataset.map(convert_to_alpaca_format)
    eval_data = dataset.map(convert_to_alpaca_format)
    
    return train_data, eval_data

# def format_prompt(example):
#     """Format prompt for Llama instruction model"""
#     if "input" in example and example["input"]:
#         prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
#     else:
#         prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
#     return prompt

def format_prompt(example):
    instruction = example['instruction']
    input_text = example['input']

    # Format the prompt based on whether input exists
    if input_text.strip():  # With input context
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Instruction: {instruction}\n"
            f"Input: {input_text}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:  # Without input context
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Instruction: {instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return prompt

if __name__ == "__main__":
    train_data, eval_data = load_and_prepare_data()
    print(f"Train size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")