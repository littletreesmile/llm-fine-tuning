from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch
from train import prepare_model
from prepare_data import format_prompt


# Path to your fine-tuned 4-bit model
model_path = "singapore_llama"


def loaded_fine_tuning_model(model_path):
    # Define 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer


def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load original model and tokenizer
# model, tokenizer = prepare_model()

# Load fine-tuning model and tokenizer
model, tokenizer = loaded_fine_tuning_model(model_path)

model.eval()

# Single Zero Shot prompting
# example = {"instruction": "What are some interesting places to visit in Singapore?"}
# prompt = format_prompt(example)
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # Tokenize input
# response = generate_response(prompt, model, tokenizer)
# print(response)

# Multiple Zero Shot prompting
# Load dataset from HuggingFace
dataset = load_dataset("SeaEval/CRAFT-Singapore-Instruction-GPT4")
for example in dataset['train']:
    prompt = format_prompt(example)
    response = generate_response(prompt, model, tokenizer)
    print(response)
