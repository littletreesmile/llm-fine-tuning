###

import torch
from datasets import load_dataset
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from utils.tools import convert_to_alpaca_format
from version2.prepare_data import format_prompt

MAX_LENGTH = 512

def generate_response(prompt):
    tokenizer.pad_token = tokenizer.eos_token  # Before generation
    tokenizer.padding_side = "right"  # Ensure consistent padding
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
    outputs = model.generate(**inputs,
                             max_new_tokens=200,
                             pad_token_id=tokenizer.eos_token_id,
                             temperature=0.7,
                             top_k=40,  # Reduce from default 50
                             top_p=0.9,  # Faster than 0.95
                             num_beams=1,  # Disable beam search (slow)
                             use_cache=True  # Critical for speed
                             )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_finetuning_model(model_name, finetuning_model_path):
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # quantize in 4-bit
                bnb_4bit_use_double_quant=True,  # optional but recommended
                bnb_4bit_quant_type="nf4",  # "nf4" or "fp4"
                bnb_4bit_compute_dtype=torch.float16  # or torch.bfloat16 if supported
        )

        base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto"
        )
        print("Successfully loaded base model from cache!")

        model = PeftModel.from_pretrained(base_model, finetuning_model_path)
        tokenizer = AutoTokenizer.from_pretrained(finetuning_model_path, use_fast=True)
        return model, tokenizer


def format_output(full_response):
    # Split into user/assistant parts
    parts = full_response.split("assistant\n\n")
    if len(parts) > 1:
        # Extract question (last user instruction)
        user_part = parts[0]
        question = user_part.split("Instruction: ")[-1].strip()
        # Extract answer (first assistant response)
        answer = parts[1].split("<|eot_id|>")[0].strip()  # Stop at EOT token if present
    else:
        question = "Unknown question"
        answer = full_response.strip()
    # Format for display
    return f"Q: {question}\nA: {answer}"


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # replace with actual model name
FINETUNING_MODEL_PATH = "./lora_adapters"
TEST_DATASET = "SeaEval/CRAFT-Singapore-Instruction-GPT4"  # Your test dataset

# Load fine-tuning model and tokenizer
model, tokenizer = load_finetuning_model(MODEL_NAME, FINETUNING_MODEL_PATH)

# Load test data
dataset = load_dataset(TEST_DATASET)  # Small subset for testing
eval_dataset = dataset['train'].select(range(6))
eval_dataset = eval_dataset.map(convert_to_alpaca_format)  # Convert and split dataset

# Load metrics
# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Evaluation
all_predictions = []
all_references = []
losses = []
for instruction, input_text, output in tqdm(zip(eval_dataset['instruction'],
                        eval_dataset['input'],
                        eval_dataset['output']),
                    desc="Evaluating"):
        example = {'instruction': instruction, 'input': input_text, 'output': output}

        # Format prompt (same as during training)
        prompt = format_prompt(example)

        # Generate prediction
        prediction = generate_response(prompt)
        # print(format_output(prediction))
        # print(format_output(example["output"]))
        # print('-----------------')
        all_predictions.append(prediction)
        all_references.append(example["output"])


        # Calculate loss for perplexity
        inputs = tokenizer(prompt + example["output"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(
                "cuda")
        with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

# Calculate metrics
bleu_results = bleu.compute(predictions=all_predictions, references=[[ref] for ref in all_references])
rouge_results = rouge.compute(predictions=all_predictions, references=all_references)
avg_loss = sum(losses) / len(losses)
ppl = torch.exp(torch.tensor(avg_loss)).item()  # Perplexity = e^(avg loss)

print(f"\n=== Evaluation Results ===")
print(f"BLEU: {bleu_results['bleu']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"Perplexity: {ppl:.2f}")
print(f"Avg Loss: {avg_loss:.4f}")
