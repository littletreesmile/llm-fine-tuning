
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force offline mode

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch
from prepare_data import load_and_prepare_data, format_prompt

def prepare_model():
    access_token = None
    # Load model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # replace with actual model name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # quantize in 4-bit
        bnb_4bit_use_double_quant=True,  # optional but recommended
        bnb_4bit_quant_type="nf4",  # "nf4" or "fp4"
        bnb_4bit_compute_dtype=torch.float16  # or torch.bfloat16 if supported
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    prompts = [format_prompt(dict(zip(examples.keys(), ex))) for ex in zip(*examples.values())]
    responses = examples['output']
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Tokenize outputs
    tokenized_outputs = tokenizer(
        responses,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_outputs["input_ids"]
    }

def main():
    # Load data
    train_data, eval_data = load_and_prepare_data()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model()

    # Tokenize datasets
    tokenized_train = train_data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=True,
        remove_columns=train_data.column_names
    )
    tokenized_eval = eval_data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_data.column_names
    )


    # Training arguments
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-4,
        fp16=True,
        label_names=["labels"]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),

    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model("./singapore_llama")
    tokenizer.save_pretrained("./singapore_llama")

if __name__ == "__main__":
    main()