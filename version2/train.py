import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force offline mode

from tqdm.auto import tqdm
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    get_scheduler
)
import evaluate
from peft import LoraConfig, get_peft_model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from prepare_data import load_and_prepare_data, format_prompt


def prepare_model():
    access_token = "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"
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


def compute_metrics(eval_preds, tokenizer):
    rouge = evaluate.load("rouge")
    predictions, labels = eval_preds

    # Decode predictions and labels to text (assuming you have a tokenizer)
    predicted_ids = np.argmax(predictions, axis=-1)  # Get token IDs (argmax)

    predicted_texts = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    rouge_results = rouge.compute(predictions=predicted_texts, references=label_texts)

    # Returning ROUGE scores, you can include additional metrics if needed
    return rouge_results


def main():
    device = torch.device('cuda')
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

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=2, collate_fn=data_collator)

    num_epochs = 3
    total_loss = 0.0
    num_steps = 0


    num_training_steps = num_epochs * len(train_dataloader)



    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)
    progress_bar = tqdm()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)
            # Update progress bar description with the current average loss
            num_steps += 1
            progress_bar.set_postfix(loss=total_loss / num_steps)

        # Optionally, print out the average loss after training
        average_loss = total_loss / num_steps
        print(f"Epoch {epoch + 1} completed. Avg loss: {average_loss:.4f}\n")












    # # Training arguments
    # training_args = TrainingArguments(
    #     output_dir="results",
    #     num_train_epochs=3,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=10,
    #     eval_strategy="steps",
    #     eval_steps=1,
    #     save_strategy="steps",
    #     save_steps=500,
    #     learning_rate=2e-4,
    #     fp16=True,
    #     label_names=["labels"]
    # )
    #
    # # Initialize trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_train,
    #     eval_dataset=tokenized_eval,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer)
    #
    # )
    #
    # # Train model
    # trainer.train()
    #
    # # Save model and tokenizer
    # trainer.save_model("./singapore_llama")
    # tokenizer.save_pretrained("./singapore_llama")


if __name__ == "__main__":
    main()