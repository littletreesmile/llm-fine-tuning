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
    get_scheduler,
    BatchEncoding
)
import evaluate
from peft import LoraConfig, get_peft_model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from prepare_data import load_and_prepare_data, format_prompt



def prepare_model():
    access_token = None
    # Load model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

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
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as pad token
    tokenizer.padding_side = "right"  # Important for causal LM
    # model.resize_token_embeddings(len(tokenizer))

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # All linear layers
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Properly tokenizes instruction-following data for Llama 3
    Handles both cases: with and without input context
    """
    # Initialize lists to store processed data
    full_texts = []
    prompt_lengths = []

    # Process each example in the batch
    for instruction, input_text, output in zip(examples['instruction'],
                                               examples['input'],
                                               examples['output']):
        prompt = format_prompt({'instruction': instruction, 'input': input_text, 'output': output})

        # Combine prompt with response
        full_text = prompt + f"Response: {output}<|eot_id|>"
        full_texts.append(full_text)

        # Tokenize just the prompt to get its length
        prompt_tokenized = tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=False  # Don't add extra special tokens
        )
        prompt_lengths.append(len(prompt_tokenized['input_ids']))

    # Tokenize all full texts (prompt + response)
    tokenized_inputs = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Create labels by masking the prompt part
    labels = tokenized_inputs['input_ids'].clone()
    for i, prompt_len in enumerate(prompt_lengths):
        # Only calculate loss on the response part
        labels[i, :prompt_len] = -100

        # Also ensure we don't calculate loss on padding tokens
        # Find the first pad token (if any)
        pad_mask = (tokenized_inputs['input_ids'][i] == tokenizer.pad_token_id)
        if pad_mask.any():
            first_pad = pad_mask.nonzero()[0].item()
            labels[i, first_pad:] = -100

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

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

device = torch.device('cuda')
# Load data
train_data, eval_data = load_and_prepare_data()

# Prepare model and tokenizer
model, tokenizer = prepare_model()

# Tokenize datasets
tokenized_train = train_data.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    batch_size=2,
    remove_columns=train_data.column_names
)
# tokenized_eval = eval_data.map(
#     lambda x: tokenize_function(x, tokenizer),
#     batched=True,
#     remove_columns=eval_data.column_names,
#     batch_size = 4,
# )

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# Recommended approach - use the Trainer
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-4,
    fp16=True,
    gradient_accumulation_steps=2,  # Helps with memory,
    label_names=["labels"],
    disable_tqdm=False,  # Ensure progress bar is enabled
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator,
    # tokenizer=tokenizer,
)

trainer.train()

# Save model and tokenizer
trainer.save_model("./finetuning_trainer")
tokenizer.save_pretrained("./finetuning_trainer")