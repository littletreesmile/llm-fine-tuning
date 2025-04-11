

# Convert to Alpaca format
def convert_to_alpaca_format(example):
    return {
        "instruction": example["instruction"],
        "input": example.get("input", ""),  # if input exists
        "output": example["output"]
    }