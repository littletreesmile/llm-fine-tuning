from huggingface_hub import snapshot_download
from huggingface_hub import login
login(token="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")  # Paste your token here
snapshot_download("meta-llama/Meta-Llama-3-8B-Instruct", local_dir="./local_model")