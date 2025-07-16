# model_loader.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B"
CACHE_DIR = "./models/"
OFFLOAD_DIR = "./offload"

def load_model_and_tokenizer():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    return model, tokenizer
