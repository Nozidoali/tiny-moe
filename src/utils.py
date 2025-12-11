#!/usr/bin/env python3
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 128,
    "temperature": 0.5,
    "top_p": 0.7,
    "repeat_penalty": 1.5,
    "repeat_last_n": 128,
    "do_sample": False,
    "seed": 42,
}

def load_model_and_tokenizer(model_path, torch_dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    config_path = Path(model_path) / "config.json"
    is_tinymoe = False
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            is_tinymoe = (config.get('model_type') == 'TinyMoE')
    
    if is_tinymoe:
        from tinymoe import AutoModelForTinyMoE
        model = AutoModelForTinyMoE.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device

def generate_text(model, tokenizer, input_ids, dataset_name=None, **kwargs):
    params = DEFAULT_GEN_PARAMS.copy()
    if dataset_name == "truthfulqa":
        params["max_new_tokens"] = 30
    elif dataset_name == "qmsum":
        params["max_new_tokens"] = 128
    params.update(kwargs)
    
    generation_kwargs = {
        "max_new_tokens": params["max_new_tokens"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": params["do_sample"],
    }
    
    if params["do_sample"]:
        generation_kwargs.update({
            "temperature": params["temperature"],
            "top_p": params["top_p"],
        })
    
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'repetition_penalty'):
        generation_kwargs["repetition_penalty"] = params["repeat_penalty"]
    
    return model.generate(input_ids, **generation_kwargs)

def get_cli_args(dataset_name=None, **kwargs):
    params = DEFAULT_GEN_PARAMS.copy()
    if dataset_name == "truthfulqa":
        params["max_new_tokens"] = 30
    elif dataset_name == "qmsum":
        params["max_new_tokens"] = 128
    params.update(kwargs)
    
    return [
        "-n", str(params["max_new_tokens"]),
        "--repeat-penalty", str(params["repeat_penalty"]),
        "--repeat-last-n", str(params["repeat_last_n"]),
        "--temp", str(params["temperature"]),
        "--top-p", str(params["top_p"]),
        "--seed", str(params["seed"]),
    ]

def format_truthfulqa_question(question: str) -> str:
    question = question.replace("'", " ").replace('"', ' ')
    return f"Question: {question}\nAnswer:"

def format_truthfulqa_training(question: str, answer: str) -> str:
    return f"{format_truthfulqa_question(question)} {answer}"

def format_qmsum_prompt(prompt: str, summary: str = None) -> str:
    if summary:
        return f"{prompt}\n\nSummary: {summary}"
    return prompt.strip()

def extract_answer_from_text(text: str, dataset_name: str) -> str:
    if dataset_name == "truthfulqa":
        return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
    elif dataset_name == "qmsum":
        return text.split("Summary:")[-1].strip() if "Summary:" in text else text.strip()
    return text.strip()

def load_eval_dataset(dataset_name):
    if dataset_name == "truthfulqa":
        return load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    elif dataset_name in ["qmsum", "longbench"]:
        ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
        return ds.select(range(min(50, len(ds))))
    return None
