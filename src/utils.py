#!/usr/bin/env python3
import torch

DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 128,
    "temperature": 0.5,
    "top_p": 0.7,
    "repeat_penalty": 1.5,
    "repeat_last_n": 128,
    "do_sample": False,
    "seed": 42,
}

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
    elif dataset_name == "mmlu":
        params["max_new_tokens"] = 5
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

def format_mmlu_question(question: str, choices: list) -> str:
    question = question.replace("'", " ").replace('"', ' ')
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    return f"Question: {question}\n{choices_text}\nAnswer:"

def extract_answer_from_text(text: str, dataset_name: str) -> str:
    if dataset_name == "truthfulqa":
        return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
    elif dataset_name == "qmsum":
        return text.split("Summary:")[-1].strip() if "Summary:" in text else text.strip()
    elif dataset_name == "mmlu":
        return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
    return text.strip()
