#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import numpy as np
from pathlib import Path
import evaluate
from utils import load_model_and_tokenizer, generate_text, format_truthfulqa_question, format_qmsum_prompt, extract_answer_from_text, load_eval_dataset
from config import RESULTS_DIR, PROMPT_FILES_TRUNCATED_DIR


def eval_truthfulqa(model, tokenizer, device):
    ds = load_eval_dataset("truthfulqa")
    bleurt = evaluate.load('bleurt', 'bleurt-large-128')
    
    max_scores = []
    acc_scores = []
    
    for i, rec in enumerate(ds):
        question = rec['question']
        correct_answers = rec['correct_answers']
        incorrect_answers = rec['incorrect_answers']
        
        prompt = format_truthfulqa_question(question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        generated = generate_text(model, tokenizer, input_ids, dataset_name="truthfulqa")
        pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        pred_answer = extract_answer_from_text(pred_text, "truthfulqa")
        
        scores_true = bleurt.compute(predictions=[pred_answer]*len(correct_answers), references=correct_answers)['scores']
        scores_false = bleurt.compute(predictions=[pred_answer]*len(incorrect_answers), references=incorrect_answers)['scores']
        
        max_score = max(scores_true)
        acc = 1 if max(scores_true) > max(scores_false) else 0
        
        max_scores.append(max_score)
        acc_scores.append(acc)
        
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(ds)}] Acc: {np.mean(acc_scores):.3f}, Score: {np.mean(max_scores):.3f}")
    
    return {
        "accuracy": np.mean(acc_scores),
        "bleurt_max_score": np.mean(max_scores)
    }


def eval_longbench(model, tokenizer, device, dataset_name="qmsum"):
    ds = load_eval_dataset(dataset_name)
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    prompts_dir = Path(PROMPT_FILES_TRUNCATED_DIR)
    
    for i, rec in enumerate(ds):
        prompt_file = prompts_dir / f"{dataset_name}_test_{i}.prompt.txt"
        
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = rec.get("context", "")
        
        prompt_formatted = format_qmsum_prompt(prompt)
        input_ids = tokenizer(prompt_formatted, return_tensors="pt", truncation=True, max_length=2028).input_ids.to(device)
        
        generated = generate_text(model, tokenizer, input_ids, dataset_name="qmsum")
        pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        pred_summary = extract_answer_from_text(pred_text, "qmsum")
        
        answers = rec.get("answers", [])
        ref = answers[0] if isinstance(answers, list) and answers else ""
        
        predictions.append(pred_summary)
        references.append(ref)
        
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{len(ds)}] Processed")
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    return {
        "rougeL": result["rougeL"],
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model path or name")
    parser.add_argument("--dataset", choices=["truthfulqa", "longbench", "qmsum"], default="truthfulqa")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    model.eval()
    
    print(f"Evaluating on: {args.dataset}")
    
    if args.dataset == "truthfulqa":
        results = eval_truthfulqa(model, tokenizer, device)
        print("\n=== TruthfulQA Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"BLEURT Max Score: {results['bleurt_max_score']:.4f}")
    else:
        dataset_name = "qmsum" if args.dataset == "longbench" else args.dataset
        results = eval_longbench(model, tokenizer, device, dataset_name)
        print(f"\n=== {dataset_name.upper()} Results ===")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")


if __name__ == "__main__":
    main()
