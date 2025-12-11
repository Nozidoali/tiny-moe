#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import subprocess
import json
import random
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import evaluate
import numpy as np
from config import GGUF_DIR, MODELS_DIR, SCRIPTS_DIR, RESULTS_DIR, PROMPT_FILES_TRUNCATED_DIR
from utils import generate_text, format_truthfulqa_training, format_truthfulqa_question, format_qmsum_prompt, extract_answer_from_text, load_model_and_tokenizer, load_eval_dataset

def freeze_mlp_only(model):
    for param in model.parameters():
        param.requires_grad = False
    
    # GPT-2 style: transformer.h[i] with c_fc and c_proj
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for layer in model.transformer.h:
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True
            if hasattr(layer, 'c_fc'):
                for param in layer.c_fc.parameters():
                    param.requires_grad = True
            if hasattr(layer, 'c_proj'):
                for param in layer.c_proj.parameters():
                    param.requires_grad = True
    # Other architectures: model.model.layers[i].mlp
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True

class CustomTrainer(Trainer):
    def __init__(self, *args, dataset_name=None, original_model=None, l2_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.original_model = original_model
        self.l2_weight = l2_weight
        self.rouge_metric = None
        if dataset_name in ["truthfulqa", "mixed"]:
            self.metric = evaluate.load('bleurt', 'bleurt-large-128')
        if dataset_name in ["qmsum", "mixed"]:
            if dataset_name == "mixed":
                self.rouge_metric = evaluate.load("rouge")
            elif dataset_name == "qmsum":
                self.metric = evaluate.load("rouge")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.dataset_name == "truthfulqa" and "correct_answers" in inputs:
            inputs.pop("correct_answers", None)
            inputs.pop("incorrect_answers", None)
        
        if self.dataset_name == "qmsum" and "answers" in inputs:
            inputs.pop("answers", None)
        
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        if self.l2_weight > 0.0 and self.original_model is not None:
            l2_reg = 0.0
            for (name, param), (_, orig_param) in zip(model.named_parameters(), self.original_model.named_parameters()):
                if param.requires_grad:
                    l2_reg += torch.sum((param - orig_param) ** 2)
            
            loss = loss + self.l2_weight * l2_reg
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.dataset_name is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        self.model.eval()
        predictions_text = []
        
        eval_ds = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_ds)
        
        for step, inputs in enumerate(eval_dataloader):
            with torch.no_grad():
                input_ids = inputs["input_ids"].to(self.model.device)
                dataset_for_gen = "truthfulqa" if self.dataset_name == "mixed" else self.dataset_name
                generated = generate_text(self.model, self.tokenizer, input_ids, dataset_name=dataset_for_gen)
                pred_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions_text.extend(pred_text)
        
        metrics = {}
        if self.dataset_name == "truthfulqa" or self.dataset_name == "mixed":
            max_scores = []
            acc_scores = []
            for i, pred in enumerate(predictions_text):
                if i < len(eval_ds):
                    ex = eval_ds[i]
                    pred_answer = extract_answer_from_text(pred, "truthfulqa")
                    correct_answers = ex.get('correct_answers', [])
                    incorrect_answers = ex.get('incorrect_answers', [])
                    if correct_answers and incorrect_answers:
                        scores_true = self.metric.compute(predictions=[pred_answer]*len(correct_answers), references=correct_answers)['scores']
                        scores_false = self.metric.compute(predictions=[pred_answer]*len(incorrect_answers), references=incorrect_answers)['scores']
                        max_scores.append(max(scores_true))
                        acc_scores.append(1 if max(scores_true) > max(scores_false) else 0)
            if max_scores:
                metrics = {
                    f"{metric_key_prefix}_bleurt_max_score": np.mean(max_scores),
                    f"{metric_key_prefix}_bleurt_accuracy": np.mean(acc_scores)
                }
        
        if self.dataset_name == "qmsum" or self.dataset_name == "mixed":
            references = []
            predictions_clean = []
            for i, pred in enumerate(predictions_text):
                if i < len(eval_ds):
                    ex = eval_ds[i]
                    if 'answers' in ex:
                        pred_summary = extract_answer_from_text(pred, "qmsum")
                        answers = ex.get("answers", [])
                        ref = answers[0] if isinstance(answers, list) and answers else ""
                        predictions_clean.append(pred_summary)
                        references.append(ref)
            
            if predictions_clean and references and len(predictions_clean) == len(references):
                rouge_eval = self.rouge_metric if self.dataset_name == "mixed" else self.metric
                result = rouge_eval.compute(predictions=predictions_clean, references=references, use_stemmer=True)
                if self.dataset_name == "qmsum":
                    metrics[f"{metric_key_prefix}_rougeL"] = result["rougeL"]
                else:
                    metrics[f"{metric_key_prefix}_rougeL"] = result["rougeL"] if references else 0.0
        
        base_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if isinstance(base_metrics, dict):
            metrics.update(base_metrics)
        
        if f"{metric_key_prefix}_rougeL" not in metrics:
            metrics[f"{metric_key_prefix}_rougeL"] = 0.0
        if f"{metric_key_prefix}_bleurt_max_score" not in metrics:
            metrics[f"{metric_key_prefix}_bleurt_max_score"] = 0.0
        
        self.log(metrics)
        return metrics

def load_original_dataset(dataset_name):
    ds = load_eval_dataset(dataset_name)
    if ds is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds

def load_and_prepare_dataset(dataset_name, tokenizer, max_length=512):
    original_ds = load_original_dataset(dataset_name)
    
    if dataset_name == "truthfulqa":
        expanded_examples = []
        
        for idx, ex in enumerate(original_ds):
            question = ex['question']
            correct_answers = ex['correct_answers'] if ex['correct_answers'] else [ex['best_answer']]
            
            for answer in correct_answers:
                full_text = format_truthfulqa_training(question, answer)
                tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
                
                question_only = format_truthfulqa_question(question)
                question_tokenized = tokenizer(question_only, truncation=True, max_length=max_length, add_special_tokens=False)
                question_length = len(question_tokenized["input_ids"])
                
                labels = tokenized["input_ids"].copy()
                labels[:question_length] = [-100] * question_length
                
                expanded_examples.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                    "correct_answers": ex['correct_answers'],
                    "incorrect_answers": ex['incorrect_answers']
                })
        
        from datasets import Dataset
        expanded_dataset = Dataset.from_list(expanded_examples)
        print(f"Expanded TruthfulQA: {len(original_ds)} questions → {len(expanded_dataset)} training examples")
        return expanded_dataset
    elif dataset_name == "qmsum":
        prompts_dir = Path(PROMPT_FILES_TRUNCATED_DIR)
        def format_fn(ex, idx):
            prompt_file = prompts_dir / f"qmsum_test_{idx}.prompt.txt"
            assert prompt_file.exists(), f"Prompt file not found: {prompt_file}"
            with open(prompt_file, "r", encoding="utf-8", errors="replace") as f:
                prompt = f.read().strip()
            answers = ex.get("answers", [])
            answer = answers[0] if isinstance(answers, list) and answers else ""
            
            full_text = format_qmsum_prompt(prompt, answer)
            tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
            
            prompt_only = format_qmsum_prompt(prompt)
            prompt_tokenized = tokenizer(prompt_only, truncation=True, max_length=max_length)
            prompt_length = len(prompt_tokenized["input_ids"])
            
            labels = tokenized["input_ids"].copy()
            if prompt_length >= len(labels):
                print(f"Warning: Prompt length ({prompt_length}) >= total length ({len(labels)}). Summary was truncated!")
                prompt_length = max(0, len(labels) - 10)
            
            labels[:prompt_length] = [-100] * prompt_length
            
            tokenized["labels"] = labels
            tokenized["answers"] = answers
            return tokenized
        def format_with_idx(ex, idx):
            return format_fn(ex, idx)
        cols_to_remove = [c for c in original_ds.column_names if c not in ['answers']]
        return original_ds.map(format_with_idx, with_indices=True, batched=False, remove_columns=cols_to_remove)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Finetune model experts (MLP layers)")
    parser.add_argument("--input_model", default="gpt2", help="Input HuggingFace model path")
    parser.add_argument("--output_dir", default=MODELS_DIR, help="Output directory for finetuned model")
    parser.add_argument("--dataset", default="truthfulqa", choices=["truthfulqa", "qmsum", "mixed"], help="Dataset name")
    parser.add_argument("--unfreeze_all", action="store_true", help="Unfreeze all layers for full model finetuning (default: only MLP layers)")
    parser.add_argument("--max_length", type=int, default=2028, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for regularization")
    parser.add_argument("--eval_split", type=float, default=0.1, help="Eval split ratio")
    parser.add_argument("--no_split", action="store_true", help="Use entire dataset for both training and evaluation (no split)")
    parser.add_argument("--eval_every_n_epochs", type=float, default=100, help="Evaluate every N epochs (can be fractional, e.g., 0.5 for twice per epoch)")
    parser.add_argument("--l2_weight", type=float, default=0.01, help="L2 regularization weight to keep model close to original (prevents catastrophic forgetting)")
    parser.add_argument("--balance_datasets", action="store_true", help="Balance datasets when using mixed mode (downsample larger to match smaller)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer using shared utility function
    torch_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    model, tokenizer, device = load_model_and_tokenizer(args.input_model, torch_dtype=torch_dtype)
    
    original_model = None
    if args.l2_weight > 0.0:
        print(f"Loading original model for L2 regularization (weight={args.l2_weight})...")
        original_model = AutoModelForCausalLM.from_pretrained(
            args.input_model,
            torch_dtype=torch.float32
        )
        original_model = original_model.to(device)
        for param in original_model.parameters():
            param.requires_grad = False
    
    if args.unfreeze_all:
        print("Unfreezing all layers for full model finetuning...")
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    else:
        print("Freezing all except MLP layers...")
        freeze_mlp_only(model)
    
    print(f"Loading dataset: {args.dataset}")
    
    if args.dataset == "mixed":
        print("Loading mixed dataset (TruthfulQA + QMSum)...")
        from datasets import concatenate_datasets
        
        dataset_tqa_full = load_and_prepare_dataset("truthfulqa", tokenizer, args.max_length)
        dataset_qms_full = load_and_prepare_dataset("qmsum", tokenizer, args.max_length)
        
        print(f"  TruthfulQA samples: {len(dataset_tqa_full)}")
        print(f"  QMSum samples: {len(dataset_qms_full)}")
        
        if args.balance_datasets:
            min_samples = min(len(dataset_tqa_full), len(dataset_qms_full))
            print(f"  Balancing datasets to {min_samples} samples each...")
            dataset_tqa_full = dataset_tqa_full.shuffle(seed=42).select(range(min_samples))
            dataset_qms_full = dataset_qms_full.shuffle(seed=42).select(range(min_samples))
        
        if args.no_split:
            combined = concatenate_datasets([dataset_tqa_full, dataset_qms_full])
            dataset = {"train": combined, "test": combined}
        else:
            dataset_tqa = dataset_tqa_full.train_test_split(test_size=args.eval_split, seed=42)
            dataset_qms = dataset_qms_full.train_test_split(test_size=args.eval_split, seed=42)
            
            train_combined = concatenate_datasets([dataset_tqa["train"], dataset_qms["train"]])
            test_combined = concatenate_datasets([dataset_tqa["test"], dataset_qms["test"]])
            dataset = {"train": train_combined, "test": test_combined}
        
        print(f"  Combined training samples: {len(dataset['train'])}")
        print(f"  Combined eval samples: {len(dataset['test'])}")
    else:
        dataset = load_and_prepare_dataset(args.dataset, tokenizer, args.max_length)
        
        if args.no_split:
            dataset = {"train": dataset, "test": dataset}
        else:
            dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)
    
    debug_dir = Path(RESULTS_DIR) / "dataset" / args.dataset
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed dataset to: {debug_dir}")
    for split_name in ["train", "test"]:
        split_file = debug_dir / f"{split_name}.json"
        examples = []
        for i, ex in enumerate(dataset[split_name]):
            decoded_input = tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
            
            labels_without_mask = [label if label != -100 else tokenizer.pad_token_id for label in ex["labels"]]
            decoded_labels = tokenizer.decode(labels_without_mask, skip_special_tokens=False)
            
            num_masked = sum(1 for label in ex["labels"] if label == -100)
            
            example_dict = {
                "index": i,
                "input_ids_length": len(ex["input_ids"]),
                "labels_masked_tokens": num_masked,
                "decoded_input": decoded_input,
                "decoded_labels": decoded_labels,
            }
            if args.dataset == "truthfulqa":
                example_dict["correct_answers"] = ex.get("correct_answers", [])
                example_dict["incorrect_answers"] = ex.get("incorrect_answers", [])
            elif args.dataset == "qmsum":
                example_dict["answers"] = ex.get("answers", [])
            examples.append(example_dict)
        
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(examples)} examples to {split_file}")
    
    
    timestamp = datetime.now().strftime("%H%M")
    model_subdir = output_dir / f"{args.dataset}_{timestamp}"
    model_subdir.mkdir(parents=True, exist_ok=True)
    
    eval_steps = None
    metric_for_best_model = None
    greater_is_better = True
    if args.eval_every_n_epochs > 0:
        steps_per_epoch = len(dataset["train"]) // (args.batch_size * args.grad_accum)
        
        if args.eval_every_n_epochs >= args.num_epochs:
            print(f"Warning: eval_every_n_epochs ({args.eval_every_n_epochs}) >= num_epochs ({args.num_epochs})")
            print(f"Setting eval_every_n_epochs to {args.num_epochs / 2} for at least 2 evaluations")
            args.eval_every_n_epochs = max(1.0, args.num_epochs / 2)
        
        eval_steps = int(steps_per_epoch * args.eval_every_n_epochs)
        if args.dataset == "truthfulqa":
            metric_for_best_model = "eval_bleurt_max_score"
        elif args.dataset == "qmsum":
            metric_for_best_model = "eval_rougeL"
        elif args.dataset == "mixed":
            metric_for_best_model = "eval_rougeL"
    
    tensorboard_log_dir = model_subdir / "logs"
    
    training_args = TrainingArguments(
        output_dir=str(model_subdir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=str(tensorboard_log_dir),
        logging_steps=args.grad_accum,
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if eval_steps else "no",
        save_steps=eval_steps,
        load_best_model_at_end=True if metric_for_best_model else False,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=42,
        prediction_loss_only=False,
    )
    
    from transformers import default_data_collator
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        original_model=original_model,
        l2_weight=args.l2_weight,
    )
    
    print("Starting training...")
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    print(f"To view logs, run: tensorboard --logdir {tensorboard_log_dir}")
    trainer.train()
    
    print(f"Saving model to: {model_subdir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(model_subdir))
    checkpoint_path = str(model_subdir)
    
    print("Converting to GGUF format...")
    convert_script = Path(SCRIPTS_DIR) / "convert.sh"
    model_name = model_subdir.name
    
    subprocess.run([
        "bash", str(convert_script),
        "--checkpoint", checkpoint_path,
        "--name", model_name,
        "--output-dir", GGUF_DIR
    ], check=True)
    
    print("Done!")

if __name__ == "__main__":
    main()
