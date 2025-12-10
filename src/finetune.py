#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import subprocess
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from config import GGUF_DIR, MODELS_DIR, SCRIPTS_DIR

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

def load_and_prepare_dataset(dataset_name, tokenizer, max_length=512):
    if dataset_name == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        def format_fn(ex):
            text = f"Question: {ex['question']}\nAnswer: {ex['best_answer']}"
            tokenized = tokenizer(text, truncation=True, max_length=max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
    elif dataset_name == "qmsum":
        ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
        ds = ds.select(range(min(50, len(ds))))
        def format_fn(ex):
            context = ex.get("context", "")
            input_text = ex.get("input", "")
            answers = ex.get("answers", [])
            answer = answers[0] if isinstance(answers, list) and answers else ""
            prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
            text = f"{prompt}\n\nSummary: {answer}"
            tokenized = tokenizer(text, truncation=True, max_length=max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds.map(format_fn, batched=False, remove_columns=ds.column_names)

def main():
    parser = argparse.ArgumentParser(description="Finetune model experts (MLP layers)")
    parser.add_argument("--input_model", default="gpt2", help="Input HuggingFace model path")
    parser.add_argument("--output_dir", default=MODELS_DIR, help="Output directory for finetuned model")
    parser.add_argument("--dataset", default="truthfulqa", choices=["truthfulqa", "qmsum"], help="Dataset name")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA instead of full MLP finetuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_split", type=float, default=0.2, help="Eval split ratio")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {args.input_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.input_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.input_model,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    if args.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Freezing all except MLP layers...")
        freeze_mlp_only(model)
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_and_prepare_dataset(args.dataset, tokenizer, args.max_length)
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=42,
    )
    
    from transformers import default_data_collator
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to: {output_dir}")
    if args.use_lora:
        model.save_pretrained(str(output_dir))
        merged_model = model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"LoRA adapter saved to: {output_dir}")
        print(f"Merged model saved to: {merged_dir}")
        checkpoint_path = str(merged_dir)
    else:
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
        checkpoint_path = str(output_dir)
    
    print("Converting to GGUF format...")
    convert_script = Path(SCRIPTS_DIR) / "convert.sh"
    model_name = output_dir.name
    
    subprocess.run([
        "bash", str(convert_script),
        "--checkpoint", checkpoint_path,
        "--name", model_name,
        "--output-dir", GGUF_DIR
    ], check=True)
    
    print("Done!")

if __name__ == "__main__":
    main()
