#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import argparse
import random
import tempfile
import time
import shutil
import re
from pathlib import Path
from datasets import load_dataset
import evaluate
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import TINYMOE_DIR, GGUF_DIR, RESULTS_DIR, SCRIPTS_DIR, MODELS_DIR, PROMPT_FILES_TRUNCATED_DIR
from utils import load_model_and_tokenizer

QUANT_LEVELS = [
    "q2_k", 
    # "q2_k_s",
    "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k",
    "q8_0",
    # "iq1_s", 
    # "iq1_m",
    # "iq2_xxs", "iq2_xs", "iq2_s", 
    # "iq2_m",
    # "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m",
    # "iq4_nl", "iq4_xs",
    "tq1_0", "tq2_0",
    "f16", "bf16"
]

def load_and_save_model(checkpoint_path, name):
    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path)
    
    model_dir = Path(MODELS_DIR) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {model_dir}...")
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    
    return str(model_dir)

def run_convert(checkpoint_path, name, quant_levels):
    script = Path(SCRIPTS_DIR) / "convert.sh"
    quant_str = ",".join(quant_levels)
    cmd = ["bash", str(script), "--checkpoint", checkpoint_path, "--name", name, "--quantize", quant_str]
    subprocess.run(cmd, check=True)

def eval_truthfulqa(model_name, num_samples=100):
    from truthful_qa_eval import run_evaluate
    from datasets import load_dataset
    
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    n = len(ds)
    indices = random.sample(range(n), min(num_samples, n))
    
    timestamp = time.strftime("%H%M")
    output_dir = Path(RESULTS_DIR) / f"outputs_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["M"] = model_name
    
    max_scores = []
    acc_scores = []
    bleurt = evaluate.load('bleurt', 'bleurt-large-128')
    
    for i in tqdm(indices, desc="TruthfulQA", unit="sample"):
        rec = ds[i]
        question = rec['question']
        correct_answers = rec['correct_answers']
        incorrect_answers = rec['incorrect_answers']
        
        from utils import format_truthfulqa_question
        question = format_truthfulqa_question(question)
        
        output_path = output_dir / f"tqa_{i}.txt"
        cli_path = f"{SCRIPTS_DIR}/run-cli-local.sh"
        from utils import get_cli_args
        cmd = ["bash", cli_path, "-no-cnv", "-p", f"\"'{question}'\"", "-n", "30"] + get_cli_args("truthfulqa")
        
        with open(output_path, "w", encoding="utf-8", errors='replace') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, env=os.environ, check=True)
        
        with open(output_path, "r", encoding='utf-8', errors='replace') as f:
            pred = f.read().strip()
        
        scores_true = bleurt.compute(predictions=[pred]*len(correct_answers), references=correct_answers)['scores']
        scores_false = bleurt.compute(predictions=[pred]*len(incorrect_answers), references=incorrect_answers)['scores']
        
        max_score = max(scores_true)
        acc = int(max(scores_true) > max(scores_false))
        max_scores.append(max_score)
        acc_scores.append(acc)
    
    return np.mean(acc_scores), np.mean(max_scores)

def eval_longbench(model_name, num_samples=20, prompt_dir=None):
    from longbench_test import run_all
    from longbench_eval import evaluate_folder_outputs, load_references
    
    if prompt_dir is None:
        prompt_dir = PROMPT_FILES_TRUNCATED_DIR
    
    prompt_path = Path(prompt_dir)
    prompt_files = sorted(prompt_path.glob("qmsum_test_*.prompt.txt"))
    
    if len(prompt_files) == 0:
        raise ValueError(f"No prompt files found in {prompt_dir}")
    
    selected_files = random.sample(prompt_files, min(num_samples, len(prompt_files)))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sampled_prompt_dir = Path(tmpdir) / "prompts"
        sampled_prompt_dir.mkdir()
        
        for pf in selected_files:
            shutil.copy2(pf, sampled_prompt_dir / pf.name)
        
        output_dir = Path(RESULTS_DIR) / f"qmsum_outputs_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cli_path = f"{SCRIPTS_DIR}/run-cli-local.sh"
        os.environ["M"] = model_name
        
        latencies, _ = run_all(
            str(sampled_prompt_dir), str(sampled_prompt_dir), str(output_dir),
            cli_path, [], model_name, True
        )
        
        ref_map_full = load_references()
        pattern = re.compile(r"qmsum_test_(\d+)\.prompt\.txt")
        ref_map = {}
        for pf in selected_files:
            m = pattern.match(pf.name)
            if m:
                idx = int(m.group(1))
                if idx in ref_map_full:
                    ref_map[idx] = ref_map_full[idx]
        
        _, aggregated = evaluate_folder_outputs(str(output_dir), ref_map)
        return aggregated['rougeL']

def eval_speed(model_name, device_serial=None):
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from speed_eval import run_speed_benchmark, parse_benchmark_output, push_model_to_device
        model_path = Path(GGUF_DIR) / model_name
        if not model_path.exists():
            tqdm.write(f"  Model {model_name} not found for speed evaluation")
            return None
        pushed_name = push_model_to_device(str(model_path), device_serial)
        stdout = run_speed_benchmark(pushed_name, device_serial)
        metrics = parse_benchmark_output(stdout)
        return metrics.get('tg_speed') or metrics.get('pp_speed')
    except Exception as e:
        tqdm.write(f"  Speed evaluation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--name", required=True, help="Model name for output files")
    parser.add_argument("--quant-levels", nargs="+", default=QUANT_LEVELS, help="Quantization levels")
    parser.add_argument("--tqa-samples", type=int, default=50, help="TruthfulQA samples")
    parser.add_argument("--longbench-samples", type=int, default=20, help="LongBench samples")
    parser.add_argument("--serial", type=str, help="Device serial number for speed evaluation")
    parser.add_argument("--output", default="quantization_results.csv", help="Output CSV file")
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    
    saved_model_path = load_and_save_model(args.checkpoint, args.name)
    
    print(f"Converting model to quantized formats...")
    run_convert(saved_model_path, args.name, args.quant_levels)
    
    results = []
    quant_levels_to_eval = [q for q in args.quant_levels if (Path(GGUF_DIR) / f"{args.name}-{q}.gguf").exists()]
    
    for quant in tqdm(quant_levels_to_eval, desc="Quantization levels", unit="level"):
        model_name = f"{args.name}-{quant}.gguf"
        model_path = Path(GGUF_DIR) / model_name
        
        tqdm.write(f"\nEvaluating {model_name}...")
        
        try:
            tqa_acc, tqa_max = eval_truthfulqa(model_name, args.tqa_samples)
            tqdm.write(f"  TruthfulQA: acc={tqa_acc:.4f}, max={tqa_max:.4f}")
        except Exception as e:
            tqdm.write(f"  TruthfulQA failed: {e}")
            tqa_acc, tqa_max = None, None
        
        try:
            rouge_l = eval_longbench(model_name, args.longbench_samples)
            tqdm.write(f"  LongBench: ROUGE-L={rouge_l:.4f}")
        except Exception as e:
            tqdm.write(f"  LongBench failed: {e}")
            rouge_l = None
        
        latency = eval_speed(model_name, args.serial)
        tqdm.write(f"  Speed: {latency:.2f} tokens/s" if latency else "  Speed: N/A")
        
        results.append({
            "quantization": quant,
            "tqa_bleurt_acc": tqa_acc if tqa_acc is not None else "",
            "tqa_bleurt_max": tqa_max if tqa_max is not None else "",
            "longbench_rougeL": rouge_l if rouge_l is not None else "",
            "latency_tokens_per_sec": latency if latency else ""
        })
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quantization", "tqa_bleurt_acc", "tqa_bleurt_max", "longbench_rougeL", "latency_tokens_per_sec"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
