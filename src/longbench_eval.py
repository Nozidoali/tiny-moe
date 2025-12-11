#!/usr/bin/env python3

import os
import re
import argparse
from datasets import load_dataset
import evaluate
from config import RESULTS_DIR

def load_references():
    """
    Load the qmsum test split and return a dict mapping sample index -> reference summary.
    You must adapt this to the correct field name in the dataset.
    """
    ds = load_dataset("zai-org/LongBench", "qmsum", split="test")
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()
    return ref_map

def evaluate_folder_outputs(output_dir: str, ref_map: dict):
    """
    Traverse output files (qmsum_test_{id}.txt) in output_dir,
    collect predictions and references, compute rouge via evaluate.
    Return results list and overall metrics.
    """
    rouge = evaluate.load("rouge")
    pattern = re.compile(r"qmsum_test_(\d+)\.txt")
    predictions = []
    references = []
    sample_ids = []

    for fname in sorted(os.listdir(output_dir)):
        m = pattern.match(fname)
        if not m:
            print(f"Skipping unrecognized file name: {fname}")
            continue
        idx = int(m.group(1))
        outpath = os.path.join(output_dir, fname)
        with open(outpath, "r", encoding="utf-8", errors="replace") as f:
            pred = f.read().strip()

        if idx not in ref_map:
            print(f"Warning: no reference for sample {idx}, skipping")
            continue

        ref = ref_map[idx]
        sample_ids.append(idx)
        predictions.append(pred)
        references.append(ref)

    # Compute ROUGE (aggregated)
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    # result is a dict: e.g. { 'rouge1': ..., 'rouge2': ..., 'rougeL': ..., 'rougeLsum': ... } :contentReference[oaicite:0]{index=0}

    # Also compute per-sample if desired by setting `use_aggregator=False`
    per_sample = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)

    # per_sample entries are lists of scores per sample e.g. per_sample['rougeL'] is list of f1 for each sample :contentReference[oaicite:1]{index=1}

    results = []
    for i, idx in enumerate(sample_ids):
        rl = per_sample["rougeL"][i]
        results.append((idx, rl))

    return results, result

def main():
    parser = argparse.ArgumentParser(description="Evaluate QMSum model outputs")
    parser.add_argument("--device", action="store_true", help="Evaluate device outputs instead of local")
    parser.add_argument("--output-dir", type=str, help="Custom output directory to evaluate")
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{RESULTS_DIR}/qmsum_outputs_device" if args.device else f"{RESULTS_DIR}/qmsum_outputs"
    
    print("Loading references...")
    ref_map = load_references()
    print(f"Loaded {len(ref_map)} references.")

    print(f"Evaluating predictions in {output_dir}")
    per_sample_scores, aggregated = evaluate_folder_outputs(output_dir, ref_map)

    print("\n=== ROUGE-L per sample ===")
    for idx, rl in per_sample_scores:
        print(f"Sample {idx}: ROUGE-L = {rl:.4f}")

    print("\n=== Aggregated ROUGE ===")
    print(f"ROUGE-L (F1): {aggregated['rougeL']:.4f}")
    print(f"ROUGE-1: {aggregated['rouge1']:.4f}, ROUGE-2: {aggregated['rouge2']:.4f}, ROUGE-Lsum: {aggregated['rougeLsum']:.4f}")

if __name__ == "__main__":
    main()
