#!/usr/bin/env python3
import os
import time
import subprocess
import argparse
from pathlib import Path
from config import PROMPT_FILES_DIR, PROMPT_FILES_TRUNCATED_DIR, RESULTS_DIR, SCRIPTS_DIR, GGUF_DIR
from utils import get_cli_args

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_one(cli_path: str, prompt_device_path: str, output_path: str, extra_args=None, stderr_file=None, model=None):
    if extra_args is None:
        extra_args = []
    
    args = extra_args.copy()
    args.extend(get_cli_args(dataset_name="qmsum"))
    cmd = [cli_path, "-no-cnv", "-f", prompt_device_path] + args
    
    env = os.environ.copy()
    if model:
        env["M"] = model
    
    if cli_path.endswith(".sh"):
        cmd = ["bash"] + cmd
        
    print(f"Running command: {' '.join(cmd)}")

    start = time.time()
    with open(output_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True, env=env)
    end = time.time()

    latency = end - start
    if proc.returncode != 0:
        # Print stderr to console
        print(f"[ERROR] CLI failed for prompt {prompt_device_path}:")
        print(proc.stderr)
    return latency

def run_all(local_prompt_dir: str, device_prompt_prefix: str, output_dir: str,
            cli_path: str, extra_args=None, model=None):
    ensure_dir(output_dir)
    local = Path(local_prompt_dir)
    prompt_files = sorted(local.glob("*.prompt.txt"))

    latencies = []
    stderr_file = open('debug.log', 'w', encoding='utf-8')
    t0 = time.time()
    for pf in prompt_files:
        fname = pf.name
        prompt_dev_path = os.path.join(device_prompt_prefix, fname)
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)

        print(f"Running prompt {fname} → output {out_fname}")
        latency = run_one(cli_path, prompt_dev_path, out_path, extra_args, stderr_file, model)
        print(f"  latency: {latency:.3f} s")
        latencies.append((fname, latency))

    t1 = time.time()
    total = t1 - t0
    return latencies, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="models-q4_0.gguf", help="Model filename in GGUF directory (e.g., models-q4_0.gguf)")
    parser.add_argument("--cli", type=str, help="Override CLI script path")
    parser.add_argument("--prompt-dir", type=str, default=PROMPT_FILES_TRUNCATED_DIR, help="Prompt files directory")
    parser.add_argument("--output-dir", type=str, default=f"{RESULTS_DIR}/qmsum_outputs", help="Output directory")
    args = parser.parse_args()
    
    local_prompt_dir = args.prompt_dir
    device_prompt_prefix = args.prompt_dir
    output_dir = args.output_dir
    cli_path = args.cli if args.cli else f"{SCRIPTS_DIR}/run-cli-local.sh"
    extra_args = []
    model = args.model

    latencies, total_time = run_all(
        local_prompt_dir, device_prompt_prefix, output_dir, cli_path, extra_args, model
    )

    print("\n=== Benchmark Summary ===")
    for fname, lat in latencies:
        print(f"{fname}: {lat:.3f} s")
    print(f"Total time for {len(latencies)} samples: {total_time:.3f} s")
    if latencies:
        avg = sum(lat for _, lat in latencies) / len(latencies)
        print(f"Average latency: {avg:.3f} s")

if __name__ == "__main__":
    main()
