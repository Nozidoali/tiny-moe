#!/usr/bin/env python3
import os
import time
import subprocess
from pathlib import Path
from config import PROMPT_FILES_DIR, RESULTS_DIR, SCRIPTS_DIR

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_one(cli_path: str, prompt_device_path: str, output_path: str, extra_args=None, stderr_file=None):
    """
    Run CLI with -f prompt_device_path, capture stdout → file, but stderr → terminal.
    Returns the latency (in seconds).
    """
    if extra_args is None:
        extra_args = []
    
    # Create a new list to avoid mutating the original extra_args
    args = extra_args.copy()
    # Split arguments correctly: each flag and value must be separate list items
    args.extend(["-n", "128", "--repeat-penalty", "1.5", "--repeat-last-n", "128"])
    cmd = [cli_path, "-no-cnv", "-f", prompt_device_path] + args
    # If cli_path is a shell script, wrap with bash
    if cli_path.endswith(".sh"):
        cmd = ["bash"] + cmd
        
    print(f"Running command: {' '.join(cmd)}")

    start = time.time()
    with open(output_path, "w", encoding="utf-8") as fout:
        # Note: we pass stderr=subprocess.PIPE so we can separately handle it
        proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True)
    end = time.time()

    latency = end - start
    if proc.returncode != 0:
        # Print stderr to console
        print(f"[ERROR] CLI failed for prompt {prompt_device_path}:")
        print(proc.stderr)
    return latency

def run_all(local_prompt_dir: str, device_prompt_prefix: str, output_dir: str,
            cli_path: str, extra_args=None):
    ensure_dir(output_dir)
    local = Path(local_prompt_dir)
    prompt_files = sorted(local.glob("*.prompt.txt"))

    latencies = []
    stderr_file = open('debug.log', 'w', encoding='utf-8')
    t0 = time.time()
    for pf in prompt_files:
        fname = pf.name  # e.g. "qmsum_test_0.prompt.txt"
        prompt_dev_path = os.path.join(device_prompt_prefix, fname)
        # derive output filename, strip ".prompt.txt"
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)

        print(f"Running prompt {fname} → output {out_fname}")
        latency = run_one(cli_path, prompt_dev_path, out_path, extra_args, stderr_file)
        print(f"  latency: {latency:.3f} s")
        latencies.append((fname, latency))

    t1 = time.time()
    total = t1 - t0
    return latencies, total

def main():
    local_prompt_dir = PROMPT_FILES_DIR
    device_prompt_prefix = PROMPT_FILES_DIR
    output_dir = f"{RESULTS_DIR}/qmsum_outputs"
    cli_path = f"{SCRIPTS_DIR}/run-cli-local.sh"
    extra_args = []  # e.g. model settings, etc.

    latencies, total_time = run_all(
        local_prompt_dir, device_prompt_prefix, output_dir, cli_path, extra_args
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
