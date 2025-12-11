#!/usr/bin/env python3
import os
import time
import subprocess
import argparse
from pathlib import Path
from config import PROMPT_FILES_DIR, PROMPT_FILES_TRUNCATED_DIR, RESULTS_DIR, SCRIPTS_DIR, GGUF_DIR, DEVICE_LLAMA_CPP_DIR
from utils import get_cli_args

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def push_model_to_device(model_path, device_serial=None):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_name = model_file.name
    device_path = f"/data/local/tmp/gguf/{model_name}"
    
    adb_cmd = ["adb"]
    if device_serial:
        adb_cmd.extend(["-s", device_serial])
    adb_cmd.extend(["push", str(model_file), device_path])
    
    print(f"Pushing model {model_name} to device...")
    start = time.time()
    result = subprocess.run(adb_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Model push failed: {result.stderr}")
    print(f"✓ Model pushed in {time.time() - start:.2f}s\n")
    
    return model_name

def push_prompts_to_device(local_prompt_dir, device_serial=None):
    device_base = "/data/local/tmp"
    local_path = Path(local_prompt_dir)
    device_prompt_dir = f"{device_base}/{local_path.name}"
    
    adb_cmd = ["adb"]
    if device_serial:
        adb_cmd.extend(["-s", device_serial])
    
    print(f"Pushing prompts to device...")
    adb_cmd.extend(["push", str(local_prompt_dir), device_base])
    
    result = subprocess.run(adb_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to push prompts: {result.stderr}")
    
    print(f"✓ Prompts pushed to {device_prompt_dir}\n")
    return device_prompt_dir

def run_one(cli_path: str, prompt_device_path: str, output_path: str, extra_args=None, stderr_file=None, model=None, local=True):
    if extra_args is None:
        extra_args = []
    
    args = extra_args.copy()
    args.extend(get_cli_args(dataset_name="qmsum"))
    cmd = [cli_path, "-no-cnv", "-f", prompt_device_path] + args
    
    env = os.environ.copy()
    if model:
        env["M"] = model
    if not local:
        env["LLAMA_CPP_DIR"] = DEVICE_LLAMA_CPP_DIR
        print(f"[Device mode] Setting LLAMA_CPP_DIR={DEVICE_LLAMA_CPP_DIR}")
    
    if cli_path.endswith(".sh"):
        cmd = ["bash"] + cmd
        
    print(f"Running command: {' '.join(cmd)}")

    start = time.time()
    with open(output_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True, env=env)
    end = time.time()

    latency = end - start
    if proc.returncode != 0:
        print(f"[ERROR] CLI failed for prompt {prompt_device_path}:")
        print(proc.stderr)
    return latency

def run_all(local_prompt_dir: str, device_prompt_prefix: str, output_dir: str,
            cli_path: str, extra_args=None, model=None, local=True):
    ensure_dir(output_dir)
    local_path = Path(local_prompt_dir)
    prompt_files = sorted(local_path.glob("*.prompt.txt"))

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
        latency = run_one(cli_path, prompt_dev_path, out_path, extra_args, stderr_file, model, local)
        print(f"  latency: {latency:.3f} s")
        latencies.append((fname, latency))

    t1 = time.time()
    total = t1 - t0
    return latencies, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="models-q4_0.gguf", help="Model filename in GGUF directory")
    parser.add_argument("--device", action="store_true", help="Use device CLI script instead of local")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing model and prompts (assume already on device)")
    parser.add_argument("--cli", type=str, help="Override CLI script path")
    parser.add_argument("--prompt-dir", type=str, default=PROMPT_FILES_TRUNCATED_DIR, help="Prompt files directory")
    parser.add_argument("--output-dir", type=str, help="Output directory (auto-set based on --device if not specified)")
    parser.add_argument("--serial", type=str, help="Device serial number (for multiple devices)")
    args = parser.parse_args()
    
    local = not args.device
    local_prompt_dir = args.prompt_dir
    
    if local:
        device_prompt_prefix = args.prompt_dir
        model_name = args.model
    else:
        if not args.no_push:
            model_path = f"{GGUF_DIR}/{args.model}" if not os.path.isabs(args.model) else args.model
            model_name = push_model_to_device(model_path, args.serial)
            device_prompt_prefix = push_prompts_to_device(local_prompt_dir, args.serial)
        else:
            model_name = args.model
            device_prompt_prefix = f"/data/local/tmp/{Path(local_prompt_dir).name}"
            print(f"Skipping push, using model: {model_name}")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{RESULTS_DIR}/qmsum_outputs" if local else f"{RESULTS_DIR}/qmsum_outputs_device"
    
    if args.cli:
        cli_path = args.cli
    else:
        cli_path = f"{SCRIPTS_DIR}/run-cli-local.sh" if local else f"{SCRIPTS_DIR}/run-cli-device.sh"
    
    extra_args = []

    latencies, total_time = run_all(
        local_prompt_dir, device_prompt_prefix, output_dir, cli_path, extra_args, model_name, local
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
