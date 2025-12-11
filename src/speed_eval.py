#!/usr/bin/env python3
import argparse
import subprocess
import time
import re
from pathlib import Path
from config import SCRIPTS_DIR

def adb_cmd(device_serial=None):
    cmd = ["adb"]
    if device_serial:
        cmd.extend(["-s", device_serial])
    return cmd

def push_run_script_to_device(device_serial=None):
    run_script_path = Path(SCRIPTS_DIR) / "run.sh"
    if not run_script_path.exists():
        raise FileNotFoundError(f"run.sh not found at {run_script_path}")
    
    device_dir = "/data/local/tmp/llama.cpp"
    device_path = f"{device_dir}/run.sh"
    
    subprocess.run(adb_cmd(device_serial) + ["shell", f"mkdir -p {device_dir}"], capture_output=True)
    
    print(f"Pushing run.sh to device...")
    start = time.time()
    result = subprocess.run(adb_cmd(device_serial) + ["push", str(run_script_path), device_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to push run.sh: {result.stderr}")
    
    subprocess.run(adb_cmd(device_serial) + ["shell", f"chmod +x {device_path}"], capture_output=True)
    print(f"✓ run.sh pushed in {time.time() - start:.2f}s\n")

def push_model_to_device(model_path, device_serial=None):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_name = model_file.name
    device_path = f"/data/local/tmp/gguf/{model_name}"
    
    print(f"Pushing {model_name} to device...")
    start = time.time()
    result = subprocess.run(adb_cmd(device_serial) + ["push", str(model_file), device_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"adb push failed: {result.stderr}")
    print(f"✓ Pushed in {time.time() - start:.2f}s\n")
    return model_name

def run_speed_benchmark(model_name, device_serial=None, backend="dsp"):
    basedir = "/data/local/tmp/llama.cpp"
    shell_cmd = f"cd {basedir} && MODEL={model_name} ./run.sh {backend}"
    result = subprocess.run(adb_cmd(device_serial) + ["shell", shell_cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")
    return result.stdout

def parse_benchmark_output(output):
    metrics = {'backend': None, 'pp_tokens': None, 'pp_speed': None, 'tg_tokens': None, 'tg_speed': None}
    
    for line in output.split('\n'):
        if '|' not in line or '-' in line[:5] or len(line.split('|')) < 13:
            continue
        
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if 'model' in parts[0].lower():
            continue
        
        if metrics['backend'] is None:
            metrics['backend'] = parts[3]
        
        test_type = parts[11].strip().lower()
        speed_str = parts[12].split('±')[0].strip()
        
        try:
            speed = float(speed_str)
            if 'pp' in test_type:
                metrics['pp_speed'] = speed
                match = re.search(r'pp(\d+)', test_type)
                if match:
                    metrics['pp_tokens'] = int(match.group(1))
            elif 'tg' in test_type:
                metrics['tg_speed'] = speed
                match = re.search(r'tg(\d+)', test_type)
                if match:
                    metrics['tg_tokens'] = int(match.group(1))
        except:
            pass
    
    return metrics

def print_results(model_name, metrics):
    print(f"{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"Backend: {metrics['backend']}")
    if metrics['pp_speed']:
        tokens = f" ({metrics['pp_tokens']} tokens)" if metrics['pp_tokens'] else ""
        print(f"Prefill:  {metrics['pp_speed']:.2f} tokens/s{tokens}")
    if metrics['tg_speed']:
        tokens = f" ({metrics['tg_tokens']} tokens)" if metrics['tg_tokens'] else ""
        print(f"Decode:   {metrics['tg_speed']:.2f} tokens/s{tokens}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Speed evaluation on Android device")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--serial", type=str)
    parser.add_argument("--no_push", action="store_true")
    parser.add_argument("--backend", type=str, default="dsp", choices=["cpu", "gpu", "dsp"])
    parser.add_argument("--save_output", action="store_true")
    args = parser.parse_args()
    
    if not args.no_push:
        push_run_script_to_device(args.serial)
    
    model_name = Path(args.model).name if args.no_push else push_model_to_device(args.model, args.serial)
    stdout = run_speed_benchmark(model_name, args.serial, args.backend)
    
    if args.save_output:
        with open(f"speed_eval_{model_name}_{int(time.time())}.txt", 'w') as f:
            f.write(stdout)
    
    metrics = parse_benchmark_output(stdout)
    print_results(model_name, metrics)

if __name__ == "__main__":
    main()
