#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import re
from pathlib import Path

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
    
    print(f"Pushing {model_name} to device...")
    start = time.time()
    result = subprocess.run(adb_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"adb push failed: {result.stderr}")
    print(f"✓ Pushed in {time.time() - start:.2f}s\n")
    
    return model_name

def run_speed_benchmark(model_name, device_serial=None, backend="dsp"):
    basedir = "/data/local/tmp/llama.cpp"
    
    adb_cmd = ["adb"]
    if device_serial:
        adb_cmd.extend(["-s", device_serial])
    
    shell_cmd = f"cd {basedir} && MODEL={model_name} ./run.sh {backend}"
    adb_cmd.extend(["shell", shell_cmd])
    
    result = subprocess.run(adb_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")
    
    return result.stdout

def parse_benchmark_output(output):
    metrics = {'backend': None, 'pp_tokens': None, 'pp_speed': None, 'tg_tokens': None, 'tg_speed': None}
    
    for line in output.split('\n'):
        if '|' not in line or '-' in line[:5]:
            continue
        
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 13 or 'model' in parts[0].lower():
            continue
        
        if metrics['backend'] is None:
            metrics['backend'] = parts[3]
        
        test_type = parts[11].strip()
        speed_str = parts[12].split('±')[0].strip()
        
        try:
            speed = float(speed_str)
            if 'pp' in test_type.lower():
                metrics['pp_speed'] = speed
                match = re.search(r'pp(\d+)', test_type.lower())
                if match:
                    metrics['pp_tokens'] = int(match.group(1))
            elif 'tg' in test_type.lower():
                metrics['tg_speed'] = speed
                match = re.search(r'tg(\d+)', test_type.lower())
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
    
    model_name = Path(args.model).name if args.no_push else push_model_to_device(args.model, args.serial)
    
    stdout = run_speed_benchmark(model_name, args.serial, args.backend)
    
    if args.save_output:
        with open(f"speed_eval_{model_name}_{int(time.time())}.txt", 'w') as f:
            f.write(stdout)
    
    metrics = parse_benchmark_output(stdout)
    print_results(model_name, metrics)

if __name__ == "__main__":
    main()
