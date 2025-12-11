#!/usr/bin/env python3
"""
Ablation study script for batch size and thread number.
Runs speed evaluation with different batch sizes and thread counts,
and saves results to CSV.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# Add src directory to path to import speed_eval
sys.path.insert(0, str(Path(__file__).parent / "src"))

from speed_eval import push_model_to_device, run_speed_benchmark, parse_benchmark_output

def run_ablation_study(model_path, device_serial=None, backend="dsp", output_csv="ablation_results.csv", no_push=False):
    """
    Run ablation study with different batch sizes and thread counts.
    
    Args:
        model_path: Path to the model file
        device_serial: Optional device serial number
        backend: Backend to use (cpu, gpu, dsp)
        output_csv: Output CSV file path
        no_push: If True, skip pushing model to device (assume already there)
    """
    batch_sizes = [1, 32, 64, 128, 256]
    thread_counts = [1, 2, 4, 8]
    
    # Get model name
    model_name = Path(model_path).name if no_push else push_model_to_device(model_path, device_serial)
    
    # Prepare CSV output
    results = []
    total_runs = len(batch_sizes) * len(thread_counts)
    current_run = 0
    
    print(f"Starting ablation study for model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Thread counts: {thread_counts}")
    print(f"Total runs: {total_runs}\n")
    
    for batch_size in batch_sizes:
        for thread in thread_counts:
            current_run += 1
            config = f"b{batch_size}_t{thread}"
            
            print(f"[{current_run}/{total_runs}] Running: batch_size={batch_size}, thread={thread}")
            
            try:
                stdout, execution_time = run_speed_benchmark(
                    model_name, 
                    device_serial, 
                    backend, 
                    batch_size, 
                    thread
                )
                
                metrics = parse_benchmark_output(stdout)
                metrics['execution_time'] = execution_time
                
                # Add config, model, batch, thread to the metrics dictionary
                metrics['config'] = config
                metrics['model'] = model_name
                metrics['batch'] = batch_size
                metrics['thread'] = thread
                
                results.append(metrics)
                
                print(f"  ✓ Completed")
                if metrics.get('pp_speed'):
                    print(f"    Prefill: {metrics['pp_speed']:.2f} tokens/s")
                if metrics.get('tg_speed'):
                    print(f"    Decode: {metrics['tg_speed']:.2f} tokens/s")
                if metrics.get('latency'):
                    print(f"    Latency: {metrics['latency']:.4f}s")
                print()
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                # Still record the attempt with error
                error_result = {
                    'config': config,
                    'model': model_name,
                    'batch': batch_size,
                    'thread': thread,
                    'backend': backend,
                    'error': str(e)
                }
                results.append(error_result)
    
    # Write results to CSV
    print(f"\nWriting results to {output_csv}...")
    
    # Collect all unique fieldnames from all results
    all_fieldnames = set()
    for result in results:
        all_fieldnames.update(result.keys())
    
    # Ensure config, model, batch, thread are first columns
    fieldnames = ['config', 'model', 'batch', 'thread']
    fieldnames.extend(sorted([f for f in all_fieldnames if f not in fieldnames]))
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Results saved to {output_csv}")
    print(f"\nSummary:")
    print(f"  Total runs: {total_runs}")
    successful = sum(1 for r in results if 'error' not in r)
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_runs - successful}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Ablation study for batch size and thread number")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--serial", type=str, help="Device serial number")
    parser.add_argument("--backend", type=str, default="dsp", choices=["cpu", "gpu", "dsp"], 
                       help="Backend to use")
    parser.add_argument("--output", type=str, default="ablation_results.csv", 
                       help="Output CSV file path")
    parser.add_argument("--no_push", action="store_true", 
                       help="Skip pushing model to device (assume already there)")
    
    args = parser.parse_args()
    
    run_ablation_study(
        args.model,
        args.serial,
        args.backend,
        args.output,
        args.no_push
    )

if __name__ == "__main__":
    main()
