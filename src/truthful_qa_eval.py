#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from datasets import load_dataset
import evaluate
import subprocess
import time
import numpy as np
from config import RESULTS_DIR, SCRIPTS_DIR, DEVICE_LLAMA_CPP_DIR, GGUF_DIR
from utils import get_cli_args, format_truthfulqa_question, format_mmlu_question

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

def run_evaluate(model_name="models-q4_0.gguf", local=True, extra_args=None, device_serial=None, dataset="truthfulqa"):
    if extra_args is None:
        extra_args = []

    if dataset == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        n = len(ds)
        print(f"Loaded {n} test samples for TruthfulQA")
        bleurt = evaluate.load('bleurt', 'bleurt-large-128')
    elif dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        n = len(ds)
        print(f"Loaded {n} test samples for MMLU")
        bleurt = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    stderr_file = open('debug.log', 'w', encoding='utf-8')
    max_score_arr = []
    acc_score_arr = []

    suffix = f"_{dataset}"
    folder = f"{RESULTS_DIR}/outputs{suffix}" if local else f"{RESULTS_DIR}/outputs_device{suffix}"
    input_folder = f"{RESULTS_DIR}/inputs{suffix}" if local else f"{RESULTS_DIR}/inputs_device{suffix}"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)

    for i, rec in enumerate(ds):
        print(f"-------- sample {i} --------")

        if dataset == "truthfulqa":
            question_text = rec['question']
            best_answer = rec['best_answer']
            correct_answers = rec['correct_answers']
            incorrect_answers = rec['incorrect_answers']
            formatted_question = format_truthfulqa_question(question_text)
            file_prefix = "tqa"
            max_tokens = 30
        elif dataset == "mmlu":
            question_text = rec['question']
            choices = rec['choices']
            correct_answer_idx = rec['answer']
            formatted_question = format_mmlu_question(question_text, choices)
            file_prefix = "mmlu"
            max_tokens = 5

        input_path = os.path.join(input_folder, f"{file_prefix}_input_{i}.txt")
        with open(input_path, "w", encoding="utf-8", errors='replace') as fout:
            fout.write(formatted_question)

        args = extra_args.copy()
        args.extend(get_cli_args(dataset_name=dataset))

        cli_path = f"{SCRIPTS_DIR}/run-cli-local.sh" if local else f"{SCRIPTS_DIR}/run-cli-device.sh"
        cmd = ["bash", cli_path, "-no-cnv", "-p", f"\"\'{formatted_question}\'\"", "-n", str(max_tokens)] + args

        print(f"Running command: {' '.join(cmd)}")

        output_path = os.path.join(folder, f"{file_prefix}_output_{i}.txt")
        
        start = time.time()
        env = os.environ.copy()
        env["M"] = model_name
        if not local:
            env["LLAMA_CPP_DIR"] = DEVICE_LLAMA_CPP_DIR
        with open(output_path, "w", encoding="utf-8", errors='replace') as fout:
            # Note: we pass stderr=subprocess.PIPE so we can separately handle it
            proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True, errors='replace', env=env)
        end = time.time()

        latency = end - start
        if proc.returncode != 0:
            print(f"[ERROR] CLI failed for prompt {formatted_question}:")
            print(proc.stderr)
            return -1, -1

        with open(output_path, "r", encoding='utf-8', errors='replace') as fin:
            pred = fin.read().strip()

            if dataset == "truthfulqa":
                predictions = [pred] * len(correct_answers)
                score_true = bleurt.compute(predictions=predictions, references=correct_answers)['scores']
                predictions = [pred] * len(incorrect_answers)
                score_false = bleurt.compute(predictions=predictions, references=incorrect_answers)['scores']

                max_score = max(score_true)
                acc_score = int(max(score_true) > max(score_false))

                print(f'    latency: {latency:.3f} s.')
                print(f'    max_score: {max_score:.3f}')
                print(f'    acc: {acc_score}')

                max_score_arr.append(max_score)
                acc_score_arr.append(acc_score)
                print(f'    avg accuracy: {np.mean(acc_score_arr)}')
                print(f'    avg max score: {np.mean(max_score_arr)}')

            elif dataset == "mmlu":
                pred_clean = pred.strip().upper()
                pred_letter = None
                for char in pred_clean:
                    if char in 'ABCD':
                        pred_letter = char
                        break

                correct_letter = chr(65 + correct_answer_idx)
                acc_score = int(pred_letter == correct_letter)

                print(f'    latency: {latency:.3f} s.')
                print(f'    predicted: {pred_letter}, correct: {correct_letter}')
                print(f'    acc: {acc_score}')

                acc_score_arr.append(acc_score)
                print(f'    avg accuracy: {np.mean(acc_score_arr)}')
            
    print('=======================================')
    print('')
    accuracy = sum(acc_score_arr) / n
    if dataset == "truthfulqa":
        print(f'avg max score: {np.mean(np.array(max_score_arr))}')
    print(f'avg accuracy: {accuracy:.3f}')


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TruthfulQA or MMLU dataset")
    parser.add_argument("--model", default="models-q4_0.gguf", help="Model name (GGUF file name)")
    parser.add_argument("--device", action="store_true", help="Use device CLI script instead of local")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing model (assume already on device)")
    parser.add_argument("--serial", type=str, help="Device serial number (for multiple devices)")
    parser.add_argument("--dataset", default="truthfulqa", choices=["truthfulqa", "mmlu"], help="Dataset to evaluate on (default: truthfulqa)")
    args = parser.parse_args()

    local = not args.device

    if local:
        model_name = args.model
    else:
        if not args.no_push:
            model_path = f"{GGUF_DIR}/{args.model}" if not os.path.isabs(args.model) else args.model
            model_name = push_model_to_device(model_path, args.serial)
        else:
            model_name = args.model
            print(f"Skipping push, using model: {model_name}")

    run_evaluate(model_name=model_name, local=local, device_serial=args.serial, dataset=args.dataset)

if __name__ == "__main__":
    main()
