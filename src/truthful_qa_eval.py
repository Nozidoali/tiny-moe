#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
import evaluate
import subprocess
import time
import numpy as np
from config import RESULTS_DIR, SCRIPTS_DIR

def run_evaluate(model_name="gpt2-ft-q4_0.gguf", local=True, extra_args=None):
    if extra_args is None:
        extra_args = []

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    n = len(ds)
    print(f"Loaded {n} test samples for Truthful QA")

    # initiate BLEURT evaluator model
    bleurt = evaluate.load('bleurt', 'bleurt-large-128') # load large model for accuracy

    #debug log
    stderr_file = open('debug.log', 'w', encoding='utf-8')
    max_score_arr = []
    acc_score_arr = []
    
    folder = f"{RESULTS_DIR}/outputs" if local else f"{RESULTS_DIR}/outputs_device"
    input_folder = f"{RESULTS_DIR}/inputs" if local else f"{RESULTS_DIR}/inputs_device"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)

    for i, rec in enumerate(ds):
        print(f"-------- sample {i} --------")
        question = rec['question']
        best_answer = rec['best_answer']
        correct_answers = rec['correct_answers']
        incorrect_answers = rec['incorrect_answers']

        question = question.replace("'", " ")
        question = question.replace('"', ' ')
        
        question = f"Question: {question} Answer in English without repeating the question.\nAnswer:"
        
        input_path = os.path.join(input_folder, f"tqa_input_{i}.txt")
        with open(input_path, "w", encoding="utf-8", errors='replace') as fout:
            fout.write(question)

        # Create a copy and add repeat penalty arguments (split correctly)
        args = extra_args.copy()
        args.extend(["--repeat-penalty", "1.5", "--repeat-last-n", "128"])
        
        cli_path = f"{SCRIPTS_DIR}/run-cli-local.sh" if local else f"{SCRIPTS_DIR}/run-cli-device.sh"
        cmd = ["bash", cli_path, "-no-cnv", "-p", f"\"\'{question}\'\"", "-n", str(30)] + args
        
        output_path = os.path.join(folder, f"tqa_output_{i}.txt")
        
        start = time.time()
        env = os.environ.copy()
        env["M"] = model_name
        with open(output_path, "w", encoding="utf-8", errors='replace') as fout:
            # Note: we pass stderr=subprocess.PIPE so we can separately handle it
            proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True, errors='replace', env=env)
        end = time.time()

        latency = end - start
        if proc.returncode != 0:
            # Print stderr to console
            print(f"[ERROR] CLI failed for prompt {question}:")
            print(proc.stderr)
            return -1, -1
        
        # start evaluate
        with open(output_path, "r", encoding='utf-8', errors='replace') as fin:
            pred = fin.read().strip()
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
            
    print('=======================================')
    print('')
    accuracy = sum(acc_score_arr) / n
    print(f'avg max score: {np.mean(np.array(max_score_arr))}')
    print(f'avg accuracy: {accuracy:.3f}')


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TruthfulQA dataset")
    parser.add_argument("--model", default="gpt2-ft-q4_0.gguf", help="Model name (GGUF file name)")
    parser.add_argument("--device", action="store_true", help="Use device CLI script instead of local")
    args = parser.parse_args()
    
    run_evaluate(model_name=args.model, local=not args.device)

if __name__ == "__main__":
    main()
