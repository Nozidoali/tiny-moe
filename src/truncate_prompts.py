#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from config import PROMPT_FILES_DIR, RESULTS_DIR

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None

def truncate_prompt(prompt: str, context_limit: int) -> str:
    if _ENCODING is None:
        return prompt
    encoded = _ENCODING.encode(prompt)
    if len(encoded) <= context_limit:
        return prompt
    head_tokens = int(context_limit * 0.6)
    tail_tokens = context_limit - head_tokens
    head = _ENCODING.decode(encoded[:head_tokens])
    tail = _ENCODING.decode(encoded[-tail_tokens:])
    return head + "\n...\n" + tail

def truncate_prompts(input_dir: str, output_dir: str, context_limit: int = 8192):
    os.makedirs(output_dir, exist_ok=True)
    input_path, output_path = Path(input_dir), Path(output_dir)
    prompt_files = sorted(input_path.glob("*.prompt.txt"))
    
    for pf in prompt_files:
        with open(pf, "r", encoding="utf-8") as f:
            truncated = truncate_prompt(f.read(), context_limit)
        with open(output_path / pf.name, "w", encoding="utf-8") as f:
            f.write(truncated)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=PROMPT_FILES_DIR)
    parser.add_argument("--output-dir", default=f"{RESULTS_DIR}/prompt_files_truncated")
    parser.add_argument("--context-limit", type=int, default=768)
    args = parser.parse_args()
    truncate_prompts(args.input_dir, args.output_dir, args.context_limit)

if __name__ == "__main__":
    main()
