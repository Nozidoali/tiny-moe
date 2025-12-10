import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

TINYMOE_DIR = os.getenv("TINYMOE_DIR", "/Users/hanyu/Documents/tinymoe")
LLAMA_CPP_DIR = os.getenv("LLAMA_CPP_DIR", "/Users/hanyu/Documents/cs259/external/llama.cpp")

GGUF_DIR = f"{TINYMOE_DIR}/gguf"
MODELS_DIR = f"{TINYMOE_DIR}/models"
SCRIPTS_DIR = f"{TINYMOE_DIR}/scripts"
RESULTS_DIR = f"{TINYMOE_DIR}/results"
PROMPT_FILES_DIR = f"{TINYMOE_DIR}/data/prompt_files"
PROMPT_FILES_TRUNCATED_DIR = f"{RESULTS_DIR}/prompt_files_truncated"

