# TinyMoE

## Setup

### 1. Clone the repository and initialize submodules

```bash
git clone git@github.com:Nozidoali/tiny-moe.git
cd tinymoe
git submodule update --init --recursive
```

### 2. Virtual Environment Setup

#### Option A: Using venv (Python built-in)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
conda create -n tinymoe python=3.10
conda activate tinymoe
pip install -r requirements.txt
```

### 3. Build llama.cpp

The llama.cpp submodule is located in `external/llama.cpp` and is checked out to the `hexagon` branch. To build it:

```bash
cd external/llama.cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

The build will create binaries in `external/llama.cpp/build/bin/` including:
- `llama-cli` - Command-line interface
- `llama-quantize` - Model quantization tool

### 4. Environment Variables Setup

```bash
export TINYMOE_DIR="/path/to/tinymoe"
```

## Usage

### Converting Models to Mixture-of-Experts (MoE)

Convert a standard transformer model to MoE architecture by duplicating MLP layers into multiple experts:

```bash
python src/moefy.py --input_model <model_path> --n_experts <num_experts>
```

**Arguments:**
- `--input_model` (required): Path to HuggingFace model (local directory path or HuggingFace model name, e.g., `models/truthfulqa_1234` or `gpt2`)
- `--n_experts`: Number of experts to create (default: 2)
- `--noise_std`: Noise standard deviation for gating network during training (default: 0.1)
- `--output_dir`: Output directory (default: auto-generated with timestamp)

**Example:**

```bash
python src/moefy.py --input_model gpt2 --n_experts 2
python src/moefy.py --input_model models/truthfulqa_1234 --n_experts 4
```

### Model Training

#### Fine-tuning on QMSum (Summarization Task)

Train a model on the QMSum dataset for meeting summarization:

```bash
python src/finetune.py --dataset qmsum --input_model BEE-spoke-data/smol_llama-81M-tied \
  --num_epochs 20 \
  --eval_every_n_epochs 10
```

#### Fine-tuning on TruthfulQA (Question Answering)

Train on TruthfulQA dataset for truthful question answering:

```bash
python src/finetune.py --dataset truthfulqa --input_model models/<qmsum_HH_MM> \
  --num_epochs 20 \
  --eval_every_n_epochs 4
```

#### Fine-tuning Arguments

- `--input_model`: Path to input model (local directory path or HuggingFace model name, e.g., `models/qmsum_1234` or `BEE-spoke-data/smol_llama-81M-tied`)
- `--output_dir`: Output directory for saved models (default: `models/`)
- `--dataset`: Dataset to train on. Choices: `truthfulqa`, `qmsum`, `mixed`
- `--unfreeze_all`: Unfreeze all layers for full finetuning (default: only MLP layers)
- `--max_length`: Maximum sequence length (default: 2028)
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size per device (default: 1)
- `--grad_accum`: Gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for regularization (default: 0.05)
- `--eval_split`: Evaluation split ratio (default: 0.1)
- `--no_split`: Use entire dataset for both training and evaluation
- `--eval_every_n_epochs`: Evaluate every N epochs (can be fractional, e.g., 0.5; default: 100)
- `--l2_weight`: L2 regularization to prevent catastrophic forgetting (default: 0.01)
- `--balance_datasets`: Balance datasets in mixed mode (downsample larger dataset)

### Model Evaluation

#### Evaluating HuggingFace Models

Evaluate models directly from HuggingFace format:

```bash
# Evaluate on TruthfulQA
python src/hf_eval.py --model models/<model_HH_MM> --dataset truthfulqa

# Evaluate on LongBench (QMSum)
python src/hf_eval.py --model models/<model_HH_MM> --dataset longbench
```

**Arguments:**
- `--model` (required): Path to HuggingFace model (local path or HuggingFace model name, e.g., `models/qmsum_1234` or `gpt2`)
- `--dataset`: Dataset to evaluate on. Choices: `truthfulqa`, `longbench`, `qmsum`

#### Evaluating GGUF Models on Device

For on-device evaluation using GGUF quantized models:

**QMSum Evaluation:**

```bash
# Generate predictions
python src/longbench_test.py --model <model_name>.gguf --device

# Compute metrics
python src/longbench_eval.py --device
```

**TruthfulQA Evaluation:**

```bash
python src/truthful_qa_eval.py --model <model_name>.gguf --device
```

**Device Evaluation Arguments:**

`longbench_test.py`:
- `--model`: Model filename (GGUF file name or local path, default: `models-q4_0.gguf`)
- `--device`: Use device CLI script instead of local execution
- `--no-push`: Skip pushing model and prompts to device
- `--cli`: Override CLI script path (local path)
- `--prompt-dir`: Prompt files directory (local path, default: `data/prompt_files/`)
- `--output-dir`: Custom output directory (local path)
- `--serial`: Device serial number for multiple devices

`longbench_eval.py`:
- `--device`: Evaluate device outputs instead of local
- `--output-dir`: Custom output directory to evaluate (local path)

`truthful_qa_eval.py`:
- `--model`: Model filename (GGUF file name or local path, default: `models-q4_0.gguf`)
- `--device`: Use device CLI script instead of local
- `--no-push`: Skip pushing model to device
- `--serial`: Device serial number for multiple devices

### Performance Benchmarking

Measure inference speed and throughput:

```bash
python src/speed_eval.py --model <model_name>.gguf --backend cpu --threads 8 --batch_size 128
```

**Arguments:**
- `--model` (required): Path to GGUF model file (local path, e.g., `gguf/model.gguf`)
- `--serial`: Device serial number
- `--no_push`: Skip pushing model to device (assume already present)
- `--backend`: Backend to use. Choices: `cpu`, `gpu`, `dsp` (default: `dsp`)
- `--save_output`: Save benchmark output to file