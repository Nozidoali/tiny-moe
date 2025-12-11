# TinyMoE

## Project requirements

Push the bin, lib to the device.
```bash
adb push ./tinymoe.llama.cpp /data/local/tmp
```

The huggingface model: see `models/` folder.
- the expert model: `mixed_0611`
- the moe model: `moe_0611`

The GGUF model: see `gguf/` folder.
- the expert gguf model: `mixed_0611-q4_0.gguf`
- the moe gguf model: `moe_0611-q4_0.gguf`

## Setup

### 1. Clone the repository and initialize submodules

```bash
git clone git@github.com:Nozidoali/tiny-moe.git
cd tinymoe
git submodule update --init --recursive
```

Prepare the truncated prompt files.
```bash
python src/truncate_prompts.py
```

### 2. Virtual Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
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

### Model Training

Finetuning model
```bash
python src/finetune.py --dataset mixed --input_model BEE-spoke-data/smol_llama-81M-tied \
  --num_epochs 20 --balance_datasets --unfreeze_all \
  --eval_every_n_epochs 5
```

Convert the expert to a MoE model
```bash
python src/moefy.py --input_model models/mixed_HHMM --n_experts 4
```


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


**Speed Evaluation:**

```bash
python src/speed_eval.py --model <model_name>.gguf --backend cpu --threads 8 --batch_size 128
```

### Evaluating HuggingFace Models

Evaluate models directly from HuggingFace format:

```bash
# Evaluate on TruthfulQA
python src/hf_eval.py --model models/<model_HH_MM> --dataset truthfulqa

# Evaluate on LongBench (QMSum)
python src/hf_eval.py --model models/<model_HH_MM> --dataset longbench
```
