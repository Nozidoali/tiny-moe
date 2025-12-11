# TinyMoE

## Environment Variables Setup

```bash
export TINYMOE_DIR="/path/to/tinymoe"
export LLAMA_CPP_DIR="/path/to/llama.cpp"
```

## Convert Model to MoE

```bash
python src/moefy.py --input_model gpt2 --n_experts 2
python src/moefy.py --input_model models/truthfulqa_1234 --n_experts 4
```

## Workflow

First finetune the QMSum model.
```
python src/finetune.py --dataset qmsum --input_model BEE-spoke-data/smol_llama-81M-tied \
  --num_epochs 20 \
  --eval_every_n_epochs 10
```

Then finetune the TruthfulQA model.
```bash
python src/finetune.py --dataset truthfulqa --input_model models/<qmsum_HH_MM> \
  --num_epochs 20 \
  --eval_every_n_epochs 4
```

Evaluate HuggingFace models directly.
```bash
# Evaluate on TruthfulQA
python src/eval_hf.py --model models/<model_HH_MM> --dataset truthfulqa

# Evaluate on LongBench (QMSum)
python src/eval_hf.py --model models/<model_HH_MM> --dataset longbench
```

Or evaluate using GGUF models on device.
```bash
python src/longbench_test.py --model <qmsum_HH_MM>.gguf --device
python src/longbench_eval.py --device
python src/truthful_qa_eval.py --model <qmsum_HH_MM>.gguf --device
```

Speed evaluation.
```bash
python src/speed_eval.py --model <qmsum_HH_MM>.gguf --backend cpu --threads 8 --batch_size 128
```