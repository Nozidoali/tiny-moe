# TinyMoE

## Environment Variables Setup

```bash
export TINYMOE_DIR="/path/to/tinymoe"
export LLAMA_CPP_DIR="/path/to/llama.cpp"
```

## Workflow

First finetune the QMSum model.
```bash
python src/finetune.py --dataset qmsum --input_model BEE-spoke-data/smol_llama-81M-tied \
  --num_epochs 200 \
  --eval_every_n_epochs 10
```

Then finetune the TruthfulQA model.
```bash
python src/finetune.py --dataset truthfulqa --input_model models/<qmsum_HH_MM> \
  --num_epochs 20 \
  --eval_every_n_epochs 4
```

Then evaluate the models on the LongBench dataset.
```bash
python src/longbench_test.py --model models/<qmsum_HH_MM> --device
python src/longbench_eval.py --device
```

Then evaluate the models on the TruthfulQA dataset.
```bash
python src/truthful_qa_eval.py --model models/<qmsum_HH_MM> --device
```

Speed evaluation.
```bash
python src/speed_eval.py --model models/<qmsum_HH_MM> --backend cpu --threads 8 --batch_size 128
```