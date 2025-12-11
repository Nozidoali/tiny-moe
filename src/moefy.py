#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import copy
from datetime import datetime
from pathlib import Path
from utils import load_model_and_tokenizer
from config import MODELS_DIR


class MoELayer(nn.Module):
    def __init__(self, experts, hidden_size, n_experts=2, noise_std=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.weight[0, :] = 10.0
            for i in range(1, n_experts):
                self.gate.weight[i, :] = -10.0
    
    def forward(self, hidden_states, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        gate_logits = self.gate(hidden_flat)
        
        if self.training and self.noise_std > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std
        
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_indices = torch.argmax(gate_probs, dim=-1)
        expert_weights = gate_probs.gather(1, expert_indices.unsqueeze(1)).squeeze(1)
        output_flat = torch.zeros_like(hidden_flat)
        
        for expert_idx in range(self.n_experts):
            expert_mask = (expert_indices == expert_idx)
            if expert_mask.any():
                expert_input = hidden_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input, **kwargs)
                output_flat[expert_mask] = expert_output * expert_weights[expert_mask].unsqueeze(1)
        
        return output_flat.view(batch_size, seq_len, hidden_size)


def get_hidden_size(model):
    config = model.config
    return getattr(config, 'hidden_size', getattr(config, 'n_embd', None))


def check_model_architecture(model):
    config = model.config
    arch = config.architectures[0] if hasattr(config, 'architectures') else None
    model_type = getattr(config, 'model_type', None)
    
    supported = False
    arch_name = None
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        if arch == "LlamaForCausalLM" or model_type == "llama":
            supported = True
            arch_name = "LlamaForCausalLM"
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        if arch == "GPT2LMHeadModel" or model_type == "gpt2":
            supported = True
            arch_name = "GPT2LMHeadModel"
    
    if not supported:
        raise ValueError(
            f"Unsupported model architecture: {arch} (type: {model_type})\n"
            f"Currently supported: LlamaForCausalLM, GPT2LMHeadModel"
        )
    
    return arch_name


def replace_mlp_with_moe(model, n_experts=2, noise_std=0.1):
    hidden_size = get_hidden_size(model)
    
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for layer in model.transformer.h:
            if hasattr(layer, 'mlp'):
                experts = [copy.deepcopy(layer.mlp) for _ in range(n_experts)]
                layer.mlp = MoELayer(experts, hidden_size, n_experts, noise_std)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'mlp'):
                experts = [copy.deepcopy(layer.mlp) for _ in range(n_experts)]
                layer.mlp = MoELayer(experts, hidden_size, n_experts, noise_std)
    
    return model


def save_moe_model(model, tokenizer, output_dir, n_experts, noise_std):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from tinymoe import AutoModelForTinyMoE
    
    config_dict = model.config.to_dict()
    original_arch = config_dict.get('architectures', ['Unknown'])[0]
    original_model_type = config_dict.get('model_type', 'unknown')
    
    config_dict['_original_architecture'] = original_arch
    config_dict['_original_model_type'] = original_model_type
    config_dict['architectures'] = [original_arch.replace('ForCausalLM', 'MoEForCausalLM')]
    config_dict['model_type'] = 'TinyMoE'
    config_dict['n_experts'] = n_experts
    config_dict['moe_top_k'] = 1
    config_dict['moe_noise_std'] = noise_std
    config_dict['moe_layer_type'] = 'mlp'
    
    model._tinymoe_config = config_dict
    
    AutoModelForTinyMoE.save_pretrained(model, output_dir)
    tokenizer.save_pretrained(str(output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True)
    parser.add_argument("--n_experts", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    
    print(f"Loading: {args.input_model}")
    model, tokenizer, device = load_model_and_tokenizer(args.input_model)
    
    print(f"Checking architecture...")
    arch_name = check_model_architecture(model)
    print(f"Detected: {arch_name}")
    
    print(f"Converting to MoE ({args.n_experts} experts)...")
    model = replace_mlp_with_moe(model, n_experts=args.n_experts, noise_std=args.noise_std)
    
    if args.output_dir is None:
        args.output_dir = Path(MODELS_DIR) / f"moe_{datetime.now().strftime('%H%M')}"
    
    print(f"Saving to: {args.output_dir}")
    save_moe_model(model, tokenizer, args.output_dir, args.n_experts, args.noise_std)
    
    if arch_name == "LlamaForCausalLM":
        print(f"\nTensor naming for LLaMA:")
        print(f"  Original: model.layers.{{i}}.mlp.gate_proj.weight")
        print(f"  MoE:      model.layers.{{i}}.mlp.experts.{{j}}.gate_proj.weight")
        print(f"  MoE:      model.layers.{{i}}.mlp.gate.weight (gating)")
    elif arch_name == "GPT2LMHeadModel":
        print(f"\nTensor naming for GPT-2:")
        print(f"  Original: transformer.h.{{i}}.mlp.c_fc.weight")
        print(f"  MoE:      transformer.h.{{i}}.mlp.experts.{{j}}.c_fc.weight")
        print(f"  MoE:      transformer.h.{{i}}.mlp.gate.weight (gating)")
    print(f"\nSample tensor names:")
    for name, _ in list(model.named_parameters())[:8]:
        print(f"  {name}")
    print(f"  ...")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
