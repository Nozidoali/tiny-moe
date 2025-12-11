#!/usr/bin/env python3
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file, save_file


class AutoModelForTinyMoE:
    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if config.get('model_type') != 'TinyMoE':
            raise ValueError(f"Not a TinyMoE model. model_type: {config.get('model_type')}")
        
        from moefy import replace_mlp_with_moe
        
        original_model_type = config.get('_original_model_type', 'llama')
        n_experts = config.get('n_experts', 2)
        noise_std = config.get('moe_noise_std', 0.1)
        
        config['model_type'] = original_model_type
        
        temp_config_path = model_path / "config_temp.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config, f)
        
        model_config = AutoConfig.from_pretrained(model_path / "config_temp.json")
        temp_config_path.unlink()
        
        model = AutoModelForCausalLM.from_config(model_config)
        model = replace_mlp_with_moe(model, n_experts=n_experts, noise_std=noise_std)
        
        state_dict = load_file(model_path / "model.safetensors")
        model.load_state_dict(state_dict, strict=True)
        
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        
        config['model_type'] = 'TinyMoE'
        model._tinymoe_config = config
        model._is_tinymoe = True
        
        return model
    
    @classmethod
    def from_model(cls, base_model, n_experts=2, noise_std=0.1):
        from moefy import replace_mlp_with_moe, check_model_architecture
        
        check_model_architecture(base_model)
        model = replace_mlp_with_moe(base_model, n_experts=n_experts, noise_std=noise_std)
        
        config = base_model.config.to_dict()
        config['model_type'] = 'TinyMoE'
        config['_original_model_type'] = base_model.config.model_type
        config['_original_architecture'] = config.get('architectures', ['Unknown'])[0]
        config['n_experts'] = n_experts
        config['moe_noise_std'] = noise_std
        
        model._tinymoe_config = config
        model._is_tinymoe = True
        
        return model
    
    @staticmethod
    def save_pretrained(model, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        state_dict = model.state_dict()
        
        if hasattr(model.config, 'tie_word_embeddings') and model.config.tie_word_embeddings:
            if 'lm_head.weight' in state_dict and 'model.embed_tokens.weight' in state_dict:
                if state_dict['lm_head.weight'].data_ptr() == state_dict['model.embed_tokens.weight'].data_ptr():
                    state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight'].clone()
        
        # Fix gate weights to ensure they're contiguous and properly formatted for GGUF conversion
        # This prevents corruption during lazy tensor handling in convert_hf_to_gguf.py
        for key in list(state_dict.keys()):
            if key.endswith('.mlp.gate.weight'):
                # Ensure gate weights are contiguous in memory
                state_dict[key] = state_dict[key].contiguous().clone()
                print(f"Fixed gate weight: {key}, shape={state_dict[key].shape}")
        
        save_file(state_dict, output_dir / "model.safetensors")
        
        if hasattr(model, '_tinymoe_config'):
            config = model._tinymoe_config
        else:
            config = model.config.to_dict()
            config['model_type'] = 'TinyMoE'
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
