import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaAttention_Sparse_Fixed(LlamaAttention):
    """Fixed pattern sparse attention"""
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.window_size = getattr(config, 'window_size', 256)
        
    def create_sparse_mask(self, seq_length):
        # Create fixed sparse attention pattern
        mask = torch.zeros(seq_length, seq_length, device=self.q_proj.weight.device)
        
        # Local window attention
        for i in range(seq_length):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_length, i + self.window_size // 2)
            mask[i, window_start:window_end] = 1
            
        return mask.bool()

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        attn_weights = outputs[1]
        
        if attn_weights is not None:
            # Apply sparse mask
            sparse_mask = self.create_sparse_mask(attn_weights.size(-1))
            attn_weights = attn_weights.masked_fill(~sparse_mask, float('-inf'))
            
        return (outputs[0], attn_weights, outputs[2])

class LlamaAttention_Sparse_Strided(LlamaAttention):
    """Strided pattern sparse attention"""
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.window_size = getattr(config, 'window_size', 256)
        self.stride = getattr(config, 'stride', 128)
        
    def create_sparse_mask(self, seq_length):
        # Create strided sparse attention pattern
        mask = torch.zeros(seq_length, seq_length, device=self.q_proj.weight.device)
        
        # Local window attention
        for i in range(seq_length):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_length, i + self.window_size // 2)
            mask[i, window_start:window_end] = 1
            
        # Strided attention
        for i in range(seq_length):
            for j in range(0, seq_length, self.stride):
                mask[i, j] = 1
                
        return mask.bool()

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        attn_weights = outputs[1]
        
        if attn_weights is not None:
            # Apply sparse mask
            sparse_mask = self.create_sparse_mask(attn_weights.size(-1))
            attn_weights = attn_weights.masked_fill(~sparse_mask, float('-inf'))
            
        return (outputs[0], attn_weights, outputs[2])

def convert_attention_type(model, attention_type: str, config: LlamaConfig):
    """Convert model's attention layers to specified type"""
    attention_classes = {
        "h2o": LlamaAttention_heavy_hitter,
        "sparse_fixed": LlamaAttention_Sparse_Fixed,
        "sparse_strided": LlamaAttention_Sparse_Strided
    }
    
    if attention_type not in attention_classes:
        raise ValueError(f"Unsupported attention type: {attention_type}")
        
    AttentionClass = attention_classes[attention_type]
    
    def convert_module(module):
        for name, child in module.named_children():
            if isinstance(child, LlamaAttention):
                setattr(module, name, AttentionClass(config))
            elif len(list(child.children())) > 0:
                convert_module(child)
                
    convert_module(model)
    return model