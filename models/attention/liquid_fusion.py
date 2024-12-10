import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from models.attention.cache import CombinedCache
import copy

class LiquidFusion(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        if layer_idx is None:
            raise ValueError("layer_idx must be provided for LiquidFusion")
            
        # Create a copy of the config to avoid modifying the original
        self.config = copy.deepcopy(config)
        self.layer_idx = layer_idx
        
        # Get model dimensions from config
        self.hidden_size = getattr(config, 'hidden_size', None)
        if self.hidden_size is None:
            raise ValueError("Config must contain hidden_size")
            
        self.num_heads = getattr(config, 'num_attention_heads', None)
        if self.num_heads is None:
            raise ValueError("Config must contain num_attention_heads")
            
        self.head_dim = self.hidden_size // self.num_heads
        
        # Handle sequence length settings
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
        self.sequence_length = getattr(config, 'sequence_length', 
                                     min(256, self.max_position_embeddings))
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.sequence_length, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.sequence_length, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Initialize rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, 'rope_base', 10000),
            device=getattr(config, 'device', None)
        )

        # Add StreamingLLM components
        self.start_size = config.start_size
        self.recent_size = config.recent_size
        
        # Add H2O components
        self.heavy_ratio = config.heavy_ratio
        self.recent_ratio = config.recent_ratio
        
        # Initialize combined cache
        self.kv_cache = CombinedCache(
            start_size=self.start_size,
            recent_size=self.recent_size,
            heavy_ratio=self.heavy_ratio,
            recent_ratio=self.recent_ratio
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project and reshape states
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Cache keys before RoPE (StreamingLLM approach)
        pre_rope_keys = key_states.clone()
        
        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=q_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # Combine past cache if exists
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Use fp32 for better stability (from H2O)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Update cache using both methods
        if use_cache:
            past_key_value = self.kv_cache.update(
                pre_rope_keys,  # StreamingLLM: store pre-RoPE keys
                value_states,
                attn_weights.detach(),  # H2O: use attention scores for heavy-hitter detection
            )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

def convert_to_liquid_fusion(model, config):
    """Convert a model to use liquid fusion attention."""
    device = next(model.parameters()).device
    
    # Get the original model's dimensions
    original_layer = model.model.layers[0].self_attn
    
    # Set dimensions from first layer
    config.hidden_size = original_layer.q_proj.weight.shape[1]  # 2048
    config.num_attention_heads = original_layer.num_heads  # 32
    config.sequence_length = original_layer.k_proj.weight.shape[0]  # 256
    
    # Convert attention layers
    for i, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        try:
            liquid_attn = LiquidFusion(config, layer_idx=i)
            # Verify dimensions match before weight transfer
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                orig_weight = getattr(original_attn, name).weight
                new_weight = getattr(liquid_attn, name).weight
                if orig_weight.shape != new_weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {name}: "
                        f"original {orig_weight.shape} vs new {new_weight.shape}"
                    )
            
            # Transfer weights
            with torch.no_grad():
                for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    getattr(liquid_attn, name).weight.data.copy_(
                        getattr(original_attn, name).weight.data
                    )
            
            # Ensure proper device placement
            liquid_attn = liquid_attn.to(device)
            layer.self_attn = liquid_attn
            
        except Exception as e:
            print(f"\nError during initialization:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise
    
    return model
