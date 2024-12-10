from models.streaming_llm.kv_cache import StartRecentKVCache
import torch


def enable_streaming_llm(model, start_size, recent_size):
    """
    Set the model instance to use streaming attention with attention sinks
    """
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        
        # Create attention mask in advance
        max_length = model.config.max_position_embeddings
        attention_mask = create_streaming_attention_mask(
            token_len=max_length,
            global_size=start_size,  # Using start_size as global_size
            band_size=recent_size    # Using recent_size as band_size
        )
        attention_mask = (~attention_mask).to(torch.float16) * torch.finfo(torch.float16).min

        # Get device list for multi-device support
        device_list = set([str(layer.self_attn.o_proj.weight.device) for layer in model.layers])
        attention_mask_dict = {device: attention_mask.to(device) for device in device_list}

        # Modify each attention layer
        for layer in model.layers:
            from types import MethodType
            from models.streaming_llm.modify_llama import LlamaAttention_streamingllm_forward
            
            # Replace forward method
            layer.self_attn.forward = MethodType(LlamaAttention_streamingllm_forward, layer.self_attn)
            
            # Add streaming-specific attributes
            layer.self_attn.global_size = start_size
            layer.self_attn.band_size = recent_size
            layer.self_attn.global_mask = attention_mask_dict[str(layer.self_attn.o_proj.weight.device)]

        # Create KV cache for managing cache state
        kv_cache = StartRecentKVCache(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        return kv_cache
    else:
        raise ValueError(f"got {model.config.model_type}")

def create_streaming_attention_mask(token_len, global_size, band_size):
    """Create attention mask for streaming attention with global tokens and banded pattern"""
    # Start with all False (0)
    mask = torch.zeros(1, 1, token_len, token_len, dtype=torch.bool)
    
    # Apply causal mask with band pattern
    for i in range(token_len):
        # Ensuring causality and the band pattern
        start = max(i - band_size + 1, 0)
        end = min(i + 1, token_len)  # Causal limit
        mask[:, :, i, start:end] = True
    
    # Add global attention to first tokens
    if global_size > 0:
        for i in range(token_len):
            mask[:, :, i, :min(global_size, i+1)] = True
    
    return mask
