import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import math

class BaseAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout = nn.Dropout(dropout)
        self.scaling = head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        bsz, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Convert scores to probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attn_output = torch.matmul(attention_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attention_probs if output_attentions else None, None

class AttentionModelWrapper(nn.Module):
    def __init__(self, attention_mechanism):
        super().__init__()
        self.attention = attention_mechanism
        self.device = next(attention_mechanism.parameters()).device
        
    def to(self, device):
        self.device = device
        return super().to(device)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, output_attentions=False, **kwargs):
        dtype = next(self.attention.parameters()).dtype
        batch_size, seq_length = input_ids.shape
        hidden_size = self.attention.hidden_size
        
        # Create hidden states matching the input sequence length
        hidden_states = torch.randn(
            batch_size, 
            seq_length,
            hidden_size,
            dtype=dtype,
            device=self.device
        )
        
        # Prepare attention mask
        if attention_mask is not None:
            # For StreamingAttention, keep mask as [batch_size, 1, seq_length, seq_length]
            if isinstance(self.attention, StreamingAttention):
                attention_mask = attention_mask.unsqueeze(1)
                attention_mask = (1.0 - attention_mask) * -10000.0
            else:
                # For other attention types, use [batch_size, 1, seq_length, seq_length]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
            
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=dtype, device=self.device) * -10000.0,
                diagonal=1
            )
            
            # Combine masks appropriately based on attention type
            if isinstance(self.attention, StreamingAttention):
                attention_mask = attention_mask + causal_mask.unsqueeze(0)
            else:
                attention_mask = attention_mask + causal_mask[None, None, :, :]
        
        try:
            attention_output, attention_weights, past_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            
            assert attention_output.shape == hidden_states.shape, \
                f"Output shape {attention_output.shape} doesn't match input shape {hidden_states.shape}"
            
            # Compute loss (simplified for testing)
            loss = torch.mean(attention_output)
            
            class OutputWrapper:
                def __init__(self, loss, attentions):
                    self.loss = loss
                    self.attentions = attentions
            
            return OutputWrapper(loss, (attention_weights,) if output_attentions else None)
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes:")
            print(f"input_ids: {input_ids.shape}")
            print(f"hidden_states: {hidden_states.shape}")
            if attention_mask is not None:
                print(f"attention_mask: {attention_mask.shape}")
            raise e

def process_batch(text, model, max_length=512, batch_size=1):
    """
    Process a batch of text through the model and return metrics
    """
    try:
        # Handle single text vs list of texts
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            text = [str(text)]
        text = [t for t in text if t.strip()]
        if not text:
            raise ValueError("No valid text in batch")
        # Ensure we have enough texts
        while len(text) < batch_size:
            text.extend(text[:batch_size - len(text)])
            
        text = text[:batch_size]
        device = next(model.parameters()).device
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create proper attention mask
        attention_mask = inputs['attention_mask']
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        inputs['attention_mask'] = extended_attention_mask
        inputs['output_attentions'] = True
        
        # Track memory and time
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - start_time
        memory_used = (torch.cuda.memory_allocated() - memory_before) if torch.cuda.is_available() else 0
        
        metrics = {
            'perplexity': math.exp(outputs.loss.item()) if hasattr(outputs, 'loss') else 0.0,
            'memory_usage': memory_used / 1024**2,
            'inference_time': inference_time,
            'attention_sparsity': 0.0
        }
        
        # Calculate attention sparsity if weights are available
        if outputs.attentions is not None:
            attn_weights = outputs.attentions[0] 
            sparsity = (attn_weights < 0.01).float().mean().item()
            metrics['attention_sparsity'] = sparsity
            
        return metrics, outputs.attentions
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        if isinstance(text, (list, tuple)):
            logger.error(f"Input shapes: text length = {len(text)}, batch_size = {batch_size}")
        return None, None

def calculate_perplexity(loss):
    return torch.exp(loss).item()

def calculate_sparsity(attention_weights, threshold=0.01):
    total_elements = attention_weights.numel()
    sparse_elements = (attention_weights < threshold).sum().item()
    return sparse_elements / total_elements

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0