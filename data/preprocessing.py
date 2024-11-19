from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedTokenizer
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AttentionPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        device: str = "cuda"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
    def prepare_batch(
        self,
        texts: List[str],
        return_attention_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs
    
    def create_causal_attention_mask(
        self,
        batch_size: int,
        sequence_length: int
    ) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length),
            diagonal=1
        ).bool()
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, 1, sequence_length, sequence_length)
        mask = mask.to(self.device)
        
        return mask
    
    def create_sliding_window_mask(
        self,
        batch_size: int,
        sequence_length: int,
        window_size: int
    ) -> torch.Tensor:
        mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool)
        
        for i in range(sequence_length):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2 + 1)
            mask[i, start:end] = False
            
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, 1, sequence_length, sequence_length)
        mask = mask.to(self.device)
        
        return mask

class DataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        pad_to_multiple_of: Optional[int] = 8
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        batch = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch['labels'] = batch['input_ids'].clone()
        return batch
