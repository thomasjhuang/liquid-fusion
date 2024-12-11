from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from transformers.models.llama.modeling_llama import LlamaAttention
from models.streaming_llm.modify_llama import LlamaModel_use_streamingllm_attention
from models.h2o.h2o_llama import convert_kvcache_llama_heavy_recent
from models.attention.liquid_fusion import convert_to_liquid_fusion

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.dtype = torch.float16 if not hasattr(config, 'dtype') else getattr(torch, config.dtype)

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        logger.info(f"ModelLoader: Loading model with attention type: {self.config.attention_type}")
        attn_implementation = "eager"
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.use_cache = True
        model_config._attn_implementation_internal = attn_implementation
        print(f"using {model_config._attn_implementation_internal} attention implementation")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
        
        # Move model to device first
        model = model.to(self.config.device)
        
        if self.config.attention_type == "streaming":
            print("Applying StreamingLLM attention")
            LlamaModel_use_streamingllm_attention(
                model.model, 
                global_size=self.config.start_size,
                band_size=self.config.recent_size,
                device=self.config.device,
                max_length=self.config.sequence_length
            )
            print("StreamingLLM attention applied successfully")
        elif self.config.attention_type == "h2o":
            model = convert_kvcache_llama_heavy_recent(
                model,
                heavy_budget=self.config.heavy_budget,
                recent_budget=self.config.recent_budget
            )
            print(f"Applied H2O attention with heavy_budget={self.config.heavy_budget}, recent_budget={self.config.recent_budget}")
        elif self.config.attention_type == "liquid_fusion":
            model = convert_to_liquid_fusion(
                model,
                heavy_budget=self.config.heavy_budget,
                recent_budget=self.config.recent_budget
            )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return model, tokenizer