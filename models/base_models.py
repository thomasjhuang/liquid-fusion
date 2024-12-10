from typing import Optional, Tuple
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.h2o.h2o_llama import LlamaAttention_heavy_hitter, convert_kvcache_llama_heavy_recent
from models.h2o.cache_manager import KVCacheManager
from models.streaming_llm.enable_streaming_llm import enable_streaming_llm
from models.attention.liquid_fusion import convert_to_liquid_fusion
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    ATTENTION_CONVERTERS = {
        "llama": {
            "heavy_hitter": convert_kvcache_llama_heavy_recent,
            "streaming": lambda model, config: enable_streaming_llm(
                model,
                start_size=config.window_size,
                recent_size=config.sequence_length - config.window_size
            ),
            "liquid_fusion": convert_to_liquid_fusion
        },
    }

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        logger.info(f"ModelLoader: Loading model with attention type: {self.config.attention_type}")
        
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.use_cache = True
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        
        if self.config.attention_type == "streaming":
            enable_streaming_llm(
                model,
                start_size=self.config.start_size,
                recent_size=self.config.recent_size
            )
        elif self.config.attention_type == "h2o":
            model_config.heavy_ratio = getattr(self.config, 'heavy_ratio', 0.1)
            model_config.recent_ratio = getattr(self.config, 'recent_ratio', 0.1)
            model = convert_kvcache_llama_heavy_recent(model, model_config)
        elif self.config.attention_type == "liquid_fusion":
            model_config.start_size = getattr(self.config, 'start_size', 4)
            model_config.recent_size = getattr(self.config, 'recent_size', 64)
            model_config.heavy_ratio = getattr(self.config, 'heavy_ratio', 0.1)
            model_config.recent_ratio = getattr(self.config, 'recent_ratio', 0.1)
            model = convert_to_liquid_fusion(model, model_config)
        
        model = model.to(self.config.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return model, tokenizer

    def _configure_model(self, model_config):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=torch.float16 if self.config.dtype == "float16" else None,
            trust_remote_code=True
        )

        kv_cache = None
        # Apply attention modifications if needed
        if hasattr(self.config, 'attention_type') and self.config.attention_type:
            if self.config.model_type in self.ATTENTION_CONVERTERS:
                converter = self.ATTENTION_CONVERTERS[self.config.model_type].get(
                    self.config.attention_type
                )
                if converter:
                    if self.config.attention_type == "streaming":
                        kv_cache = converter(model, model_config)
                    else:
                        model = converter(model, model_config)
                else:
                    raise ValueError(f"Attention type {self.config.attention_type} not supported for {self.config.model_type}")

        # Handle dtype and device placement
        model = self._set_model_dtype_and_device(model)
        return model, kv_cache

    def _configure_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.config.model_type in ["gpt_neox", "llama"]:
            tokenizer.model_max_length = int(1e30)
            if "Llama-3" in self.config.model_name:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
            else:
                tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.add_bos_token = False
            
        return tokenizer

    def _set_model_dtype_and_device(self, model):
        # Set dtype
        if self.config.dtype == "float16":
            model = model.half()
        elif self.config.dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        
        model = model.eval()
        
        # Set device
        device_map = {
            "cuda": lambda m: m.cuda() if torch.cuda.is_available() else None,
            "mps": lambda m: m.to("mps") if torch.backends.mps.is_available() else None,
            "cpu": lambda m: m.cpu()
        }
        
        if self.config.device not in device_map:
            raise ValueError(f"Unknown device: {self.config.device}")
            
        device_fn = device_map[self.config.device]
        model = device_fn(model)
        if model is None:
            raise RuntimeError(f"{self.config.device} not available")
            
        return model