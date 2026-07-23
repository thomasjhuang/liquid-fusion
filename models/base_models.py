from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from models.attention.liquid_fusion import convert_to_liquid_fusion

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.dtype = getattr(torch, config.dtype)

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        logger.info(f"ModelLoader: Loading model with attention type: {self.config.attention_type}")
        custom_attention = self.config.attention_type in {
            "streaming",
            "h2o",
            "liquid_fusion",
        }
        attn_implementation = "eager" if custom_attention else "sdpa"
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        model_config.use_cache = True
        model_config._attn_implementation_internal = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=self.dtype,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.model_revision,
        ).eval()
        model = model.to(self.config.device)

        if self.config.attention_type == "streaming":
            model = convert_to_liquid_fusion(
                model,
                sink_size=self.config.start_size,
                heavy_budget=0,
                recent_budget=self.config.recent_size,
            )
        elif self.config.attention_type == "h2o":
            model = convert_to_liquid_fusion(
                model,
                sink_size=0,
                heavy_budget=self.config.heavy_budget,
                recent_budget=self.config.recent_budget,
            )
        elif self.config.attention_type == "liquid_fusion":
            model = convert_to_liquid_fusion(
                model,
                sink_size=self.config.sink_size,
                heavy_budget=self.config.heavy_budget,
                recent_budget=self.config.recent_budget,
            )
        elif self.config.attention_type not in {"default", "full"}:
            raise ValueError(f"Unsupported attention strategy: {self.config.attention_type}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
            revision=self.config.model_revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer