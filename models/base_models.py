from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from transformers.models.llama.modeling_llama import LlamaAttention
from models.MoA.models.llama.modeling_llama import LlamaModel_use_streamingllm_attention

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.dtype = torch.float16 if not hasattr(config, 'dtype') else getattr(torch, config.dtype)

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        logger.info(f"ModelLoader: Loading model with attention type: {self.config.attention_type}")
        
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.use_cache = True
        model_config._attn_implementation_internal = "sdpa"
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation=("sdpa"),
        ).eval()
        
        # Move model to device first
        model = model.to(self.config.device)
        
        if self.config.attention_type == "streaming":
            logger.info("Applying StreamingLLM attention")
            # Apply streaming attention directly to the model
            LlamaModel_use_streamingllm_attention(
                model.model,  # Pass the base model
                global_size=self.config.start_size,
                band_size=self.config.recent_size,
                device=self.config.device,
                max_length=model.config.max_position_embeddings
            )
            logger.info("StreamingLLM attention applied successfully")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return model, tokenizer