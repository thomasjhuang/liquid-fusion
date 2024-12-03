import torch

# from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.h2o.h2o_gptneox import GPTNeoXAttention_Mask, convert_kvcache_gpt_neox_heavy_recent
from models.h2o.h2o_llama import LlamaAttention_heavy_hitter, convert_kvcache_llama_heavy_recent
from models.h2o.h2o_opt import OPTAttention_Mask, convert_kvcache_opt_heavy_recent

class ModelLoader:
    ATTENTION_CONVERTERS = {
        "llama": convert_kvcache_llama_heavy_recent,
        "opt": convert_kvcache_opt_heavy_recent,
        "gpt_neox": convert_kvcache_gpt_neox_heavy_recent
    }

    def __init__(self, config):
        self.config = config

    def load_model_and_tokenizer(self):
        # Load config first
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Load model with config
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=torch.float16 if self.config.dtype == "float16" else None,
            trust_remote_code=True
        )

        # Set up tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Configure tokenizer
        if self.config.model_type in ["gpt_neox", "llama"]:
            tokenizer.model_max_length = int(1e30)
            if "Llama-3" in self.config.model_name:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
            else:
                tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.add_bos_token = False

        # Apply attention modifications if needed
        if self.config.attention_type == "heavy_hitter":
            model_config.heavy_ratio = self.config.heavy_ratio
            model_config.recent_ratio = self.config.recent_ratio
            converter = self.ATTENTION_CONVERTERS[self.config.model_type]
            model = converter(model, model_config)

        # Handle different devices
        if self.config.dtype == "float16":
            model = model.half()
        elif self.config.dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        
        model = model.eval()
        
        # Move to appropriate device
        if self.config.device == "cuda":
            assert torch.cuda.is_available(), "CUDA not available"
            model = model.cuda()
        elif self.config.device == "mps":
            assert torch.backends.mps.is_available(), "MPS not available"
            model = model.to("mps")
        elif self.config.device == "cpu":
            model = model.cpu()
        else:
            raise ValueError(f"Unknown device: {self.config.device}")

        return model, tokenizer