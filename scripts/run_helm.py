import argparse
import logging
import numpy as np
import torch
import json
import os
import time
import signal
import tqdm
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
}

TAGET_MODULE = {
    "llama": LlamaAttention_heavy_hitter,
}

def set_seed(args):
    """Set random seeds for reproducibility."""
    try:
        # CPU seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # CUDA seed - only if CUDA is actually available and being used
        if not args.no_cuda and torch.cuda.is_available() and args.n_gpu > 0:
            try:
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            except Exception as e:
                logger.warning(f"Warning: Could not set CUDA seed: {e}")
                logger.warning("Continuing with CPU-only seed initialization")
    except Exception as e:
        logger.warning(f"Warning: Could not fully set random seed: {e}")
        logger.warning("Continuing with partial seed initialization")
    
    # Set deterministic flags if needed
    try:
        if hasattr(args, 'deterministic') and args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.warning(f"Warning: Could not set deterministic flags: {e}")

def get_repo_root():
    """Get the absolute path to the repository root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Args:
    input_path: str
    output_path: str
    model_name: str
    model_arch: str
    cache_dir: str = "./cache"
    heavy_ratio: float = 0.1
    recent_ratio: float = 0.1
    enable_small_cache: bool = True
    sample_num: int = 10
    no_cuda: bool = False
    fp16: bool = False
    k: int = 0
    seed: int = 42
    device: Optional[torch.device] = None
    n_gpu: Optional[int] = None

def run_experiment(args):
    """Main experiment function that can be called directly or from command line."""
    # Constants
    MAX_GENERATION_TIME = 30  # seconds per sample
    SAVE_INTERVAL = 5  # Save every N samples

    # Convert relative paths to absolute paths using repo root
    repo_root = get_repo_root()
    args.input_path = os.path.join(repo_root, args.input_path)
    args.output_path = os.path.join(repo_root, args.output_path)
    args.cache_dir = os.path.join(repo_root, args.cache_dir)

    # Setup device
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model and tokenizer with CUDA handling
    try:
        # First load config and tokenizer (these don't use CUDA)
        config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)
        
        # Set model loading options
        model_kwargs = {
            "config": config,
            "cache_dir": args.cache_dir,
            "device_map": None,  # Disable auto device mapping
            "torch_dtype": torch.float16 if args.fp16 else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        # Load model on CPU first
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs
        )
        
        # Apply heavy hitter modifications while still on CPU
        if args.enable_small_cache:
            print('Enable Small Cache Size')
            config.heavy_ratio = args.heavy_ratio
            config.recent_ratio = args.recent_ratio
            with torch.no_grad():
                model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
        
        # Now move to GPU if available
        if args.device.type == "cuda":
            # Move model to GPU in parts to avoid OOM
            for param in model.parameters():
                param.data = param.data.to(args.device)
            model = model.to(args.device)
            
            if args.fp16:
                model = model.half()
        
        model.eval()
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

    # After loading the model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Load requests
    requests = []
    with open(args.input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples'.format(args.sample_num))
    requests = requests[:args.sample_num]

    results = []
    with torch.no_grad():
        for i, request in enumerate(tqdm.tqdm(requests)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                start_time = time.time()
                request = request['request']
                result = {'request': request, 'result': {}}
                prompt = request['prompt']
                temperature = request['temperature']
                stop = request['stop']

                # Create input_ids and attention_mask
                tokenizer_output = tokenizer(
                    prompt, 
                    add_special_tokens=False, 
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                input_ids = tokenizer_output['input_ids'].to(model.device)
                attention_mask = tokenizer_output['attention_mask'].to(model.device)

                with torch.inference_mode(), timeout(seconds=MAX_GENERATION_TIME):
                    # Add safety parameters to generation
                    generation_config = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'max_length': request['max_tokens'] + len(input_ids[0]),
                        'temperature': max(temperature, 1e-7),  # Prevent temperature=0
                        'top_k': args.k,
                        'top_p': min(max(request['top_p'], 1e-7), 1.0),  # Ensure valid range
                        'do_sample': True,
                        'num_return_sequences': request['n'],
                        'return_dict_in_generate': True,
                        'output_scores': True,
                        'pad_token_id': tokenizer.pad_token_id,
                        'eos_token_id': tokenizer.eos_token_id,
                        'use_cache': True,
                        # Add safety parameters
                        'min_length': 1,
                        'repetition_penalty': 1.0,
                        'no_repeat_ngram_size': 0,
                        'early_stopping': True,
                    }

                    # Generate with extra error checking
                    try:
                        output_sequences = model.generate(**generation_config)
                    except RuntimeError as e:
                        if "device-side assert triggered" in str(e):
                            logger.error("CUDA assertion error during generation. Retrying with safer parameters...")
                            # Retry with even safer parameters
                            generation_config.update({
                                'temperature': 1.0,
                                'top_p': 0.9,
                                'do_sample': True,
                                'num_beams': 1,
                            })
                            output_sequences = model.generate(**generation_config)

                    # Reset masks after generation
                    for name, m in model.named_modules():
                        if isinstance(m, TAGET_MODULE[args.model_arch]):
                            m._reset_masks()

                    # Process outputs
                    generated_tokens = output_sequences['sequences'][:, len(input_ids[0]):]
                    tokens = tokenizer.convert_ids_to_tokens(generated_tokens.squeeze(0))
                    
                    # Safely compute logprobs
                    logprobs = []
                    for logits in output_sequences['scores']:
                        # Ensure valid probabilities
                        logits = logits.float()  # Convert to float32 for numerical stability
                        log_probs = torch.log_softmax(logits, dim=-1)
                        max_logprob = log_probs.max().item()
                        logprobs.append(max_logprob)

                    top_logprobs = [{i: v} for i, v in zip(tokens, logprobs)]

                    # Decode generated text
                    generate_text = tokenizer.decode(generated_tokens.squeeze(0))
                    if stop:
                        generate_text = generate_text[: generate_text.find(stop[0])] if stop[0] in generate_text else generate_text

                    generation_time = time.time() - start_time

                    result['result'] = {
                        "choices": [{
                            "text": generate_text,
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": logprobs,
                                "top_logprobs": top_logprobs,
                                "text_offset": []
                            },
                            "finish_reason": "length"
                        }],
                        "request_time": {
                            "batch_time": generation_time,
                            "batch_size": 1
                        }
                    }
                    results.append(result)

            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                continue

            # Save intermediate results
            if (i + 1) % SAVE_INTERVAL == 0:
                with open(args.output_path, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')

    # Save final results
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_arch", type=str, default="llama")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint/")
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--enable_small_cache", action="store_true")
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()
