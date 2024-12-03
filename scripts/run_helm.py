import argparse
import logging
import torch
import json
import os
import time
import signal
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter

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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_repo_root():
    """Get the absolute path to the repository root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_experiment(args):
    # Constants for fail-fast behavior
    MAX_GENERATION_TIME = 30  # seconds per sample
    FAIL_FAST_THRESHOLD = 2048  # Maximum input tokens
    MAX_TOTAL_TOKENS = 2048  # Maximum total tokens (input + output)
    SAVE_INTERVAL = 5  # Save every N samples

    repo_root = get_repo_root()
    input_path = os.path.join(repo_root, args.input_path)
    output_path = os.path.join(repo_root, args.output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = get_device()
    logger.info(f"Using device: {device}")

    try:
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        config = AutoConfig.from_pretrained(
            args.model_name, 
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            use_fast=True,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Apply H2O modifications if enabled
        if args.enable_small_cache:
            logger.info('Enabling H2O Cache')
            config.heavy_ratio = args.heavy_ratio
            config.recent_ratio = args.recent_ratio
            checkpoint = model.state_dict()
            model = convert_kvcache_llama_heavy_recent(model, config)
            model.load_state_dict(checkpoint)

        model.half().eval().to(device)

        # Load requests
        logger.info("Loading requests...")
        requests = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    requests.append(json.loads(line))

        total_requests = len(requests)
        logger.info(f"Loaded {total_requests} requests")
        
        if args.sample_num < total_requests:
            logger.info(f'Sampling {args.sample_num} examples')
            requests = requests[:args.sample_num]

        # Process requests
        results = []
        for i, request in enumerate(tqdm.tqdm(requests, desc="Processing requests")):
            try:
                start_time = time.time()
                logger.debug(f"Processing request {i+1}/{len(requests)}")
                
                result = {'request': request['request']}
                prompt = request['request']['prompt']
                
                # Check input length
                input_ids = tokenizer(
                    prompt, 
                    return_tensors='pt'
                ).input_ids.to(device)
                
                input_length = len(input_ids[0])
                if input_length > FAIL_FAST_THRESHOLD:
                    logger.warning(f"Input too long ({input_length} tokens), skipping")
                    continue

                # Calculate max new tokens
                max_new_tokens = min(
                    request['request']['max_tokens'],
                    MAX_TOTAL_TOKENS - input_length
                )
                
                if max_new_tokens <= 0:
                    logger.warning(f"No room for generation after input ({input_length} tokens), skipping")
                    continue

                logger.debug(f"Input length: {input_length} tokens, generating up to {max_new_tokens} new tokens")
                
                # Generate with timeout
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    with torch.inference_mode(), timeout(seconds=MAX_GENERATION_TIME):
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
                            temperature=request['request']['temperature'],
                            do_sample=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        
                        # Reset masks after generation
                        for name, m in model.named_modules():
                            if isinstance(m, LlamaAttention_heavy_hitter):
                                m._reset_masks()
                                
                except TimeoutError:
                    logger.error(f"Generation timed out after {MAX_GENERATION_TIME}s")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear GPU memory
                    continue
                
                # Process outputs
                tokens = tokenizer.convert_ids_to_tokens(outputs['sequences'][0])[len(input_ids[0]):]
                logprobs = [logits.log_softmax(dim=-1).max().item() 
                           for logits in outputs['scores']]

                generation_time = time.time() - start_time
                logger.debug(f"Generation took {generation_time:.2f}s")

                result['result'] = {
                    "choices": [{
                        "text": tokenizer.decode(outputs['sequences'][0][len(input_ids[0]):]),
                        "logprobs": {
                            "tokens": tokens,
                            "token_logprobs": logprobs,
                            "top_logprobs": [{t: p} for t, p in zip(tokens, logprobs)],
                            "text_offset": []
                        },
                        "finish_reason": "length"
                    }],
                    "request_time": {"batch_time": generation_time, "batch_size": 1}
                }
                results.append(result)

                # Save intermediate results
                if (i + 1) % SAVE_INTERVAL == 0:
                    logger.debug(f"Saving intermediate results after {i+1} samples")
                    with open(output_path, 'w') as f:
                        for result in results:
                            f.write(json.dumps(result) + '\n')

            except Exception as e:
                logger.error(f"Error processing request {i}: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        # Save final results
        logger.info("Saving final results...")
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_arch", type=str, default="llama")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--enable_small_cache", action="store_true")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_experiment(args)

if __name__ == "__main__":
    main()
