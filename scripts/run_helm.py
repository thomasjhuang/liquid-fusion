import argparse
import logging
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.utils_hh.modify_llama import convert_kvcache_llama_heavy_recent

logger = logging.getLogger(__name__)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_repo_root():
    """Get the absolute path to the repository root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_experiment(args):

    repo_root = get_repo_root()
    input_path = os.path.join(repo_root, args.input_path)
    output_path = os.path.join(repo_root, args.output_path)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
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
    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                requests.append(json.loads(line))

    if args.sample_num < len(requests):
        requests = requests[:args.sample_num]

    # Process requests
    results = []
    for request in requests:
        result = {'request': request['request']}
        input_ids = tokenizer(
            request['request']['prompt'], 
            return_tensors='pt'
        ).input_ids.to(device)
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=min(request['request']['max_tokens'] + len(input_ids[0]), 512),  # Adding length limit for memory
                temperature=request['request']['temperature'],
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        tokens = tokenizer.convert_ids_to_tokens(outputs['sequences'][0])[len(input_ids[0]):]
        logprobs = [logits.log_softmax(dim=-1).max().item() 
                   for logits in outputs['scores']]

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
            "request_time": {"batch_time": 0, "batch_size": 1}
        }
        results.append(result)

    # Save results
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

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
    parser.add_argument("--sample_num", type=int, default=100)  # Reduced sample size for testing
    
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()
