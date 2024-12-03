import argparse
import logging
import numpy as np
import torch
import json
import os
import time
import signal
import tqdm
import copy
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
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

    # Constants
    MAX_GENERATION_TIME = 30  # seconds per sample
    SAVE_INTERVAL = 5  # Save every N samples

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

        if args.enable_small_cache:
            print('Enable Small Cache Size')
            config.heavy_ratio = args.heavy_ratio
            config.recent_ratio = args.recent_ratio
            checkpoint = copy.deepcopy(model.state_dict())
            model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
            model.load_state_dict(checkpoint)

        model.half().eval().cuda()
        logger.info(args)

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
                try:
                    start_time = time.time()
                    request = request['request']
                    result = {'request': request, 'result': {}}
                    prompt = request['prompt']
                    temperature = request['temperature']
                    stop = request['stop']

                    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

                    # Skip if input is too long
                    if len(input_ids[0]) > model.config.max_position_embeddings:
                        logger.warning(f"Input too long ({len(input_ids[0])} tokens), skipping")
                        continue

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    with torch.inference_mode(), timeout(seconds=MAX_GENERATION_TIME):
                        output_sequences = model.generate(
                            input_ids=input_ids,
                            max_length=request['max_tokens'] + len(input_ids[0]),
                            temperature=temperature,
                            top_k=args.k,
                            top_p=request['top_p'],
                            do_sample=True,
                            num_return_sequences=request['n'],
                            return_dict_in_generate=True,
                            output_scores=True,
                        )

                    for name, m in model.named_modules():
                        if isinstance(m, TAGET_MODULE[args.model_arch]):
                            m._reset_masks()

                    tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
                    logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
                    top_logprobs = [{i: v} for i, v in zip(tokens, logprobs)]

                    generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
                    generate_text = generate_text[: generate_text.find(stop[0])]

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

                except TimeoutError:
                    logger.error(f"Generation timed out after {MAX_GENERATION_TIME}s")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(f"Error processing request {i}:")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error message: {str(e)}")
                    logger.error("Full traceback:")
                    logger.error(traceback.format_exc())
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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

    except Exception as e:
        logger.error("Fatal error in main:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()