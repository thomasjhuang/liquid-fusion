import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig
import json
from typing import List, Optional, Tuple, Dict
import time
from dataclasses import dataclass
import lm_eval
from lm_eval import evaluator, tasks, utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.tasks import get_task_dict
from functools import partial
from datasets import load_dataset
import os
import ossaudiodev
from tqdm import tqdm
import numpy as np
import multiprocessing
import ftfy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from itertools import zip_longest

import json
from typing import List, Tuple, Dict
from lm_eval.api.model import LM
from lm_eval.tasks import get_task_dict
from lm_eval.evaluator import evaluate
from datasets import load_dataset, Dataset, DatasetDict
import os



print(f"torch version: {torch.__version__}")
print(f"transformers version: {transformers.__version__}")


def grouper(n, iterable, fillvalue):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


# divide the seq length by 2 until it would truncate actual context
def shrink_seq(examples, min_seq=None):
    length = examples["obs"].shape[-1]

    new_length = length // 2

    if min_seq is not None:
        if new_length < min_seq:
            return examples

    max_length = np.max(examples["eval_mask"] * np.arange(0, length)) + 1

    if max_length < new_length:
        examples["obs"] = examples["obs"][:, :new_length]
        examples["target"] = examples["target"][:, :new_length]
        examples["eval_mask"] = examples["eval_mask"][:, :new_length]

        return shrink_seq(examples, min_seq=min_seq)
    else:
        return examples


def sample_batch(examples, bs, zero_example_shape):
    zero_example = {
        "obs": np.zeros_like(zero_example_shape["obs"]),
        "target": np.zeros_like(zero_example_shape["target"]),
        "eval_mask": np.zeros_like(zero_example_shape["eval_mask"]),
        "ctx_length": 0,
    }

    for batch in grouper(bs, examples, zero_example):
        batch_flattened = {
            "obs": [],
            "target": [],
            "eval_mask": [],
            "ctx_length": [],
            "text": [],
        }

        for sample in batch:
            batch_flattened["obs"].append(sample["obs"])
            batch_flattened["target"].append(sample["target"])
            batch_flattened["eval_mask"].append(sample["eval_mask"])
            batch_flattened["ctx_length"].append(sample["ctx_length"])
            batch_flattened["text"].append(sample["text"])

        batch_flattened["obs"] = np.array(batch_flattened["obs"])
        batch_flattened["target"] = np.array(batch_flattened["target"])
        batch_flattened["eval_mask"] = np.array(batch_flattened["eval_mask"])
        batch_flattened["ctx_length"] = np.array(batch_flattened["ctx_length"])

        yield batch_flattened

tokenizer = None

def process_init():
    global tokenizer
    model_name = os.environ.get('MODEL_NAME', 'facebook/opt-1.3b')

    if model_name == "EleutherAI/gpt-neox-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    elif model_name == 'huggyllama/llama-7b':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_bos_token = False

def process_request(x, seq):
    global tokenizer

    ctx, cont = x
#     ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    cont_text = ftfy.fix_text(cont, normalization="NFKC")
    all_text = ctx_text + cont_text

    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)['input_ids']
    cont_tokens = tokenizer(cont_text, add_special_tokens=False)['input_ids']

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
        "prompt": ctx_text,
        "target": cont_text,
        "text": all_text,
    }


class EvalHarnessAdaptor(LM):
    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(processes=1, initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch):

            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    def generate_until(self, requests):
        """
        Generate tokens until a stopping condition is met for each request.
        """
        # Simple implementation that returns empty strings
        return ["" for _ in requests]

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return self.seq

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch

    @property
    def device(self):
        return "cuda"

# Cell 1: Load and cache dataset
def load_cached_dataset(dataset_name="hellaswag", split="validation"):
    """Load dataset and cache it in memory"""
    if not hasattr(load_cached_dataset, 'cached_datasets'):
        load_cached_dataset.cached_datasets = {}

    cache_key = f"{dataset_name}_{split}"
    if cache_key not in load_cached_dataset.cached_datasets:
        print(f"Loading {dataset_name} dataset ({split} split)...")
        dataset = load_dataset(dataset_name, split=split)
        load_cached_dataset.cached_datasets[cache_key] = dataset
        print(f"Dataset cached! Size: {len(dataset)} examples")
    else:
        print(f"Using cached {dataset_name} dataset ({split} split)")
        dataset = load_cached_dataset.cached_datasets[cache_key]

    return dataset

# Example usage
dataset = load_cached_dataset("hellaswag", "validation")


class DryRunner(LM):
    def __init__(self, output_file):
        super().__init__()
        self.output_file = output_file
        self.cache_hook = None

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        outputs = []
        with open(self.output_file, 'a') as f:
            for req in requests:
                context, continuation = req.args
                item = {
                    "prompt": context + continuation,
                    "context": context,
                    "continuation": continuation,
                    "echo": True
                }
                f.write(json.dumps(item) + '\n')
                outputs.append((0.0, True))
        return outputs

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError()

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        return ""

    @property
    def tokenizer_name(self) -> str:
        return "dummy-tokenizer"

def generate_task_data(output_file: str = "task_prompts.jsonl",
                      task_name: str = "hellaswag",
                      num_fewshot: int = 0,
                      cached_dataset=None):

    # Clear output file
    with open(output_file, 'w') as f:
        pass

    # Create task dict
    task_dict = get_task_dict([task_name])

    # Modify the task's dataset if we have a cached one
    if cached_dataset is not None:
        # Create a DatasetDict with the original dataset structure
        dataset_dict = DatasetDict({
            'validation': cached_dataset
        })

        for task in task_dict.values():
            task.dataset = dataset_dict

    # Create runner and evaluate
    runner = DryRunner(output_file)

    results = evaluate(
        lm=runner,
        task_dict=task_dict,
        limit=None,
        bootstrap_iters=1,
        log_samples=True
    )

    print("Done generating prompts!")
    return results

def load_cached_dataset(dataset_name="hellaswag", split="validation"):
    """Load and cache dataset"""
    if not hasattr(load_cached_dataset, 'cached_datasets'):
        load_cached_dataset.cached_datasets = {}

    cache_key = f"{dataset_name}_{split}"
    if cache_key not in load_cached_dataset.cached_datasets:
        print(f"Loading {dataset_name} dataset ({split} split)...")
        dataset = load_dataset(dataset_name, split=split)
        load_cached_dataset.cached_datasets[cache_key] = dataset
        print(f"Dataset cached! Size: {len(dataset)} examples")
    else:
        print(f"Using cached {dataset_name} dataset ({split} split)")
        dataset = load_cached_dataset.cached_datasets[cache_key]

    return dataset

# Usage
dataset = load_cached_dataset("hellaswag", "validation")

# Print example of raw data
print("\nRaw example:")
print(json.dumps(dataset[0], indent=2))

results = generate_task_data(cached_dataset=dataset)
print("\nResults:", results)



def main():
        # First, let's look at the dataset structure
    dataset = load_cached_dataset("hellaswag", "validation")
    print("Dataset columns:", dataset.column_names)
    print("\nExample row:", dataset[0])

    # Let's examine the generated prompts
    with open("task_prompts.jsonl", "r") as f:
        prompts = [json.loads(line) for line in f]

    # Print a few examples
    print(f"Total prompts generated: {len(prompts)}")
    print("\nExample prompts:")
    for i in range(min(3, len(prompts))):
        print(f"\nPrompt {i+1}:")
        print(json.dumps(prompts[i], indent=2))