# data.py
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any
import torch
from datasets.utils.logging import set_verbosity_error
import datasets
import torch.multiprocessing as mp
# Set start method to spawn
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

class ModelDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if isinstance(item, dict):
            if 'text' in item:
                text = item['text']
            elif 'question' in item and 'passage' in item:  # BoolQ format
                text = f"Question: {item['question']}\nPassage: {item['passage']}"
            elif 'premise' in item and 'hypothesis' in item:  # CB format
                text = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}"
            elif 'sentence1' in item and 'sentence2' in item:  # WiC/WSC format
                text = f"{item['sentence1']} {item['sentence2']}"
            elif 'goal' in item:  # PIQA format
                text = f"{item['goal']}"
            elif 'ctx' in item:  # Hellaswag format
                text = item['ctx']
            # New format handlers
            elif 'sentence' in item:  # Winogrande format
                text = f"{item['sentence']} Option 1: {item['option1']} Option 2: {item['option2']}"
            elif 'context' in item and 'question' in item:  # PubMedQA format
                text = f"Question: {item['question']}\nContext: {item['context']}"
            elif 'Problem' in item:  # MathQA format
                text = f"Problem: {item['Problem']}\nRationale: {item['Rationale']}"
            else:
                raise ValueError(f"Unknown dataset format with keys: {item.keys()}")
        else:
            text = str(item)

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Add labels for language modeling
        input_ids = encodings['input_ids'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels  # Add labels for perplexity calculation
        }

def collate_fn(batch):
    # Combine all items in the batch
    batch_dict = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }
    return batch_dict

class DatasetManager:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.cached_datasets = {}
        set_verbosity_error()

    def load_dataset(self, dataset_name: str, split: str, batch_size: int) -> Any:
        dataset_config = next((d for d in self.config.datasets if d.name == dataset_name), None)
        if not dataset_config:
            print(f"No configuration found for dataset {dataset_name}")
            return None
            
        cache_key = f"{dataset_name}_{dataset_config.config}_{split}_{batch_size}" if dataset_config.config else f"{dataset_name}_{split}_{batch_size}"
        
        if cache_key not in self.cached_datasets:
            print(f"Loading {dataset_name} dataset{f' ({dataset_config.config})' if dataset_config.config else ''} ({split} split) with batch_size={batch_size}...")
            try:
                # Handle SuperGLUE datasets differently
                if dataset_name == "super_glue":
                    raw_dataset = load_dataset(
                        dataset_name,
                        dataset_config.config,
                        split=split,
                    )
                else:
                    raw_dataset = load_dataset(
                        dataset_name,
                        dataset_config.config,
                        split=split,
                    )
                
                processed_dataset = ModelDataset(raw_dataset, self.tokenizer, max_length=self.config.sequence_length)
                
                dataloader = DataLoader(
                    processed_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=collate_fn
                )
                self.cached_datasets[cache_key] = dataloader
                print(f"Dataset cached! Size: {len(raw_dataset)} examples")
                
            except Exception as e:
                print(f"Error loading dataset {dataset_name} split {split}: {str(e)}")
                return None
                
        return self.cached_datasets[cache_key]

    def validate_dataset(self, dataset_name: str) -> bool:
        try:
            datasets.get_dataset_infos(dataset_name)
            return True
        except Exception as e:
            print(f"Warning: Dataset '{dataset_name}' not found or cannot be accessed: {str(e)}")
            return False

    def get_all_datasets(self) -> Dict[str, Any]:
        datasets = {}
        for dataset_config in self.config.datasets:
            for split in dataset_config.splits:
                key = f"{dataset_config.name}_{split}"
                dataset = self.load_dataset(dataset_config.name, split)
                if dataset is not None:
                    datasets[key] = dataset
        return datasets
