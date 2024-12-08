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
            if 'goal' in item:  # PIQA format
                choices = f"1. {item['sol1']}\n2. {item['sol2']}"
                text = f"Goal: {item['goal']}\nChoices:\n{choices}"
                target = item.get('label', '')  # PIQA target
            elif 'ctx' in item:  # Hellaswag format
                choices = "\n".join([f"{i+1}. {ending}" for i, ending in enumerate(item['endings'])])
                text = f"Context: {item['ctx']}\nChoices:\n{choices}"
                target = item.get('label', '')  # Hellaswag target
            elif 'question' in item and 'choices' in item:  # MMLU format
                choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(item['choices'])])
                text = f"Question: {item['question']}\nChoices:\n{choices}"
                target = item.get('answer', '')  # MMLU target
            elif 'document' in item:  # XSUM format
                text = f"Article: {item['document']}\nSummarize the above article:"
                target = item.get('summary', '')  # XSUM target
            elif 'article' in item:  # CNN/DailyMail format
                text = f"Article: {item['article']}\nSummarize the above article:"
                target = item.get('highlights', '')  # CNN/DailyMail target
            elif 'text' in item:
                text = item['text']
                target = ''
            elif 'premise' in item and 'choice1' in item:  # COPA format
                choices = f"1. {item['choice1']}\n2. {item['choice2']}"
                text = f"Premise: {item['premise']}\nQuestion: {item['question']}\nChoices:\n{choices}"
                target = str(item.get('label', ''))  # COPA target
            else:
                raise ValueError(f"Unknown dataset format with keys: {item.keys()}")
        else:
            text = str(item)
            target = ''

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'target': target
        }

def collate_fn(batch):
    """Combine items into batches."""
    return {
        'text': [item['text'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'target': [item['target'] for item in batch]
    }

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
                raw_dataset = load_dataset(
                    dataset_name,
                    dataset_config.config,
                    split=split,
                    trust_remote_code=True  # Add trust_remote_code
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

    def get_dataloader(self):
        """Get dataloader for the first dataset in config"""
        if not self.config.datasets:
            raise ValueError("No datasets configured")
            
        dataset_config = self.config.datasets[0]
        batch_size = getattr(self.config, 'batch_size', 1)
        
        # Use first split if multiple are specified
        split = dataset_config.splits[0]
        if '[' in split:  # Handle slice notation like "test[:10]"
            split = split.split('[')[0]
            
        return self.load_dataset(
            dataset_config.name,
            split=split,
            batch_size=batch_size
        )
