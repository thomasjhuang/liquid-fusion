from typing import Dict, Any, Optional, List
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionDataset:
    def __init__(
        self,
        dataset_name: str = "xsum",
        split: str = "test",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        self.seed = seed
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Dataset:
        logger.info(f"Loading dataset {self.dataset_name} ({self.split} split)")
        try:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            if self.max_samples and self.max_samples < len(dataset):
                dataset = dataset.shuffle(seed=self.seed)
                dataset = dataset.select(range(self.max_samples))
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
            
    def prepare_for_attention(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_column: str = "document"
    ) -> Dataset:
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        
        return tokenized_dataset
    
    def get_sample_texts(self, num_samples: int = 5) -> List[str]:
        samples = self.dataset.shuffle(seed=self.seed).select(range(num_samples))
        return [sample['document'] for sample in samples]
    
    def save_samples(self, output_dir: str, num_samples: int = 100):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        samples = self.get_sample_texts(num_samples)
        
        with open(output_path / 'samples.json', 'w') as f:
            json.dump({'samples': samples}, f, indent=2)