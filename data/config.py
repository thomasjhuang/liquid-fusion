# config.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_RANKING_BINARY,
)

class AdapterMethod(str, Enum):
    """Mapping of task types to HELM adapter methods"""
    MULTIPLE_CHOICE = ADAPT_MULTIPLE_CHOICE_JOINT
    MULTIPLE_CHOICE_SEPARATE = ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL
    MULTIPLE_CHOICE_CALIBRATED = ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED
    GENERATION = ADAPT_GENERATION
    LANGUAGE_MODELING = ADAPT_LANGUAGE_MODELING
    RANKING = ADAPT_RANKING_BINARY

@dataclass
class DatasetConfig:
    name: str
    splits: List[str]
    config: Optional[str] = None
    helm_scenario: Optional[str] = None
    helm_args: Optional[Dict[str, Any]] = None
    adapter_method: AdapterMethod = AdapterMethod.MULTIPLE_CHOICE  # Default
    adapter_args: Dict[str, Any] = field(default_factory=dict)

    def get_adapter_spec(self, model_name: str, max_tokens: int, 
                        temperature: float, num_fewshot: int) -> Dict[str, Any]:
        """Get the appropriate adapter specification for this dataset"""
        base_spec = {
            "method": self.adapter_method.value,
            "model": model_name,
            "model_deployment": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_train_trials": num_fewshot,
        }
        return {**base_spec, **self.adapter_args}

def get_default_dataset_configs() -> Dict[str, DatasetConfig]:
    """Get default dataset configurations with appropriate adapter methods"""
    return {
        "hellaswag": DatasetConfig(
            name="hellaswag",
            splits=["validation"],
            helm_scenario="helm.benchmark.scenarios.hellaswag_scenario.HellaswagScenario",
            helm_args={},
            adapter_method=AdapterMethod.MULTIPLE_CHOICE,
            adapter_args={"input_prefix": "Context: ", "output_prefix": "Answer: "}
        ),
        "mmlu": DatasetConfig(
            name="mmlu",
            splits=["validation"],
            helm_scenario="helm.benchmark.scenarios.mmlu_scenario.MMLUScenario",
            helm_args={"subject": "all"},
            adapter_method=AdapterMethod.MULTIPLE_CHOICE,
            adapter_args={"input_prefix": "Question: ", "output_prefix": "Answer: "}
        ),
        "truthful_qa": DatasetConfig(
            name="truthful_qa",
            splits=["validation"],
            helm_scenario="helm.benchmark.scenarios.truthful_qa_scenario.TruthfulQAScenario",
            helm_args={"task": "mc"},
            adapter_method=AdapterMethod.MULTIPLE_CHOICE,
            adapter_args={"input_prefix": "Q: ", "output_prefix": "A: "}
        ),
        "lambada_openai": DatasetConfig(
            name="lambada_openai",
            splits=["test"],
            helm_scenario="helm.benchmark.scenarios.wikitext_103_scenario.Wikitext103Scenario",
            helm_args={},
            adapter_method=AdapterMethod.LANGUAGE_MODELING
        ),
        "natural_qa": DatasetConfig(
            name="natural_qa",
            splits=["validation"],
            helm_scenario="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario",
            helm_args={},
            adapter_method=AdapterMethod.GENERATION,
            adapter_args={"input_prefix": "Question: ", "output_prefix": "Answer: "}
        ),
        "xsum": DatasetConfig(
            name="xsum",
            splits=["test"],
            helm_scenario="helm.benchmark.scenarios.summarization_scenario.SummarizationScenario",
            helm_args={"dataset_name": "xsum"},
            adapter_method=AdapterMethod.GENERATION,
            adapter_args={"input_prefix": "Article: ", "output_prefix": "Summary: "}
        ),
    }

class ModelArchitecture(str, Enum):
    OPT = "opt"
    LLAMA = "llama"
    GPT_NEOX = "gpt-neox"

class ModelSize(str, Enum):
    OPT_6_7B = "facebook/opt-6.7b"
    OPT_13B = "facebook/opt-13b"
    OPT_30B = "facebook/opt-30b"
    OPT_66B = "facebook/opt-66b"
    LLAMA_7B = "meta-llama/Llama-2-7b"
    LLAMA_13B = "meta-llama/Llama-2-13b"
    LLAMA_70B = "meta-llama/Llama-2-70b"
    NEOX_20B = "EleutherAI/gpt-neox-20b"

@dataclass
class BenchmarkConfig:
    model_name: str
    model_type: str
    attention_type: str = "default"
    gradient_checkpointing: bool = False
    datasets: List[DatasetConfig] = field(default_factory=list)
    batch_size: int = 32
    sequence_length: int = 2048
    dtype: str = "bfloat16"
    device: str = "cuda"
    temperature: float = 0.0
    max_tokens: int = 100
    num_fewshot: int = 0
    helm_metrics: List[str] = field(default_factory=list)
    max_sequence_length: int = 2048 
    batch_size: int = 1  
    client_type: str = "huggingface"
    service_type: str = "local"

    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if not self.helm_metrics:
            self.helm_metrics = [
                # Basic Generation Metrics
                "helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
                "helm.benchmark.metrics.basic_metrics.BasicReferenceMetric",
                
                # Specific Metrics
                "helm.benchmark.metrics.exact_match_metric.ExactMatchMetric",
                "helm.benchmark.metrics.rouge_metric.RougeMetric",
                "helm.benchmark.metrics.perplexity_metric.PerplexityMetric",
                "helm.benchmark.metrics.calibration_metric.CalibrationMetric",
                "helm.benchmark.metrics.efficiency_metrics.EfficiencyMetric",
                
                # Task-specific Metrics
                "helm.benchmark.metrics.code_metrics.CodeMetric",
                "helm.benchmark.metrics.toxicity_metrics.ToxicityMetric",
                "helm.benchmark.metrics.bias_metrics.BiasMetric",
                "helm.benchmark.metrics.factuality_metrics.FactualityMetric",
                
                # Additional Stats
                "logprob",
                "num_perplexity_tokens",
                "num_bytes",
                "max_prob",
                "exact_match",
                "predicted_index",
                "ece_10_bin",
                "ece_1_bin",
                "selective_cov_acc_area",
                "selective_acc@10",
                "platt_ece_10_bin",
                "platt_ece_1_bin"
            ]

    def switch_dataset(self, dataset_name: str):
        """Switch datasets based on a string identifier."""
        dataset_configs = get_default_dataset_configs()
        if dataset_name in dataset_configs:
            self.datasets = [dataset_configs[dataset_name]]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(dataset_configs.keys())}")
        

def get_default_datasets() -> List:
    return []
