"""
Centralized configuration for Llama 3.2 MCQ CoT project
Optimized for Google Colab Free (T4 GPU, 12GB RAM)
"""

import torch
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model and quantization settings"""
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"

    # 4-bit quantization config
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True

    # LoRA config
    lora_r: int = 16  # Reduced from 64 for Colab Free
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class SFTConfig:
    """SFT training hyperparameters"""
    output_dir: str = "outputs/sft-llama32-1b-mcq"

    # Training params (Colab Free optimized)
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Effective batch size = 16
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Optimization
    optim: str = "paged_adamw_8bit"
    fp16: bool = True

    # Logging and saving
    logging_steps: int = 25
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    # Wandb
    report_to: str = "wandb"
    run_name: str = "llama32-1b-sft-ecqa"

    # Data
    max_seq_length: int = 512
    dataset_text_field: str = "text"


@dataclass
class DPOConfig:
    """DPO training hyperparameters"""
    output_dir: str = "outputs/dpo-llama32-1b-mcq"

    # Training params (smaller than SFT)
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    learning_rate: float = 5e-5  # Lower than SFT
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # Optimization
    optim: str = "paged_adamw_8bit"
    fp16: bool = True

    # Logging and saving
    logging_steps: int = 25
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    # Wandb
    report_to: str = "wandb"
    run_name: str = "llama32-1b-dpo-ecqa"

    # DPO specific
    beta: float = 0.1  # DPO temperature
    max_length: int = 512
    max_prompt_length: int = 384


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "tau/commonsense_qa"
    data_dir: str = "data"
    dpo_pairs_file: str = "data/dpo_pairs.jsonl"

    # Sampling (set to None for full dataset)
    train_sample_size: int = None  # Use full 10k
    val_sample_size: int = None

    # Random seed
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation settings"""
    batch_size: int = 1
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Number of samples to evaluate (None = all)
    eval_sample_size: int = None


# Create global config instances
model_config = ModelConfig()
sft_config = SFTConfig()
dpo_config = DPOConfig()
data_config = DataConfig()
eval_config = EvalConfig()
