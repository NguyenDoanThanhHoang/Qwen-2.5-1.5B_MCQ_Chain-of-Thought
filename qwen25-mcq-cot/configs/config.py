"""
Centralized configuration for Qwen 2.5 MCQ CoT project
Optimized for Google Colab Free (T4 GPU, 12GB RAM)
"""

import torch
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model and quantization settings"""
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class SFTConfig:
    """SFT training hyperparameters"""
    output_dir: str = "outputs/sft-qwen25-1.5b-mcq"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 10
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "wandb"
    run_name: str = "qwen25-1.5b-sft-ecqa"
    max_seq_length: int = 512
    dataset_text_field: str = "text"


@dataclass
class DPOConfig:
    """DPO training hyperparameters"""
    output_dir: str = "outputs/dpo-qwen25-1.5b-mcq"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 5
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "wandb"
    run_name: str = "qwen25-1.5b-dpo-ecqa"
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 384


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "tasksource/ecqa"
    data_dir: str = "data"
    dpo_pairs_file: str = "data/dpo_pairs.jsonl"
    dpo_val_pairs_file: str = "data/dpo_val_pairs.jsonl"
    train_sample_size: int = None
    val_sample_size: int = 500
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation settings"""
    batch_size: int = 1
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    eval_sample_size: int = None


model_config = ModelConfig()
sft_config = SFTConfig()
dpo_config = DPOConfig()
data_config = DataConfig()
eval_config = EvalConfig()
