"""
Direct Preference Optimization (DPO) Training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import DPOTrainer, DPOConfig
import wandb

from configs.config import model_config, dpo_config, sft_config, data_config
from src.build_dpo_data import load_dpo_pairs
from src.utils import get_bnb_config, get_lora_config, load_tokenizer


def load_sft_model_and_tokenizer():
    """Load SFT checkpoint for DPO training"""
    checkpoint_dir = sft_config.output_dir
    print(f"Loading base model with SFT adapter from: {checkpoint_dir}")

    from pathlib import Path
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found at: {checkpoint_dir}\n"
            f"Please make sure SFT training completed successfully."
        )

    tokenizer = load_tokenizer(model_config.model_id)
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        dtype=model_config.bnb_4bit_compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model)
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dpo_dataset(dpo_pairs: list) -> Dataset:
    """Convert DPO pairs to HuggingFace Dataset"""
    prompts = [pair['prompt'] for pair in dpo_pairs]
    chosen = [pair['chosen'] for pair in dpo_pairs]
    rejected = [pair['rejected'] for pair in dpo_pairs]

    dataset = Dataset.from_dict({
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected,
    })

    return dataset


def get_training_args():
    """Create DPO training arguments"""
    return DPOConfig(
        output_dir=dpo_config.output_dir,
        num_train_epochs=dpo_config.num_train_epochs,
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        learning_rate=dpo_config.learning_rate,
        max_grad_norm=dpo_config.max_grad_norm,
        warmup_ratio=dpo_config.warmup_ratio,
        lr_scheduler_type=dpo_config.lr_scheduler_type,
        optim=dpo_config.optim,
        fp16=dpo_config.fp16,
        logging_steps=dpo_config.logging_steps,
        save_strategy=dpo_config.save_strategy,
        save_total_limit=dpo_config.save_total_limit,
        report_to=dpo_config.report_to,
        run_name=dpo_config.run_name,
        push_to_hub=False,
        remove_unused_columns=False,
        beta=dpo_config.beta,
        max_length=dpo_config.max_length,
        max_prompt_length=dpo_config.max_prompt_length,
    )


def main():
    """Main DPO training pipeline"""
    print("="*80)
    print("DIRECT PREFERENCE OPTIMIZATION (DPO) - Qwen 2.5 1.5B")
    print("="*80)

    print("\nInitializing wandb...")
    wandb.init(
        project="qwen25-mcq-cot",
        name=dpo_config.run_name,
        config={
            "model": model_config.model_id,
            "task": "DPO",
            "dataset": data_config.dataset_name,
            "lora_r": model_config.lora_r,
            "lora_alpha": model_config.lora_alpha,
            "learning_rate": dpo_config.learning_rate,
            "batch_size": dpo_config.per_device_train_batch_size,
            "gradient_accumulation_steps": dpo_config.gradient_accumulation_steps,
            "beta": dpo_config.beta,
        }
    )

    print("\nLoading DPO preference pairs...")
    dpo_train_pairs = load_dpo_pairs(data_config.dpo_pairs_file)
    dpo_val_pairs = load_dpo_pairs(data_config.dpo_val_pairs_file)

    train_dataset = prepare_dpo_dataset(dpo_train_pairs)
    val_dataset = prepare_dpo_dataset(dpo_val_pairs)

    print(f"DPO training samples: {len(train_dataset)}")
    print(f"DPO validation samples: {len(val_dataset)}")

    print("\nLoading SFT checkpoint...")
    model, tokenizer = load_sft_model_and_tokenizer()

    print("\nLoading reference model...")
    ref_model, _ = load_sft_model_and_tokenizer()

    training_args = get_training_args()

    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting DPO training...")
    print(f"Effective batch size: {dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps) * dpo_config.num_train_epochs}")

    trainer.train()

    print("\nSaving DPO model...")
    trainer.save_model(dpo_config.output_dir)
    tokenizer.save_pretrained(dpo_config.output_dir)

    print(f"\nDPO model saved to: {dpo_config.output_dir}")

    print("\nMerging and saving full model...")
    merged_model = model.merge_and_unload()
    merged_output_dir = f"{dpo_config.output_dir}-merged"
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    print(f"Merged DPO model saved to: {merged_output_dir}")

    wandb.finish()
    print("\nDPO training completed! ✓")


if __name__ == "__main__":
    main()
