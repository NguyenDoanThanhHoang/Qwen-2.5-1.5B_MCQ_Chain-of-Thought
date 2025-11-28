"""
Supervised Fine-Tuning (SFT) with QLoRA
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import wandb

from configs.config import model_config, sft_config, data_config
from src.prepare_data import prepare_sft_dataset
from src.utils import get_bnb_config, get_lora_config, load_tokenizer


def load_model_and_tokenizer():
    """Load Qwen 2.5 1.5B model with 4-bit quantization"""
    print(f"Loading model: {model_config.model_id}")

    tokenizer = load_tokenizer(model_config.model_id)
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        dtype=model_config.bnb_4bit_compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def get_training_args():
    """Create training arguments"""
    return TrainingArguments(
        output_dir=sft_config.output_dir,
        num_train_epochs=sft_config.num_train_epochs,
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        learning_rate=sft_config.learning_rate,
        max_grad_norm=sft_config.max_grad_norm,
        warmup_ratio=sft_config.warmup_ratio,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        optim=sft_config.optim,
        fp16=sft_config.fp16,
        logging_steps=sft_config.logging_steps,
        eval_strategy=sft_config.eval_strategy,
        eval_steps=sft_config.eval_steps,
        save_strategy="steps",
        save_steps=sft_config.eval_steps,
        save_total_limit=sft_config.save_total_limit,
        load_best_model_at_end=sft_config.load_best_model_at_end,
        metric_for_best_model=sft_config.metric_for_best_model,
        report_to=sft_config.report_to,
        run_name=sft_config.run_name,
        push_to_hub=False,
        remove_unused_columns=True,
        group_by_length=True,
    )


def main():
    """Main SFT training pipeline"""
    print("="*80)
    print("SUPERVISED FINE-TUNING (SFT) - Qwen 2.5 1.5B")
    print("="*80)

    print("\nInitializing wandb...")
    wandb.init(
        project="qwen25-mcq-cot",
        name=sft_config.run_name,
        config={
            "model": model_config.model_id,
            "task": "SFT",
            "dataset": data_config.dataset_name,
            "lora_r": model_config.lora_r,
            "lora_alpha": model_config.lora_alpha,
            "learning_rate": sft_config.learning_rate,
            "batch_size": sft_config.per_device_train_batch_size,
            "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
        }
    )

    print("\nLoading datasets...")
    train_dataset = prepare_sft_dataset(
        split="train",
        sample_size=data_config.train_sample_size
    )
    val_dataset = prepare_sft_dataset(
        split="validation",
        sample_size=data_config.val_sample_size
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    training_args = get_training_args()

    print("\nInitializing SFT Trainer...")

    def formatting_func(example):
        return example[sft_config.dataset_text_field]

    from trl import SFTConfig

    sft_trainer_config = SFTConfig(
        **training_args.to_dict(),
        dataset_text_field=None,
        completion_only_loss=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_trainer_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("\nStarting training...")
    print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps) * sft_config.num_train_epochs}")

    trainer.train()

    print("\nSaving model...")
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)

    print(f"\nModel saved to: {sft_config.output_dir}")

    print("\nMerging and saving full model...")
    merged_model = model.merge_and_unload()
    merged_output_dir = f"{sft_config.output_dir}-merged"
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    print(f"Merged model saved to: {merged_output_dir}")

    wandb.finish()
    print("\nSFT training completed! ✓")


if __name__ == "__main__":
    main()
