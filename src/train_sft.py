"""
Supervised Fine-Tuning (SFT) with QLoRA
Train Llama 3.2 1B on ECQA dataset with Chain-of-Thought reasoning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import wandb

from configs.config import model_config, sft_config, data_config
from src.prepare_data import prepare_sft_dataset


def get_bnb_config():
    """Create 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )


def get_lora_config():
    """Create LoRA adapter config"""
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer():
    """
    Load Llama 3.2 1B model with 4-bit quantization

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_config.model_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # Important for causal LM

    # Load model with quantization
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        torch_dtype=model_config.bnb_4bit_compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
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
        save_strategy=sft_config.save_strategy,
        save_total_limit=sft_config.save_total_limit,
        report_to=sft_config.report_to,
        run_name=sft_config.run_name,
        push_to_hub=False,
        remove_unused_columns=True,
        group_by_length=True,  # Optimize training speed
    )


def main():
    """Main SFT training pipeline"""

    print("="*80)
    print("SUPERVISED FINE-TUNING (SFT) - Llama 3.2 1B")
    print("="*80)

    # Initialize wandb
    print("\nInitializing wandb...")
    wandb.init(
        project="llama32-mcq-cot",
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

    # Load datasets
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

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Create training arguments
    training_args = get_training_args()

    # Create SFT trainer
    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field=sft_config.dataset_text_field,
        max_seq_length=sft_config.max_seq_length,
        packing=False,  # Don't pack sequences
    )

    # Start training
    print("\nStarting training...")
    print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps) * sft_config.num_train_epochs}")

    trainer.train()

    # Save model
    print("\nSaving model...")
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)

    print(f"\nModel saved to: {sft_config.output_dir}")

    # Save merged model (optional, for easier loading later)
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
