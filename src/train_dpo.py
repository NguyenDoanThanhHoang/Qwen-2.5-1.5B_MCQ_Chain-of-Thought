"""
Direct Preference Optimization (DPO) Training
Train on preference pairs to improve CoT quality and answer accuracy
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import DPOTrainer
import wandb

from configs.config import model_config, dpo_config, sft_config, data_config
from src.build_dpo_data import load_dpo_pairs


def get_bnb_config():
    """Create 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )


def get_lora_config():
    """Create LoRA adapter config for DPO"""
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_sft_model_and_tokenizer(use_merged: bool = True):
    """
    Load SFT checkpoint for DPO training

    Args:
        use_merged: Whether to use merged model or adapter model

    Returns:
        model, tokenizer
    """
    # Determine which checkpoint to load
    if use_merged:
        checkpoint_dir = f"{sft_config.output_dir}-merged"
        print(f"Loading merged SFT model from: {checkpoint_dir}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = "right"

        # Load merged model with quantization
        bnb_config = get_bnb_config()

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            quantization_config=bnb_config,
            torch_dtype=model_config.bnb_4bit_compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        # Load base model + adapter (alternative method)
        checkpoint_dir = sft_config.output_dir
        print(f"Loading base model with SFT adapter from: {checkpoint_dir}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = "right"

        # Load base model with quantization
        bnb_config = get_bnb_config()

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            quantization_config=bnb_config,
            torch_dtype=model_config.bnb_4bit_compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load SFT adapter
        model = PeftModel.from_pretrained(model, checkpoint_dir)

        # Merge for DPO
        model = model.merge_and_unload()

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add new LoRA adapters for DPO
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dpo_dataset(dpo_pairs: list) -> Dataset:
    """
    Convert DPO pairs to HuggingFace Dataset

    Args:
        dpo_pairs: List of DPO preference pairs

    Returns:
        Dataset with prompt, chosen, rejected columns
    """
    # Extract columns
    prompts = [pair['prompt'] for pair in dpo_pairs]
    chosen = [pair['chosen'] for pair in dpo_pairs]
    rejected = [pair['rejected'] for pair in dpo_pairs]

    # Create dataset
    dataset = Dataset.from_dict({
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected,
    })

    return dataset


def get_training_args():
    """Create DPO training arguments"""
    return TrainingArguments(
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
        remove_unused_columns=False,  # Important for DPO
    )


def main():
    """Main DPO training pipeline"""

    print("="*80)
    print("DIRECT PREFERENCE OPTIMIZATION (DPO) - Llama 3.2 1B")
    print("="*80)

    # Initialize wandb
    print("\nInitializing wandb...")
    wandb.init(
        project="llama32-mcq-cot",
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

    # Load DPO pairs
    print("\nLoading DPO preference pairs...")
    dpo_pairs = load_dpo_pairs()

    # Convert to dataset
    train_dataset = prepare_dpo_dataset(dpo_pairs)
    print(f"DPO training samples: {len(train_dataset)}")

    # Load SFT model and tokenizer
    print("\nLoading SFT checkpoint...")
    model, tokenizer = load_sft_model_and_tokenizer(use_merged=True)

    # Create reference model (copy of SFT model)
    print("\nLoading reference model...")
    ref_model, _ = load_sft_model_and_tokenizer(use_merged=True)

    # Create training arguments
    training_args = get_training_args()

    # Create DPO trainer
    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=dpo_config.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=dpo_config.max_length,
        max_prompt_length=dpo_config.max_prompt_length,
    )

    # Start training
    print("\nStarting DPO training...")
    print(f"Effective batch size: {dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps) * dpo_config.num_train_epochs}")

    trainer.train()

    # Save model
    print("\nSaving DPO model...")
    trainer.save_model(dpo_config.output_dir)
    tokenizer.save_pretrained(dpo_config.output_dir)

    print(f"\nDPO model saved to: {dpo_config.output_dir}")

    # Save merged model
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
