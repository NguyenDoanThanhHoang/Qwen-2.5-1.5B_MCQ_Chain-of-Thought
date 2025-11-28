"""
Shared utility functions for model loading, data processing, and configuration
"""

import re
import logging
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from configs.config import model_config


def get_bnb_config() -> BitsAndBytesConfig:
    """Create 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )


def get_lora_config() -> LoraConfig:
    """Create LoRA adapter config"""
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(model_path: str, local_files_only: bool = False):
    """Load tokenizer with standardized padding configuration"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    return tokenizer


def extract_answer(text: str) -> str:
    """Extract answer letter (A-E) from generated text"""
    match = re.search(r'Answer:\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def find_answer_key(answer_text: str, choices: list) -> str:
    """Find answer key (A-E) from answer text by matching with choices"""
    for i, choice in enumerate(choices):
        if choice.lower().strip() == answer_text.lower().strip():
            return chr(65 + i)

    logging.warning(
        f"Answer text '{answer_text}' not found in choices. "
        f"Using 'A' as fallback. Choices: {choices}"
    )
    return 'A'


def has_good_explanation(example: dict, min_length: int = 20) -> bool:
    """Check if ECQA example has a good explanation"""
    explanation = example.get('taskB', '').strip()
    return explanation and len(explanation) >= min_length
