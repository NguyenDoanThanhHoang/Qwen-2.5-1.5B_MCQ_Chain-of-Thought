"""
Data preparation for ECQA (Commonsense QA) dataset
Loads dataset from HuggingFace and formats for SFT training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset, Dataset
from typing import Dict, List
from configs.config import data_config


def load_ecqa_dataset(split: str = "train"):
    """
    Load Commonsense QA dataset from HuggingFace

    Args:
        split: 'train', 'validation', or 'test'

    Returns:
        Dataset object
    """
    print(f"Loading {split} split from {data_config.dataset_name}...")
    ds = load_dataset(data_config.dataset_name, split=split)
    print(f"Loaded {len(ds)} samples")
    return ds


def format_prompt(question: str, choices: List[str]) -> str:
    """
    Format multiple-choice question into prompt

    Args:
        question: Question text
        choices: List of answer choices

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

    prompt = f"""Answer the following question with step-by-step reasoning.

Question: {question}
Options:
{choices_text}

Think through this step by step, then provide your answer as "Answer: X"."""

    return prompt


def convert_to_sft_format(example: Dict) -> Dict:
    """
    Convert ECQA example to SFT training format

    Args:
        example: Single example from ECQA dataset

    Returns:
        Dictionary with formatted text, prompt, completion, and answer key
    """
    question = example['question']
    choices = example['choices']['text']
    answer_key = example['answerKey']

    # Extract explanation if available
    explanation = example.get('explanation', None)

    # If no explanation, use simple reasoning
    if not explanation or explanation.strip() == "":
        explanation = "Let me think step by step about this question."

    # Format prompt
    prompt = format_prompt(question, choices)

    # Format completion (explanation + answer)
    completion = f"{explanation}\nAnswer: {answer_key}"

    # Combined text for SFT
    text = prompt + "\n" + completion

    return {
        "text": text,
        "prompt": prompt,
        "completion": completion,
        "answer_key": answer_key,
        "question": question,
        "choices": choices
    }


def prepare_sft_dataset(split: str = "train", sample_size: int = None) -> Dataset:
    """
    Prepare full SFT dataset from ECQA

    Args:
        split: Dataset split to prepare
        sample_size: Number of samples to take (None = all)

    Returns:
        Formatted Dataset ready for SFT training
    """
    # Load raw dataset
    raw_ds = load_ecqa_dataset(split)

    # Sample if requested
    if sample_size is not None and sample_size < len(raw_ds):
        raw_ds = raw_ds.shuffle(seed=data_config.seed).select(range(sample_size))
        print(f"Sampled {sample_size} examples")

    # Convert to SFT format
    print("Converting to SFT format...")
    sft_ds = raw_ds.map(
        convert_to_sft_format,
        remove_columns=raw_ds.column_names,
        desc="Formatting examples"
    )

    return sft_ds


def validate_dataset(ds: Dataset, num_samples: int = 3):
    """
    Print sample examples to validate formatting

    Args:
        ds: Dataset to validate
        num_samples: Number of samples to print
    """
    print(f"\n{'='*80}")
    print(f"DATASET VALIDATION - Showing {num_samples} samples")
    print(f"{'='*80}\n")

    for i in range(min(num_samples, len(ds))):
        example = ds[i]
        print(f"--- Sample {i+1} ---")
        print(f"Answer Key: {example['answer_key']}")
        print(f"\nFull Text:\n{example['text'][:500]}...")
        print(f"\n{'-'*80}\n")


if __name__ == "__main__":
    """Test data preparation pipeline"""

    print("Testing ECQA data preparation...\n")

    # Prepare train and validation datasets
    train_ds = prepare_sft_dataset(
        split="train",
        sample_size=data_config.train_sample_size
    )

    val_ds = prepare_sft_dataset(
        split="validation",
        sample_size=data_config.val_sample_size
    )

    print(f"\nTrain dataset: {len(train_ds)} samples")
    print(f"Validation dataset: {len(val_ds)} samples")

    # Validate formatting
    validate_dataset(train_ds, num_samples=2)

    # Check for missing explanations
    train_examples = [ex for ex in train_ds]
    no_explanation = sum(1 for ex in train_examples if "Let me think step by step" in ex['completion'])

    print(f"\nDataset Statistics:")
    print(f"Total train samples: {len(train_ds)}")
    print(f"Samples without explanation: {no_explanation}")
    print(f"Samples with explanation: {len(train_ds) - no_explanation}")
    print(f"Coverage: {((len(train_ds) - no_explanation) / len(train_ds) * 100):.1f}%")
