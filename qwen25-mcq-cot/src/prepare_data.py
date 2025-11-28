"""
Data preparation for ECQA (Commonsense QA) dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset, Dataset
from typing import Dict, List
from configs.config import data_config
from src.utils import find_answer_key, has_good_explanation


def load_ecqa_dataset(split: str = "train"):
    """Load Commonsense QA dataset from HuggingFace"""
    print(f"Loading {split} split from {data_config.dataset_name}...")
    ds = load_dataset(data_config.dataset_name, split=split)
    print(f"Loaded {len(ds)} samples")
    return ds


def format_prompt(question: str, choices: List[str]) -> str:
    """Format multiple-choice question into prompt"""
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

    prompt = f"""Answer the following question with step-by-step reasoning.

Question: {question}
Options:
{choices_text}

Think through this step by step, then provide your answer as "Answer: X"."""

    return prompt


def convert_to_sft_format(example: Dict) -> Dict:
    """Convert ECQA example to SFT training format"""
    question = example['q_text']
    choices = [example[f'q_op{i}'] for i in range(1, 6)]

    answer_text = example['q_ans']
    answer_key = find_answer_key(answer_text, choices)

    explanation = example.get('taskB', '').strip()

    if not explanation:
        explanation = "Let me think step by step about this question."

    prompt = format_prompt(question, choices)
    completion = f"{explanation}\nAnswer: {answer_key}"
    text = prompt + "\n" + completion

    return {
        "text": text,
        "prompt": prompt,
        "completion": completion,
        "answer_key": answer_key,
        "question": question,
        "choices": choices
    }


def prepare_sft_dataset(split: str = "train", sample_size: int = None, filter_bad_explanations: bool = True) -> Dataset:
    """Prepare full SFT dataset from ECQA"""
    raw_ds = load_ecqa_dataset(split)

    if filter_bad_explanations:
        print("Filtering samples without good explanations...")
        original_size = len(raw_ds)

        raw_ds = raw_ds.filter(has_good_explanation, desc="Filtering bad explanations")
        filtered_count = original_size - len(raw_ds)
        print(f"Filtered out {filtered_count} samples ({filtered_count/original_size*100:.1f}%)")
        print(f"Remaining: {len(raw_ds)} samples with good explanations")

    if sample_size is not None and sample_size < len(raw_ds):
        raw_ds = raw_ds.shuffle(seed=data_config.seed).select(range(sample_size))
        print(f"Sampled {sample_size} examples")

    print("Converting to SFT format...")
    sft_ds = raw_ds.map(
        convert_to_sft_format,
        remove_columns=raw_ds.column_names,
        desc="Formatting examples"
    )

    return sft_ds


def validate_dataset(ds: Dataset, num_samples: int = 3):
    """Print sample examples to validate formatting"""
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
    print("Testing ECQA data preparation...\n")

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

    validate_dataset(train_ds, num_samples=2)

    train_examples = [ex for ex in train_ds]
    no_explanation = sum(1 for ex in train_examples if "Let me think step by step" in ex['completion'])

    print(f"\nDataset Statistics:")
    print(f"Total train samples: {len(train_ds)}")
    print(f"Samples without explanation: {no_explanation}")
    print(f"Samples with explanation: {len(train_ds) - no_explanation}")
    print(f"Coverage: {((len(train_ds) - no_explanation) / len(train_ds) * 100):.1f}%")
