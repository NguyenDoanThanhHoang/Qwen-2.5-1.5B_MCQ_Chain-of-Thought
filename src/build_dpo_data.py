"""
Build DPO preference pairs from ECQA dataset
Create (prompt, chosen, rejected) triplets for Direct Preference Optimization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import random
from typing import Dict, List
from datasets import load_dataset

from configs.config import data_config
from src.prepare_data import load_ecqa_dataset, format_prompt


def create_dpo_pair(example: Dict) -> Dict:
    """
    Create a single DPO preference pair

    Args:
        example: Single example from ECQA dataset

    Returns:
        Dictionary with prompt, chosen, rejected
    """
    question = example['question']
    choices = example['choices']['text']
    answer_key = example['answerKey']

    # Get explanation
    explanation = example.get('explanation', None)
    if not explanation or explanation.strip() == "":
        explanation = "Let me think step by step about this question."

    # Format prompt (same as SFT)
    prompt = format_prompt(question, choices)

    # CHOSEN: Full explanation + correct answer
    chosen = f"{explanation}\nAnswer: {answer_key}"

    # REJECTED: Create wrong answer with minimal/wrong reasoning
    # Strategy: Random wrong answer with shorter explanation

    # Get list of possible answers
    num_choices = len(choices)
    all_answers = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D, E

    # Choose random wrong answer
    wrong_answers = [ans for ans in all_answers if ans != answer_key]
    wrong_answer = random.choice(wrong_answers)

    # Create rejected response with minimal reasoning
    rejected_templates = [
        f"I think the answer is {wrong_answer}.\nAnswer: {wrong_answer}",
        f"After considering the options, I believe {wrong_answer} is correct.\nAnswer: {wrong_answer}",
        f"The most reasonable choice appears to be {wrong_answer}.\nAnswer: {wrong_answer}",
        f"Looking at the options, {wrong_answer} seems right.\nAnswer: {wrong_answer}",
    ]

    rejected = random.choice(rejected_templates)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "answer_key": answer_key,
        "wrong_answer": wrong_answer,
    }


def build_dpo_dataset(split: str = "train", sample_size: int = None) -> List[Dict]:
    """
    Build full DPO dataset from ECQA

    Args:
        split: Dataset split to use
        sample_size: Number of samples (None = all)

    Returns:
        List of DPO preference pairs
    """
    # Load raw dataset
    print(f"Loading {split} split for DPO...")
    raw_ds = load_ecqa_dataset(split)

    # Sample if requested
    if sample_size is not None and sample_size < len(raw_ds):
        raw_ds = raw_ds.shuffle(seed=data_config.seed).select(range(sample_size))
        print(f"Sampled {sample_size} examples")

    # Set random seed for consistent rejected answers
    random.seed(data_config.seed)

    # Convert to DPO pairs
    print("Creating DPO preference pairs...")
    dpo_pairs = []

    for example in raw_ds:
        pair = create_dpo_pair(example)
        dpo_pairs.append(pair)

    return dpo_pairs


def save_dpo_pairs(dpo_pairs: List[Dict], output_file: str = None):
    """
    Save DPO pairs to JSONL file

    Args:
        dpo_pairs: List of preference pairs
        output_file: Output file path
    """
    if output_file is None:
        output_file = data_config.dpo_pairs_file

    # Create directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving DPO pairs to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Saved {len(dpo_pairs)} DPO pairs ✓")


def validate_dpo_pairs(dpo_pairs: List[Dict], num_samples: int = 2):
    """
    Print sample DPO pairs to validate formatting

    Args:
        dpo_pairs: List of preference pairs
        num_samples: Number of samples to print
    """
    print(f"\n{'='*80}")
    print(f"DPO PAIRS VALIDATION - Showing {num_samples} samples")
    print(f"{'='*80}\n")

    for i in range(min(num_samples, len(dpo_pairs))):
        pair = dpo_pairs[i]

        print(f"--- Sample {i+1} ---")
        print(f"Correct Answer: {pair['answer_key']}")
        print(f"Wrong Answer: {pair['wrong_answer']}")

        print(f"\nPROMPT:\n{pair['prompt'][:300]}...")

        print(f"\nCHOSEN (correct):\n{pair['chosen'][:200]}...")

        print(f"\nREJECTED (wrong):\n{pair['rejected'][:200]}...")

        print(f"\n{'-'*80}\n")


def load_dpo_pairs(file_path: str = None) -> List[Dict]:
    """
    Load DPO pairs from JSONL file

    Args:
        file_path: Path to JSONL file

    Returns:
        List of DPO pairs
    """
    if file_path is None:
        file_path = data_config.dpo_pairs_file

    print(f"Loading DPO pairs from {file_path}...")

    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} DPO pairs")
    return pairs


if __name__ == "__main__":
    """Build and save DPO dataset"""

    print("Building DPO preference pairs from ECQA...\n")

    # Build DPO pairs from training set
    dpo_pairs = build_dpo_dataset(
        split="train",
        sample_size=data_config.train_sample_size
    )

    print(f"\nGenerated {len(dpo_pairs)} DPO preference pairs")

    # Validate formatting
    validate_dpo_pairs(dpo_pairs, num_samples=2)

    # Save to file
    save_dpo_pairs(dpo_pairs)

    # Verify saved pairs
    print("\nVerifying saved file...")
    loaded_pairs = load_dpo_pairs()

    print(f"\nValidation:")
    print(f"  Generated: {len(dpo_pairs)} pairs")
    print(f"  Saved and loaded: {len(loaded_pairs)} pairs")
    print(f"  Match: {'✓' if len(dpo_pairs) == len(loaded_pairs) else '✗'}")

    print("\nDPO data preparation completed! ✓")
