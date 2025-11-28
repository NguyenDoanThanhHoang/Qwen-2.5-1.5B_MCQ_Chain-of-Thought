"""
Generate realistic rejected samples from SFT model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel

from configs.config import model_config, sft_config, data_config
from src.prepare_data import load_ecqa_dataset, format_prompt
from src.utils import get_bnb_config, load_tokenizer, extract_answer, find_answer_key


def load_sft_model_for_generation():
    """Load SFT model for generating rejected samples"""
    print("Loading SFT model for generation...")

    tokenizer = load_tokenizer(model_config.model_id)
    bnb_config = get_bnb_config()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    sft_checkpoint = sft_config.output_dir
    print(f"Loading SFT adapter from: {sft_checkpoint}")

    model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    model.eval()

    return model, tokenizer


def generate_rejected_sample(model, tokenizer, question, choices, correct_answer):
    """Generate rejected sample from SFT model"""
    prompt = format_prompt(question, choices)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=1.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=3,
        )

    candidates = []
    for output in outputs:
        generated_ids = output[inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = extract_answer(generated_text)

        if predicted_answer and predicted_answer != correct_answer:
            candidates.append(generated_text)

    return candidates[0] if candidates else None


def build_sft_rejected_samples(split="train", sample_size=None, max_retries=3):
    """Build rejected samples from SFT model"""
    print(f"\n{'='*80}")
    print(f"GENERATING REJECTED SAMPLES FROM SFT MODEL - {split.upper()}")
    print(f"{'='*80}\n")

    model, tokenizer = load_sft_model_for_generation()

    print(f"Loading {split} dataset...")
    raw_ds = load_ecqa_dataset(split)

    if sample_size is not None and sample_size < len(raw_ds):
        raw_ds = raw_ds.shuffle(seed=data_config.seed).select(range(sample_size))
        print(f"Using {sample_size} samples")

    rejected_samples = []
    failed_count = 0

    print(f"\nGenerating rejected samples (this may take a while)...")

    for example in tqdm(raw_ds, desc="Generating"):
        question = example['q_text']
        choices = [example[f'q_op{i}'] for i in range(1, 6)]
        answer_text = example['q_ans']
        answer_key = find_answer_key(example['q_ans'], choices)

        if not answer_key:
            continue

        rejected_text = None
        for retry in range(max_retries):
            rejected_text = generate_rejected_sample(
                model, tokenizer, question, choices, answer_key
            )
            if rejected_text:
                break

        if rejected_text:
            rejected_samples.append({
                'question': question,
                'choices': choices,
                'correct_answer': answer_key,
                'rejected_text': rejected_text,
            })
        else:
            failed_count += 1

    print(f"\nGeneration complete!")
    print(f"  Successfully generated: {len(rejected_samples)} samples")
    print(f"  Failed (model always correct): {failed_count} samples")
    print(f"  Success rate: {len(rejected_samples)/(len(rejected_samples)+failed_count)*100:.1f}%")

    return rejected_samples


def save_rejected_samples(rejected_samples, output_file):
    """Save rejected samples to JSONL"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in rejected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(rejected_samples)} rejected samples to: {output_file}")


if __name__ == "__main__":
    print("="*80)
    print("SFT-BASED REJECTED SAMPLE GENERATION")
    print("="*80)
    print("\nThis script generates realistic rejected samples from SFT model.")
    print("It will take ~2-4 hours for 10k samples on T4 GPU.")
    print("\nIMPORTANT: Make sure SFT training is completed first!")
    print("="*80)

    sft_checkpoint = Path(sft_config.output_dir)
    if not sft_checkpoint.exists():
        print("\n❌ ERROR: SFT checkpoint not found!")
        print(f"Expected at: {sft_checkpoint}")
        print("Please train SFT model first before running this script.")
        sys.exit(1)

    print("\n\n" + "="*80)
    print("TRAINING SET")
    print("="*80)

    train_rejected = build_sft_rejected_samples(
        split="train",
        sample_size=data_config.train_sample_size,
    )

    save_rejected_samples(
        train_rejected,
        "data/sft_rejected_train.jsonl"
    )

    print("\n\n" + "="*80)
    print("VALIDATION SET")
    print("="*80)

    val_rejected = build_sft_rejected_samples(
        split="validation",
        sample_size=data_config.val_sample_size,
    )

    save_rejected_samples(
        val_rejected,
        "data/sft_rejected_val.jsonl"
    )

    print("\n" + "="*80)
    print("✓ REJECTED SAMPLE GENERATION COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - data/sft_rejected_train.jsonl")
    print("  - data/sft_rejected_val.jsonl")
    print("\nNext: Update build_dpo_data.py to use these files")
