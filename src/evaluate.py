"""
Evaluation script for MCQ models
Compare base model, SFT model, and DPO model on ECQA validation set
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import re
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from configs.config import model_config, sft_config, dpo_config, data_config, eval_config
from src.prepare_data import prepare_sft_dataset, format_prompt


def extract_answer(generated_text: str) -> str:
    """
    Extract answer letter from generated text

    Args:
        generated_text: Full generated response

    Returns:
        Answer letter (A-E) or None if not found
    """
    # Pattern 1: "Answer: X"
    match = re.search(r'Answer:\s*([A-E])', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: Last letter in text
    match = re.search(r'\b([A-E])\b(?!.*\b[A-E]\b)', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def load_model_for_eval(model_path: str, use_4bit: bool = True):
    """
    Load model for evaluation

    Args:
        model_path: Path to model checkpoint
        use_4bit: Whether to use 4-bit quantization

    Returns:
        model, tokenizer
    """
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    sample_size: int = None
) -> Dict:
    """
    Evaluate model on dataset

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Dataset with questions
        sample_size: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with evaluation results
    """
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))

    correct = 0
    total = 0
    no_answer = 0
    predictions = []

    print(f"Evaluating on {len(dataset)} samples...")

    for example in tqdm(dataset):
        prompt = example['prompt']
        true_answer = example['answer_key']

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=eval_config.max_new_tokens,
                temperature=eval_config.temperature,
                top_p=eval_config.top_p,
                do_sample=eval_config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        predicted_answer = extract_answer(generated_text)

        # Check correctness
        if predicted_answer is None:
            no_answer += 1
        elif predicted_answer == true_answer:
            correct += 1

        total += 1

        predictions.append({
            'question': example.get('question', ''),
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'generated_text': generated_text,
        })

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    no_answer_rate = no_answer / total if total > 0 else 0

    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'no_answer': no_answer,
        'no_answer_rate': no_answer_rate,
        'predictions': predictions,
    }

    return results


def print_results(model_name: str, results: Dict):
    """Print evaluation results"""
    print(f"\n{'='*80}")
    print(f"Results for: {model_name}")
    print(f"{'='*80}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"No answer: {results['no_answer']} ({results['no_answer_rate']*100:.1f}%)")
    print(f"{'='*80}\n")


def print_sample_outputs(predictions: List[Dict], num_samples: int = 3):
    """Print sample predictions"""
    print(f"\n{'='*80}")
    print(f"Sample Predictions (showing {num_samples})")
    print(f"{'='*80}\n")

    for i, pred in enumerate(predictions[:num_samples]):
        print(f"--- Sample {i+1} ---")
        print(f"Question: {pred['question'][:200]}...")
        print(f"True Answer: {pred['true_answer']}")
        print(f"Predicted: {pred['predicted_answer']}")
        print(f"Generated:\n{pred['generated_text'][-300:]}")
        print(f"\n{'-'*80}\n")


def main():
    """Main evaluation pipeline"""

    print("="*80)
    print("MODEL EVALUATION - Llama 3.2 1B on ECQA")
    print("="*80)

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = prepare_sft_dataset(
        split="validation",
        sample_size=eval_config.eval_sample_size
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Store all results
    all_results = {}

    # Evaluate base model
    print("\n" + "="*80)
    print("1. Evaluating BASE MODEL")
    print("="*80)

    base_model, base_tokenizer = load_model_for_eval(model_config.model_id)
    base_results = evaluate_model(base_model, base_tokenizer, val_dataset)
    all_results['base'] = base_results
    print_results("Base Llama-3.2-1B-Instruct", base_results)

    # Clear memory
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # Evaluate SFT model
    print("\n" + "="*80)
    print("2. Evaluating SFT MODEL")
    print("="*80)

    sft_model_path = f"{sft_config.output_dir}-merged"
    sft_model, sft_tokenizer = load_model_for_eval(sft_model_path)
    sft_results = evaluate_model(sft_model, sft_tokenizer, val_dataset)
    all_results['sft'] = sft_results
    print_results("SFT Model", sft_results)

    # Clear memory
    del sft_model, sft_tokenizer
    torch.cuda.empty_cache()

    # Evaluate DPO model
    print("\n" + "="*80)
    print("3. Evaluating DPO MODEL")
    print("="*80)

    dpo_model_path = f"{dpo_config.output_dir}-merged"
    dpo_model, dpo_tokenizer = load_model_for_eval(dpo_model_path)
    dpo_results = evaluate_model(dpo_model, dpo_tokenizer, val_dataset)
    all_results['dpo'] = dpo_results
    print_results("DPO Model", dpo_results)

    # Print comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<15} {'Correct':<15} {'No Answer':<15}")
    print("-"*80)
    print(f"{'Base Model':<20} {all_results['base']['accuracy']*100:>6.2f}%        {all_results['base']['correct']:>3}/{all_results['base']['total']:<10} {all_results['base']['no_answer']:>3} ({all_results['base']['no_answer_rate']*100:.1f}%)")
    print(f"{'SFT Model':<20} {all_results['sft']['accuracy']*100:>6.2f}%        {all_results['sft']['correct']:>3}/{all_results['sft']['total']:<10} {all_results['sft']['no_answer']:>3} ({all_results['sft']['no_answer_rate']*100:.1f}%)")
    print(f"{'DPO Model':<20} {all_results['dpo']['accuracy']*100:>6.2f}%        {all_results['dpo']['correct']:>3}/{all_results['dpo']['total']:<10} {all_results['dpo']['no_answer']:>3} ({all_results['dpo']['no_answer_rate']*100:.1f}%)")
    print("="*80)

    # Print improvements
    sft_improvement = (all_results['sft']['accuracy'] - all_results['base']['accuracy']) * 100
    dpo_improvement = (all_results['dpo']['accuracy'] - all_results['sft']['accuracy']) * 100

    print(f"\nImprovements:")
    print(f"  SFT vs Base: {sft_improvement:+.2f}%")
    print(f"  DPO vs SFT:  {dpo_improvement:+.2f}%")

    # Print sample outputs from DPO model
    print_sample_outputs(dpo_results['predictions'], num_samples=3)

    print("\nEvaluation completed! ✓")


if __name__ == "__main__":
    main()
