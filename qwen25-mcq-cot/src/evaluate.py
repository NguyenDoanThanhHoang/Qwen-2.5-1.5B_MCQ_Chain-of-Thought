"""
Evaluation script for MCQ models
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import re
import torch
from tqdm import tqdm
from typing import Dict, List
from transformers import AutoModelForCausalLM
from datasets import Dataset

from configs.config import model_config, sft_config, dpo_config, data_config, eval_config
from src.prepare_data import prepare_sft_dataset, format_prompt
from src.utils import extract_answer, get_bnb_config, load_tokenizer


def load_model_for_eval(model_path: str, use_4bit: bool = True):
    """Load model for evaluation"""
    print(f"Loading model from: {model_path}")
    tokenizer = load_tokenizer(model_path, local_files_only=True)

    if use_4bit:
        bnb_config = get_bnb_config()

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
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
    """Evaluate model on dataset"""
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

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_answer(generated_text)

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
    print("MODEL EVALUATION - Qwen 2.5 1.5B on ECQA")
    print("="*80)

    print("\nLoading validation dataset...")
    val_dataset = prepare_sft_dataset(
        split="validation",
        sample_size=eval_config.eval_sample_size
    )
    print(f"Validation samples: {len(val_dataset)}")

    all_results = {}

    print("\n" + "="*80)
    print("1. Evaluating BASE MODEL")
    print("="*80)

    base_model, base_tokenizer = load_model_for_eval(model_config.model_id)
    base_results = evaluate_model(base_model, base_tokenizer, val_dataset)
    all_results['base'] = base_results
    print_results("Base Qwen2.5-1.5B-Instruct", base_results)

    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("2. Evaluating SFT MODEL")
    print("="*80)

    sft_model_path = f"{sft_config.output_dir}-merged"
    sft_model, sft_tokenizer = load_model_for_eval(sft_model_path)
    sft_results = evaluate_model(sft_model, sft_tokenizer, val_dataset)
    all_results['sft'] = sft_results
    print_results("SFT Model", sft_results)

    del sft_model, sft_tokenizer
    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("3. Evaluating DPO MODEL")
    print("="*80)

    dpo_model_path = f"{dpo_config.output_dir}-merged"
    dpo_model, dpo_tokenizer = load_model_for_eval(dpo_model_path)
    dpo_results = evaluate_model(dpo_model, dpo_tokenizer, val_dataset)
    all_results['dpo'] = dpo_results
    print_results("DPO Model", dpo_results)

    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<15} {'Correct':<15} {'No Answer':<15}")
    print("-"*80)
    print(f"{'Base Model':<20} {all_results['base']['accuracy']*100:>6.2f}%        {all_results['base']['correct']:>3}/{all_results['base']['total']:<10} {all_results['base']['no_answer']:>3} ({all_results['base']['no_answer_rate']*100:.1f}%)")
    print(f"{'SFT Model':<20} {all_results['sft']['accuracy']*100:>6.2f}%        {all_results['sft']['correct']:>3}/{all_results['sft']['total']:<10} {all_results['sft']['no_answer']:>3} ({all_results['sft']['no_answer_rate']*100:.1f}%)")
    print(f"{'DPO Model':<20} {all_results['dpo']['accuracy']*100:>6.2f}%        {all_results['dpo']['correct']:>3}/{all_results['dpo']['total']:<10} {all_results['dpo']['no_answer']:>3} ({all_results['dpo']['no_answer_rate']*100:.1f}%)")
    print("="*80)

    sft_improvement = (all_results['sft']['accuracy'] - all_results['base']['accuracy']) * 100
    dpo_improvement = (all_results['dpo']['accuracy'] - all_results['sft']['accuracy']) * 100

    print(f"\nImprovements:")
    print(f"  SFT vs Base: {sft_improvement:+.2f}%")
    print(f"  DPO vs SFT:  {dpo_improvement:+.2f}%")

    print_sample_outputs(dpo_results['predictions'], num_samples=3)

    print("\nEvaluation completed! ✓")


if __name__ == "__main__":
    main()
