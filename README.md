# Llama 3.2 1B: SFT + DPO for Multiple-Choice Reasoning with CoT

Fine-tuning Llama 3.2 1B Instruct on the ECQA (Commonsense QA) dataset with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to generate Chain-of-Thought (CoT) reasoning for multiple-choice questions.

## 🎯 Project Overview

This project demonstrates:
- **Supervised Fine-Tuning (SFT)** with QLoRA on commonsense reasoning questions
- **Direct Preference Optimization (DPO)** to improve reasoning quality
- **Chain-of-Thought (CoT)** generation for explainable answers
- Training optimized for **Google Colab Free** (T4 GPU, 12GB RAM)

## 📊 Dataset

**ECQA (Explanations for Commonsense QA)**
- Source: `tau/commonsense_qa` on HuggingFace
- ~10k training samples with human-written explanations
- 5-choice multiple-choice questions on commonsense reasoning

Example:
```
Question: Where would you find a jellyfish that has not been captured?
A. ocean
B. store
C. tank
D. internet
E. aquarium

Explanation: Jellyfish live naturally in the ocean before being captured.
Answer: A
```

## 🏗️ Architecture

### Model
- **Base**: `meta-llama/Llama-3.2-1B-Instruct`
- **Quantization**: 4-bit (QLoRA) with `bitsandbytes`
- **LoRA**: r=16, alpha=32 on Q/K/V/O projection layers

### Training Strategy
1. **SFT Phase**: Train on (question + choices) → (CoT explanation + answer)
2. **DPO Phase**: Optimize preference between correct reasoning + answer vs. wrong answer

### Hyperparameters (Colab Free Optimized)

**SFT:**
- Batch size: 1 (gradient accumulation: 16)
- Learning rate: 2e-4
- Optimizer: paged_adamw_8bit
- Max sequence length: 512
- Memory usage: ~5-6GB

**DPO:**
- Batch size: 1 (gradient accumulation: 8)
- Learning rate: 5e-5
- Beta: 0.1
- Max sequence length: 512

## 📁 Project Structure

```
llama32-mcq-cot/
├── src/
│   ├── prepare_data.py       # Load & format ECQA dataset
│   ├── train_sft.py          # SFT training with QLoRA
│   ├── build_dpo_data.py     # Create preference pairs
│   ├── train_dpo.py          # DPO training
│   └── evaluate.py           # Evaluation script
├── configs/
│   └── config.py             # Centralized configuration
├── data/
│   └── dpo_pairs.jsonl       # Generated preference pairs
├── notebooks/                # (Optional) Exploration notebooks
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Setup (Local or Colab)

```bash
# Clone or download project
cd llama32-mcq-cot

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for model access)
huggingface-cli login

# Login to wandb (for experiment tracking)
wandb login
```

### 2. Prepare Data

```bash
# Test data loading and formatting
python src/prepare_data.py
```

This will:
- Load ECQA dataset from HuggingFace
- Format prompts and completions
- Show sample examples
- Validate data quality

### 3. Build DPO Preference Pairs

```bash
python src/build_dpo_data.py
```

Creates `data/dpo_pairs.jsonl` with (prompt, chosen, rejected) triplets.

### 4. Train SFT Model

```bash
python src/train_sft.py
```

**Outputs:**
- `outputs/sft-llama32-1b-mcq/` (LoRA adapters)
- `outputs/sft-llama32-1b-mcq-merged/` (merged model)

**Expected time:** ~3-4 hours on Colab T4 (1 epoch, 10k samples)

### 5. Train DPO Model

```bash
python src/train_dpo.py
```

**Outputs:**
- `outputs/dpo-llama32-1b-mcq/` (LoRA adapters)
- `outputs/dpo-llama32-1b-mcq-merged/` (merged model)

**Expected time:** ~2-3 hours on Colab T4 (1 epoch)

### 6. Evaluate Models

```bash
python src/evaluate.py
```

Compares:
- Base Llama-3.2-1B-Instruct
- SFT model
- DPO model

On validation set with accuracy metrics.

## 💻 Running on Google Colab

### Setup

1. Upload project to Google Drive: `MyDrive/llama32-mcq-cot/`
2. Create new Colab notebook
3. Mount Drive and install dependencies:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/llama32-mcq-cot

# Install dependencies
!pip install -q -r requirements.txt

# Login to HF and wandb
!huggingface-cli login
!wandb login
```

### Run Training Pipeline

```python
# Step 1: Prepare data
!python src/prepare_data.py

# Step 2: Build DPO pairs
!python src/build_dpo_data.py

# Step 3: Train SFT (3-4 hours)
!python src/train_sft.py

# Step 4: Train DPO (2-3 hours)
!python src/train_dpo.py

# Step 5: Evaluate
!python src/evaluate.py
```

### Managing Colab Sessions

If Colab disconnects:
- Checkpoints are saved to `outputs/` after each epoch
- Resume by re-running from the last completed step
- All outputs are saved to Drive

## 📈 Expected Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Base Llama-3.2-1B | ~40-50% | Random baseline: 20% |
| SFT | ~55-65% | +10-15% improvement |
| DPO | ~57-68% | +2-5% improvement |

**Note:** Results may vary based on random seed, hyperparameters, and dataset sampling.

## 🔧 Configuration

Edit [configs/config.py](configs/config.py) to customize:

- Model settings (LoRA rank, quantization)
- Training hyperparameters
- Dataset sampling (for faster experiments)
- Evaluation settings

Examples:
```python
# Use 5k samples instead of 10k for faster testing
data_config.train_sample_size = 5000

# Increase LoRA rank for potentially better quality
model_config.lora_r = 32

# Adjust learning rates
sft_config.learning_rate = 3e-4
dpo_config.learning_rate = 1e-4
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution 1:** Reduce sequence length
```python
sft_config.max_seq_length = 384
dpo_config.max_length = 384
```

**Solution 2:** Reduce LoRA rank
```python
model_config.lora_r = 8
```

**Solution 3:** Clear cache between runs
```python
import torch
torch.cuda.empty_cache()
```

### Slow Training

- Subsample dataset: `data_config.train_sample_size = 5000`
- Increase batch size if memory allows
- Use fewer epochs: `sft_config.num_train_epochs = 1`

### Dataset Not Found

Ensure you have internet connection and HuggingFace access:
```bash
huggingface-cli login
```

## 📚 Learning Outcomes

This project covers:
- ✅ Loading and preprocessing HuggingFace datasets
- ✅ 4-bit quantization with `bitsandbytes`
- ✅ QLoRA (Low-Rank Adaptation) for efficient fine-tuning
- ✅ Supervised Fine-Tuning with TRL's `SFTTrainer`
- ✅ Creating preference pairs for alignment
- ✅ Direct Preference Optimization (DPO)
- ✅ Evaluation of generative models
- ✅ Experiment tracking with wandb
- ✅ Google Colab workflow with Drive integration

## 🎓 CV/Portfolio Description

```
Multiple-Choice Reasoning with CoT and DPO (Llama 3.2 1B)

• Fine-tuned Llama 3.2 1B Instruct model with QLoRA on the ECQA commonsense
  reasoning dataset (~10k samples) to generate chain-of-thought explanations
  and accurate answers for 5-choice multiple-choice questions.

• Constructed preference pairs contrasting correct reasoning chains with
  wrong answers and trained with Direct Preference Optimization (DPO) to
  improve reasoning quality and answer accuracy.

• Evaluated on held-out questions, achieving ~15% accuracy improvement over
  base model with SFT and additional gains with DPO, demonstrating effective
  alignment through preference learning.

• Optimized training pipeline for Google Colab Free (T4 GPU, 12GB RAM) using
  4-bit quantization, gradient accumulation, and memory-efficient optimizers.

Tech Stack: PyTorch, HuggingFace Transformers, PEFT, TRL, bitsandbytes, wandb
```

## 📖 References

- **Llama 3.2**: [Meta AI](https://ai.meta.com/llama/)
- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **DPO**: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- **ECQA Dataset**: [HuggingFace](https://huggingface.co/datasets/tau/commonsense_qa)
- **TRL**: [HuggingFace TRL](https://github.com/huggingface/trl)

## 📝 License

This project is for educational purposes. Model weights are subject to Meta's Llama license.

## 🙏 Acknowledgments

- Meta AI for Llama 3.2
- HuggingFace for Transformers, PEFT, and TRL
- Tim Dettmers for bitsandbytes and QLoRA
- ECQA dataset creators

---

**Happy Training! 🚀**

For questions or issues, please refer to the code comments or check the configuration files.
