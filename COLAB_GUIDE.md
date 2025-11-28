# Google Colab Training Guide

Hướng dẫn chi tiết chạy project trên Google Colab Free

## 📋 Chuẩn bị

### 1. Upload Project lên Google Drive

```bash
# Trên máy local, zip project
cd llama32-mcq-cot
zip -r llama32-mcq-cot.zip .

# Hoặc upload thẳng folder lên Drive:
# MyDrive/llama32-mcq-cot/
```

### 2. Tạo Colab Notebook mới

1. Vào [Google Colab](https://colab.research.google.com/)
2. File → New notebook
3. Runtime → Change runtime type → T4 GPU

## 🚀 Notebook Cells

### Cell 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Navigate to Project

```python
import os
os.chdir('/content/drive/MyDrive/llama32-mcq-cot')

# Verify
!pwd
!ls
```

### Cell 3: Install Dependencies

```python
!pip install -q transformers>=4.44.0
!pip install -q datasets>=2.14.0
!pip install -q accelerate>=0.24.0
!pip install -q bitsandbytes>=0.41.0
!pip install -q peft>=0.6.0
!pip install -q trl>=0.7.0
!pip install -q wandb>=0.15.0
!pip install -q scipy scikit-learn

# Verify installations
import transformers
import torch
print(f"Transformers: {transformers.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 4: Login to HuggingFace & Wandb

```python
# HuggingFace login (for model access)
from huggingface_hub import notebook_login
notebook_login()

# Wandb login (for experiment tracking)
import wandb
wandb.login()
```

**Note:**
- HF Token: Get from https://huggingface.co/settings/tokens
- Wandb Token: Get from https://wandb.ai/authorize

### Cell 5: Test Data Loading

```python
!python src/prepare_data.py
```

**Expected output:**
- Loading dataset messages
- Sample examples printed
- Data statistics

**Time:** ~2-3 minutes

### Cell 6: Build DPO Preference Pairs

```python
!python src/build_dpo_data.py
```

**Expected output:**
- DPO pairs generated
- Saved to `data/dpo_pairs.jsonl`
- Sample pairs shown

**Time:** ~1-2 minutes

### Cell 7: Train SFT Model

```python
!python src/train_sft.py
```

**Expected:**
- Training progress bar
- Wandb tracking link
- Model saved to `outputs/sft-llama32-1b-mcq/`

**Time:** ~3-4 hours for 10k samples, 1 epoch

**Monitor:**
- Training loss should decrease
- Check wandb dashboard for metrics

**Tips:**
- Don't close browser (Colab may disconnect)
- Check periodically to ensure still running
- Checkpoints saved every epoch

### Cell 8: Train DPO Model

```python
!python src/train_dpo.py
```

**Expected:**
- Loads SFT checkpoint
- Training progress
- Model saved to `outputs/dpo-llama32-1b-mcq/`

**Time:** ~2-3 hours

### Cell 9: Evaluate All Models

```python
!python src/evaluate.py
```

**Expected:**
- Evaluates base, SFT, DPO models
- Prints accuracy comparison
- Sample outputs shown

**Time:** ~15-30 minutes

## 💡 Tips & Tricks

### Memory Management

If you get OOM errors:

```python
# Clear cache between runs
import torch
torch.cuda.empty_cache()

# Check memory usage
!nvidia-smi
```

### Subsample for Testing

Edit `configs/config.py` before training:

```python
# In configs/config.py
data_config.train_sample_size = 1000  # Use 1k instead of 10k
```

This reduces training time to ~30 minutes for quick testing.

### Save Checkpoints Frequently

Training gets interrupted? Models are auto-saved:
- After each epoch (default)
- Checkpoints in `outputs/*/checkpoint-*/`

### Monitor Training

**Option 1: Wandb Dashboard**
- Click wandb link in training output
- Monitor loss, learning rate, etc.

**Option 2: Check Logs**
```python
# In separate cell while training
!tail -f outputs/sft-llama32-1b-mcq/trainer_log.txt
```

### Download Models

After training, download to local machine:

```python
# Zip the outputs
!zip -r models.zip outputs/

# Download via Colab UI
from google.colab import files
files.download('models.zip')
```

## ⚠️ Common Issues

### Issue 1: Colab Disconnects

**Solution:**
- Use Colab Pro (longer sessions)
- Or split training into multiple sessions
- Resume from last checkpoint

### Issue 2: "No space left on device"

**Solution:**
```python
# Clean up previous runs
!rm -rf outputs/sft-llama32-1b-mcq/checkpoint-*
!rm -rf ~/.cache/huggingface/
```

### Issue 3: Import errors

**Solution:**
```python
# Restart runtime and reinstall
# Runtime → Restart runtime
# Then run install cells again
```

### Issue 4: Dataset download fails

**Solution:**
```python
# Check internet connection
!ping -c 3 huggingface.co

# Manual download
from datasets import load_dataset
ds = load_dataset("tau/commonsense_qa", cache_dir="./hf_cache")
```

## 📊 Expected Timeline (Colab Free)

| Step | Time | Notes |
|------|------|-------|
| Setup | 5-10 min | Installing deps, login |
| Data prep | 2-3 min | Loading ECQA |
| Build DPO data | 1-2 min | Creating pairs |
| SFT training | 3-4 hours | 10k samples, 1 epoch |
| DPO training | 2-3 hours | 10k pairs, 1 epoch |
| Evaluation | 15-30 min | All 3 models |
| **Total** | **6-8 hours** | Can split across days |

## 🔄 Multi-Session Strategy

If Colab disconnects or you need to split:

**Day 1 (4 hours):**
1. Setup + Data prep (10 min)
2. SFT training (3-4 hours)
3. Save to Drive ✓

**Day 2 (3 hours):**
1. Resume session, mount Drive
2. Build DPO data (2 min)
3. DPO training (2-3 hours)
4. Save to Drive ✓

**Day 3 (1 hour):**
1. Resume session
2. Evaluate all models (30 min)
3. Document results

## 📈 Tracking Progress

Create a progress tracking cell:

```python
import os
from pathlib import Path

print("Project Status Check")
print("="*50)

# Check data
if Path("data/dpo_pairs.jsonl").exists():
    print("✓ DPO data prepared")
else:
    print("✗ DPO data not ready")

# Check SFT
if Path("outputs/sft-llama32-1b-mcq-merged").exists():
    print("✓ SFT training completed")
else:
    print("✗ SFT training pending")

# Check DPO
if Path("outputs/dpo-llama32-1b-mcq-merged").exists():
    print("✓ DPO training completed")
else:
    print("✗ DPO training pending")

print("="*50)
```

## 🎯 Quick Test Run (30 minutes)

For testing the pipeline without full training:

```python
# Edit configs/config.py
data_config.train_sample_size = 500
data_config.val_sample_size = 100
sft_config.num_train_epochs = 1
dpo_config.num_train_epochs = 1

# Then run full pipeline
!python src/prepare_data.py
!python src/build_dpo_data.py
!python src/train_sft.py  # ~20-30 min with 500 samples
!python src/train_dpo.py  # ~15-20 min
!python src/evaluate.py   # ~5 min
```

This gives you:
- Full pipeline verification
- Basic results
- Experience with the workflow

## 🆘 Getting Help

If stuck:
1. Check error messages carefully
2. Google the specific error
3. Check HuggingFace/TRL documentation
4. Review code comments in source files

## ✅ Final Checklist

Before starting:
- [ ] GPU runtime enabled (T4)
- [ ] Project uploaded to Drive
- [ ] HuggingFace token ready
- [ ] Wandb account created
- [ ] Dependencies installed
- [ ] Enough Drive space (~5GB free)

After training:
- [ ] Models saved to Drive
- [ ] Wandb dashboard reviewed
- [ ] Results documented
- [ ] Checkpoints backed up

---

**Happy Training on Colab! 🚀**
