# Quick Start Guide: DINOv3 Shadow Detection

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision tensorboard matplotlib opencv-python pillow numpy --break-system-packages
```

### Step 2: Organize Your Files

Place the DINOv3 files in your MAMNet project structure:

```
your_project/
├── models/
│   ├── mamnet/           # Your existing MAMNet models
│   └── dinov3/           # NEW: DINOv3 models
│       ├── __init__.py
│       ├── dinov3_backbone.py
│       ├── dinov3_decoder.py
│       └── dinov3_model.py
├── data/                 # Your existing dataset.py
├── utils/                # Your existing metrics.py, losses.py, etc.
├── train_dinov3.py       # NEW: DINOv3 training script
├── compare_models.py     # NEW: Comparison script
└── outputs/              # Training outputs
```

### Step 3: Quick Test

Test the model before training:

```bash
# Test if DINOv3 loads correctly
python -c "from dinov3_model import DINOv3ShadowDetector; model = DINOv3ShadowDetector(); print('✓ Model loaded successfully!')"
```

### Step 4: Start Training

#### Option A: Single City (Quick Test)
```bash
python train_dinov3.py \
    --data_root /path/to/chicago/highres \
    --mode single \
    --batch_size 8 \
    --epochs 20 \
    --output_dir ./outputs/test_run
```

#### Option B: LOCO (Full Experiment)
```bash
python train_dinov3.py \
    --mode loco \
    --fold_id 0 \
    --base_data_root /path/to/Final_data_test \
    --resolution highres \
    --batch_size 8 \
    --epochs 50 \
    --output_dir ./outputs/loco_fold0
```

---

## 📊 Complete Workflow Example

### 1. Train MAMNet (if not done already)

```bash
python train.py \
    --data_root /path/to/chicago/highres \
    --mode single \
    --epochs 15 \
    --lr 0.001 \
    --batch_size 8 \
    --output_dir ./outputs/mamnet_baseline
```

### 2. Train DINOv3

```bash
python train_dinov3.py \
    --data_root /path/to/chicago/highres \
    --mode single \
    --epochs 50 \
    --lr 5e-5 \
    --batch_size 8 \
    --output_dir ./outputs/dinov3_baseline
```

### 3. Compare Results

```bash
python compare_models.py \
    --dinov3_dir ./outputs/dinov3_baseline \
    --mamnet_dir ./outputs/mamnet_baseline \
    --save_dir ./outputs/comparison
```

This generates:
- `model_comparison.png` - Side-by-side bar chart
- `comparison_summary.txt` - Detailed metrics comparison

---

## 🔄 LOCO Experiments (Leave-One-City-Out)

### Full LOCO Protocol (3 folds)

```bash
# Fold 0: Test on Phoenix
python train_dinov3.py \
    --mode loco --fold_id 0 \
    --base_data_root /path/to/Final_data_test \
    --resolution highres \
    --epochs 50 --batch_size 8 \
    --output_dir ./outputs/dinov3_loco_fold0

# Fold 1: Test on Miami
python train_dinov3.py \
    --mode loco --fold_id 1 \
    --base_data_root /path/to/Final_data_test \
    --resolution highres \
    --epochs 50 --batch_size 8 \
    --output_dir ./outputs/dinov3_loco_fold1

# Fold 2: Test on Chicago
python train_dinov3.py \
    --mode loco --fold_id 2 \
    --base_data_root /path/to/Final_data_test \
    --resolution highres \
    --epochs 50 --batch_size 8 \
    --output_dir ./outputs/dinov3_loco_fold2
```

### Aggregate LOCO Results

```python
# aggregate_loco.py
import json
import numpy as np

folds = ['fold0', 'fold1', 'fold2']
metrics = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']

results = {}
for fold in folds:
    path = f'./outputs/dinov3_loco_{fold}/test_results.json'
    with open(path, 'r') as f:
        results[fold] = json.load(f)

print("LOCO Aggregated Results (Mean ± Std)")
print("="*50)
for metric in metrics:
    values = [results[fold][metric] for fold in folds]
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric:<15}: {mean:.2f} ± {std:.2f}%")
```

---

## 🎯 Hyperparameter Tuning

### Learning Rate Search

```bash
for lr in 1e-5 5e-5 1e-4 5e-4; do
    python train_dinov3.py \
        --data_root /path/to/data \
        --lr $lr \
        --epochs 30 \
        --output_dir ./outputs/lr_${lr}
done
```

### Batch Size Search (if GPU memory allows)

```bash
for bs in 4 8 16; do
    python train_dinov3.py \
        --data_root /path/to/data \
        --batch_size $bs \
        --epochs 30 \
        --output_dir ./outputs/bs_${bs}
done
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Option 1: Reduce batch size
python train_dinov3.py --batch_size 4 ...

# Option 2: Freeze backbone
python train_dinov3.py --frozen_stages 6 ...

# Option 3: Use CPU (slow but works)
python train_dinov3.py --device cpu ...
```

### Issue 2: DINOv3 Download Fails

**Error**: `torch.hub.load` fails to download

**Solution**:
```bash
# Pre-download the model
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"

# Then run training normally
```

### Issue 3: Import Errors

**Error**: `ModuleNotFoundError: No module named 'data'`

**Solution**:
```bash
# Make sure you're in the project root directory
cd /path/to/your/project

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"
```

### Issue 4: Slow Training

**Expected**: ~2-3 minutes per epoch (depending on dataset size)

**If slower**:
```bash
# Reduce data loading workers if CPU bottleneck
python train_dinov3.py --num_workers 2 ...

# Check GPU utilization
nvidia-smi -l 1  # Should be >80% during training
```

---

## 📈 Monitoring Training

### Real-time with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir ./outputs/dinov3_baseline/tensorboard

# Open browser to http://localhost:6006
```

### Check Progress

```bash
# Quick peek at latest results
tail -f ./outputs/dinov3_baseline/training.log

# Or check saved plots
ls ./outputs/dinov3_baseline/*.png
```

---

## 🔬 Experiment Tracking

### Create experiment log

```bash
# experiments_log.sh
#!/bin/bash

EXP_NAME="dinov3_experiment_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="experiments/${EXP_NAME}.log"

echo "Experiment: $EXP_NAME" | tee -a $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE

python train_dinov3.py \
    --data_root /path/to/data \
    --epochs 50 \
    --output_dir ./outputs/$EXP_NAME \
    2>&1 | tee -a $LOG_FILE

echo "Completed: $(date)" | tee -a $LOG_FILE
```

---

## 🎓 Next Steps

1. **Baseline Comparison**: Train both MAMNet and DINOv3, compare results
2. **LOCO Evaluation**: Run full 3-fold LOCO to assess generalization
3. **Multi-scale**: Test on both highres and midres data
4. **Ablation Study**: Try different model variants (ViT-S vs ViT-B)
5. **Fine-tuning**: Experiment with frozen vs unfrozen backbone

---

## 💡 Pro Tips

1. **Start Small**: Test with 10-20 epochs first to ensure everything works
2. **Use Checkpoints**: Always save checkpoints (`--save_freq 5`)
3. **Monitor Overfitting**: Watch val vs train metrics in TensorBoard
4. **Compare Fairly**: Use same data splits and preprocessing for all models
5. **Document Everything**: Keep a log of experiments and hyperparameters

---

## 📞 Need Help?

- Check `README_DINOv3.md` for detailed documentation
- Review model architecture in `dinov3_model.py`
- Test individual components (`python dinov3_backbone.py`)
- Compare with MAMNet training to ensure data consistency

---

Good luck with your experiments! 🚀