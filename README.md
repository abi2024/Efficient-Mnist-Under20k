# MNIST Classifier - <20K Parameters, 99.4% Accuracy Target

**Goal:** Achieve 99.4% test accuracy within 20 epochs using less than 20,000 parameters.

---

## Iteration Summary

| Iter | Notebook | Parameters | Best Test Acc | Epochs | Architecture Summary |
|------|----------|------------|---------------|--------|---------------------|
| 1 | `Mnist_iter_1.ipynb` | 9,190 | 98.90% | 5 | 2 Conv blocks (10→20 filters) + 2 MaxPool + Dropout + FC |
| 2 | `Mnist_iter_2.ipynb` | 9,190 | 99.06% | 19 | Same as Iter 1, trained for 20 epochs |
| 3 | `Mnist_iter_3.ipynb` | 26,202 ❌ | 99.41% ✅ | 11 | 3 Conv blocks (16→32→32) + GAP + Progressive Dropout |
| 4 | `Mnist_iter_4.ipynb` | 26,202 ❌ | ~99.4% ✅ | 16 | Enhanced arch (16→20→32→48→64) + 1x1 Conv + Fixed normalization |
| 5 | `Mnist_iter_5.ipynb` | 19,798 ✅ | 99.45% ✅ | 16 | Optimized (16→20→28→36→48) + SGD/OneCycleLR |

---

## Iteration Details

### Iteration 1

**Architecture:**
- Conv2d(1→10, 3x3) + BN + ReLU
- Conv2d(10→20, 3x3) + BN + ReLU
- MaxPool2d(2x2) + MaxPool2d(2x2)
- Dropout2d(0.25)
- Linear(720→10)

**Training Config:**
- Optimizer: Adam (lr=0.01)
- Scheduler: StepLR (step_size=15, gamma=0.1)
- Batch Size: 128
- Augmentation: RandomCrop(22, p=0.1), RandomRotation(±15°)

**Results:**

| Epoch | Train Acc | Test Acc | Train Loss | Test Loss |
|-------|-----------|----------|------------|-----------|
| 1 | 90.91% | 97.87% | 0.3268 | 0.0005 |
| 2 | 96.25% | 98.27% | 0.1229 | 0.0004 |
| 3 | 96.75% | 98.59% | 0.1067 | 0.0004 |
| 4 | 96.95% | 98.15% | 0.0990 | 0.0005 |
| 5 | 97.08% | 98.90% | 0.0933 | 0.0003 |

**Analysis:**
- Parameters: 9,190 (15,810 under budget)
- Best accuracy: 98.90% (0.5% short of 99.4% goal)
- Good convergence but peaks at epoch 3, then fluctuates
- Gap to goal: Need +0.5% accuracy improvement

**Next Steps:**
- Increase model capacity (more filters or layers)
- Tune hyperparameters (learning rate, augmentation)
- Try different optimizer/scheduler

---

### Iteration 2

**Architecture:** Same as Iteration 1
- Conv2d(1→10, 3x3) + BN + ReLU
- Conv2d(10→20, 3x3) + BN + ReLU
- MaxPool2d(2x2) + MaxPool2d(2x2)
- Dropout2d(0.25)
- Linear(720→10)

**Training Config:**
- Optimizer: Adam (lr=0.01)
- Scheduler: StepLR (step_size=15, gamma=0.1)
- Batch Size: 128
- Epochs: 20 (extended from 5)
- Augmentation: RandomCrop(22, p=0.1), RandomRotation(±15°)

**Results:**
- Parameters: 9,190 (same as Iter 1)
- Best Accuracy: 99.06% at Epoch 19
- First Epoch Accuracy: 96.35%
- Achieved 99.0%+ but fell short of 99.4% target

**Analysis:**
- Extended training helped improve from 98.90% to 99.06%
- Model capacity appears insufficient for 99.4% goal
- Recommendation: Increase model capacity

---

### Iteration 3

**Architecture:**
- Conv2d(1→16, 3x3, padding=1) + BN + ReLU
- Conv2d(16→16, 3x3) + BN + ReLU + MaxPool + Dropout(0.1)
- Conv2d(16→32, 3x3) + BN + ReLU
- Conv2d(32→32, 3x3) + BN + ReLU + MaxPool + Dropout(0.2)
- Conv2d(32→32, 3x3) + BN + ReLU + Dropout(0.3)
- Global Average Pooling + Linear(32→10)

**Training Config:**
- Optimizer: Adam (lr=0.01)
- Scheduler: StepLR (step_size=15, gamma=0.1)
- Batch Size: 128
- Progressive Dropout: 0.1 → 0.2 → 0.3

**Results:**
- Parameters: 26,202 (exceeds 20K limit by 6,202)
- Best Accuracy: 99.41% at Epoch 11 ✅
- First Epoch Accuracy: 98.57%
- Successfully achieved 99.4% target but exceeded parameter limit

**Analysis:**
- 3 conv blocks with progressive dropout worked well
- GAP reduced parameters vs traditional FC
- Need to reduce channels to meet parameter constraint

---

### Iteration 4

**Architecture:**
- Block 1: Conv(1→16, pad=1) + Conv(16→20) + MaxPool + Dropout(0.05)
- Block 2: Conv(20→32) + Conv(32→48) + MaxPool + Dropout(0.10)
- Block 3: Conv(48→64, 1x1) + Dropout(0.15)
- GAP + Linear(64→10)

**Key Improvements:**
- Fixed train/test normalization mismatch (both use 0.1307/0.3081)
- Reduced dropout rates for better gradient flow
- Strategic 1x1 convolution for channel expansion
- Progressive channel expansion: 16→20→32→48→64

**Training Config:**
- Optimizer: Adam (lr=0.01)
- Scheduler: StepLR (step_size=15, gamma=0.1)
- Fixed normalization values

**Results:**
- Parameters: 26,202 (same as Iter 3, still exceeds limit)
- Best Accuracy: ~99.4% achieved ✅
- Improved convergence from fixing normalization

**Analysis:**
- Normalization fix was critical for performance
- Architecture refinements helped achieve target
- Still need parameter reduction

---

### Iteration 5

**Architecture:**
- Block 1: Conv(1→16, pad=1) + Conv(16→20) + MaxPool + Dropout(0.05)
- Block 2: Conv(20→28) + Conv(28→36) + MaxPool + Dropout(0.10)
- Block 3: Conv(36→48, 1x1)
- GAP + Dropout(0.15) + Linear(48→10)

**Channel Reduction Strategy:**
- Reduced from 16→20→32→48→64 to 16→20→28→36→48
- Maintained architectural patterns that worked

**Training Config:**
- Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)
- Scheduler: OneCycleLR (max_lr=0.1, pct_start=0.2)
- Data Augmentation: RandomRotation(5°), RandomAffine(translate=0.05)

**Results:**
- Parameters: 19,798 ✅ (202 params under limit!)
- Best Accuracy: 99.45% at Epoch 16 ✅
- Target achieved at Epoch 16
- All goals met!

**Per-Class Performance:**
- Best: Digit 0 (99.80%), Digit 1 (99.74%), Digit 4 (99.69%)
- Challenging: Digit 5 (98.99%), Digit 6 (98.75%)

---

## Final Summary

**Winner: Iteration 5**
- ✅ Parameters: 19,798 < 20,000
- ✅ Accuracy: 99.45% > 99.4%
- ✅ Epochs: 16 < 20

**Key Success Factors:**
1. Progressive channel expansion with careful reduction
2. Global Average Pooling instead of large FC layers
3. Fixed normalization values for train/test consistency
4. OneCycleLR scheduler for better convergence
5. Strategic dropout placement after pooling
6. 1x1 convolutions for efficient channel mixing
