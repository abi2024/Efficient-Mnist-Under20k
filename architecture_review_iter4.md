# Architecture Review - Iteration 4

## Executive Summary

**Current Status:** ⚠️ **Mixed Results**
- ✅ Achieved accuracy goal (~99.4%)
- ❌ Exceeded parameter limit (26,202 vs 20,000 goal)
- Need to reduce 6,202 parameters while maintaining performance

---

## Performance Analysis

### What Worked Well ✅

1. **Fixed Normalization Statistics**
   - Both train and test now use (0.1307, 0.3081)
   - Eliminated train/test distribution mismatch
   - **Impact:** Improved convergence and test accuracy

2. **Optimized Dropout Strategy**
   - Reduced rates: 0.05 → 0.10 → 0.15
   - Placement after pooling preserves spatial features
   - **Impact:** Better gradient flow, reduced overfitting

3. **Progressive Channel Expansion**
   - 1 → 16 → 20 → 32 → 48 → 64
   - More gradual than iteration 3
   - **Impact:** Better feature hierarchy learning

4. **1x1 Convolution in Block 3**
   - Increases channels without spatial reduction (48→64)
   - More efficient than 3x3 on small feature maps
   - **Impact:** Better feature mixing at high level

5. **Global Average Pooling**
   - Reduces FC layer from 720→10 to 64→10
   - **Impact:** Significant parameter reduction

### What Didn't Work ❌

1. **Parameter Budget Exceeded**
   - Current: 26,202 parameters
   - Goal: 20,000 parameters
   - Over by: 6,202 (31% over budget)

2. **Channel Progression Too Aggressive**
   - Jump from 32→48 adds 9,248 parameters
   - Could be more conservative while maintaining accuracy

---

## Detailed Parameter Breakdown

```
Layer               Shape                   Parameters
===============================================================
conv1               [1, 16, 3, 3]          160        (1*16*9 + 16)
bn1                 [16] x2                32
conv2               [16, 20, 3, 3]         2,900      (16*20*9 + 20)
bn2                 [20] x2                40
conv3               [20, 32, 3, 3]         5,792      (20*32*9 + 32)
bn3                 [32] x2                64
conv4               [32, 48, 3, 3]         13,872     (32*48*9 + 48) ← BIGGEST!
bn4                 [48] x2                96
conv5               [48, 64, 1, 1]         3,136      (48*64*1 + 64)
bn5                 [64] x2                128
fc                  [64, 10]               650        (64*10 + 10)
===============================================================
TOTAL                                      26,202
```

**Key Insight:** Conv4 (32→48) accounts for 53% of total parameters!

---

## Parameter Reduction Strategies

### 🎯 Strategy 1: Reduce Mid-layer Channels (Recommended)

**Target Reduction:** 6,500+ parameters

```python
# Current progression:
1 → 16 → 20 → 32 → 48 → 64

# Proposed progression:
1 → 16 → 20 → 28 → 36 → 48

# Parameter savings:
conv4: 32→48 (13,872) becomes 28→36 (9,108) = -4,764 params
conv5: 48→64 (3,136) becomes 36→48 (1,776) = -1,360 params
fc: 64→10 (650) becomes 48→10 (490) = -160 params
Total savings: ~6,284 parameters
```

**Expected parameters:** ~19,918 ✅

---

### 🎯 Strategy 2: Use Depthwise Separable Convolutions

**Target Reduction:** 8,000+ parameters

Replace standard convolutions with depthwise separable:

```python
# Standard Conv (current):
self.conv4 = nn.Conv2d(32, 48, 3)  # 13,872 params

# Depthwise Separable:
self.conv4_dw = nn.Conv2d(32, 32, 3, groups=32)  # 288 params
self.conv4_pw = nn.Conv2d(32, 48, 1)             # 1,536 params
# Total: 1,824 params (savings: 12,048!)

# Similarly for conv3:
self.conv3_dw = nn.Conv2d(20, 20, 3, groups=20)  # 180 params
self.conv3_pw = nn.Conv2d(20, 32, 1)             # 640 params
# Total: 820 params (savings: 4,972!)
```

**Total savings:** ~17,020 parameters
**Expected parameters:** ~9,182 ✅

---

### 🎯 Strategy 3: Bottleneck Architecture

**Target Reduction:** 7,000+ parameters

Use 1x1 convolutions to reduce channels before expensive 3x3 operations:

```python
# Block 2 with bottleneck:
self.reduce2 = nn.Conv2d(20, 16, 1)    # 320 params
self.conv3 = nn.Conv2d(16, 32, 3)      # 4,608 params (was 5,792)
self.conv4 = nn.Conv2d(32, 32, 3)      # 9,216 params (was 13,872)
self.expand2 = nn.Conv2d(32, 48, 1)    # 1,536 params

# Total Block 2: 15,680 (was 19,664)
# Savings: 3,984 params
```

---

### 🎯 Strategy 4: Asymmetric Convolutions

Replace some 3x3 convolutions with 3x1 followed by 1x3:

```python
# Instead of:
self.conv2 = nn.Conv2d(16, 20, 3)  # 2,880 params

# Use:
self.conv2_h = nn.Conv2d(16, 20, (1, 3), padding=(0, 1))  # 960 params
self.conv2_v = nn.Conv2d(20, 20, (3, 1), padding=(1, 0))  # 1,200 params
# Total: 2,160 params (savings: 720)
```

---

## Recommended Architecture for Next Iteration

### Option A: Conservative Channel Reduction (Safest)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1: 28x28 → 13x13
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28x16 (160)
        self.bn1 = nn.BatchNorm2d(16)                 # (32)
        self.conv2 = nn.Conv2d(16, 20, 3)             # 26x26x20 (2,900)
        self.bn2 = nn.BatchNorm2d(20)                 # (40)
        self.dropout1 = nn.Dropout2d(0.05)

        # Block 2: 13x13 → 4x4
        self.conv3 = nn.Conv2d(20, 28, 3)             # 11x11x28 (5,068)
        self.bn3 = nn.BatchNorm2d(28)                 # (56)
        self.conv4 = nn.Conv2d(28, 36, 3)             # 9x9x36 (9,108)
        self.bn4 = nn.BatchNorm2d(36)                 # (72)
        self.dropout2 = nn.Dropout2d(0.10)

        # Block 3: 4x4 → 4x4
        self.conv5 = nn.Conv2d(36, 48, 1)             # 4x4x48 (1,776)
        self.bn5 = nn.BatchNorm2d(48)                 # (96)

        # Output
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout3 = nn.Dropout(0.15)
        self.fc = nn.Linear(48, 10)                   # (490)

    # Total: ~19,798 parameters ✅
```

### Option B: Depthwise Separable (Most Efficient)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1: Standard convolutions for initial feature extraction
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # (160)
        self.bn1 = nn.BatchNorm2d(16)                 # (32)
        self.conv2 = nn.Conv2d(16, 24, 3)             # (3,480)
        self.bn2 = nn.BatchNorm2d(24)                 # (48)
        self.dropout1 = nn.Dropout2d(0.05)

        # Block 2: Depthwise separable convolutions
        self.conv3_dw = nn.Conv2d(24, 24, 3, groups=24)  # (216)
        self.conv3_pw = nn.Conv2d(24, 32, 1)             # (768)
        self.bn3 = nn.BatchNorm2d(32)                    # (64)

        self.conv4_dw = nn.Conv2d(32, 32, 3, groups=32)  # (288)
        self.conv4_pw = nn.Conv2d(32, 48, 1)             # (1,536)
        self.bn4 = nn.BatchNorm2d(48)                    # (96)
        self.dropout2 = nn.Dropout2d(0.10)

        # Block 3: Channel expansion
        self.conv5 = nn.Conv2d(48, 64, 1)             # (3,072)
        self.bn5 = nn.BatchNorm2d(64)                 # (128)

        # Output
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout3 = nn.Dropout(0.15)
        self.fc = nn.Linear(64, 10)                   # (650)

    # Total: ~10,538 parameters ✅ (but may need tuning)
```

---

## Training Recommendations

### If Using Strategy 1 (Channel Reduction):
- Keep current training setup (works well)
- May need 1-2 extra epochs to compensate for capacity reduction

### If Using Strategy 2 (Depthwise Separable):
- Increase learning rate slightly (0.01 → 0.015)
- Add warmup for first 2 epochs
- May need gradient clipping

### If Using Strategy 3/4 (Bottleneck/Asymmetric):
- Add residual connections if accuracy drops
- Consider using OneCycleLR scheduler

---

## Key Learnings from Iteration 4

1. **Normalization fix was critical** - Same stats for train/test improved accuracy significantly
2. **Dropout after pooling works better** - Preserves spatial features during training
3. **1x1 convolutions are efficient** - Good for channel mixing without spatial reduction
4. **Conv4 is the parameter bottleneck** - 32→48 jump consumes 53% of parameters
5. **GAP + small FC is effective** - Reduces parameters while maintaining performance

---

## Recommended Next Steps

### Priority 1: Implement Strategy 1 (Conservative Channel Reduction)
- **Risk:** Low
- **Expected accuracy impact:** < 0.2%
- **Parameter reduction:** 6,284
- **Implementation time:** 10 minutes

### Priority 2: Test Strategy 2 (Depthwise Separable) if Strategy 1 insufficient
- **Risk:** Medium (may need architecture tuning)
- **Expected accuracy impact:** 0.3-0.5% (can be recovered with training)
- **Parameter reduction:** 17,020
- **Implementation time:** 30 minutes

### Backup: Combine Strategies
- Use channel reduction + depthwise separable only for conv4
- Achieves parameter goal with minimal risk

---

## Summary

**Current Architecture Strengths:**
- Achieves 99.4% accuracy goal ✅
- Well-structured progressive feature extraction
- Effective regularization strategy

**Required Improvements:**
- Reduce parameters by 6,202 (from 26,202 to < 20,000)
- Maintain 99.4% accuracy

**Recommended Approach:**
1. Start with Strategy 1 (channel reduction: 16→20→28→36→48)
2. If accuracy maintained, you're done
3. If accuracy drops > 0.2%, add residual connections or slight capacity increase
4. If still over budget, selectively apply depthwise separable to conv4

**Expected Outcome:**
- Parameters: ~19,798 ✅
- Accuracy: 99.3-99.4% ✅
- Training time: Similar to current