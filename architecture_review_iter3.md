# Architecture Review - Iteration 3

## Current Architecture Analysis

### What's Good:
1. ✅ Increased capacity (16→32→32 filters) vs iteration 2
2. ✅ Three convolutional blocks for deeper feature extraction
3. ✅ Global Average Pooling reduces parameters efficiently
4. ✅ Batch normalization after each conv layer
5. ✅ Progressive approach to building depth

---

## Critical Issues to Fix

### 🔴 PRIORITY 1: Normalization Statistics Mismatch
**Problem:**
```python
train: Normalize((0.1307,), (0.3081,))
test:  Normalize((0.1407,), (0.4081,))  # WRONG!
```

**Impact:** Train and test distributions don't match. This WILL hurt accuracy.

**Fix:** Use same normalization for both:
```python
# Both should use:
transforms.Normalize((0.1307,), (0.3081,))
```

---

### 🟠 PRIORITY 2: Dropout Placement & Rates

**Problem 1 - Placement:**
Current: `Conv → BN → ReLU → Dropout → MaxPool`
Standard: `Conv → BN → ReLU → MaxPool → Dropout`

Dropout before pooling can interfere with spatial downsampling.

**Problem 2 - Too High with BatchNorm:**
- Using 0.3 dropout PLUS BatchNorm is excessive
- BatchNorm already provides regularization
- High dropout can prevent model from learning complex features

**Fix:**
```python
# Block 1
x = F.relu(self.bn1(self.conv1(x)))
x = F.relu(self.bn2(self.conv2(x)))
x = F.max_pool2d(x, 2)
x = self.dropout1(x)  # Move dropout AFTER pooling

# Reduce dropout rates:
self.dropout1 = nn.Dropout2d(0.05)  # Was 0.1
self.dropout2 = nn.Dropout2d(0.10)  # Was 0.2
self.dropout3 = nn.Dropout2d(0.15)  # Was 0.3
```

---

### 🟡 PRIORITY 3: Inefficient Last Convolution

**Problem:**
```python
self.conv5 = nn.Conv2d(32, 32, kernel_size=3)  # 4x4 → 2x2
```

Going from 4x4 to 2x2 with a 3x3 conv loses a lot of spatial information. Then immediately applying GAP on 2x2 is wasteful.

**Fix Option A - Remove last conv:**
```python
# Remove conv5 entirely, apply GAP directly to 4x4x32
# Block 2 ends at 4x4x32
x = self.gap(x)  # 4x4x32 → 1x1x32
```

**Fix Option B - Use 1x1 conv instead:**
```python
self.conv5 = nn.Conv2d(32, 64, kernel_size=1)  # 4x4x32 → 4x4x64
# Then GAP: 4x4x64 → 1x1x64
self.fc = nn.Linear(64, 10)
```

**Recommendation:** Option B adds more capacity without losing spatial info.

---

### 🟡 PRIORITY 4: Channel Progression

**Problem:**
Current: `1 → 16 → 16 → 32 → 32 → 32`

Not optimal - jumps from 16→32 then stays flat at 32.

**Better progression:**
```python
# Option A: Gradual increase
1 → 16 → 20 → 32 → 48

# Option B: Bottleneck style
1 → 16 → 16 → 32 → 32 → 48
```

**Fix:**
```python
# Block 1
self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(16, 20, kernel_size=3)  # 16→20 instead of 16→16

# Block 2
self.conv3 = nn.Conv2d(20, 32, kernel_size=3)  # 20→32
self.conv4 = nn.Conv2d(32, 48, kernel_size=3)  # 32→48 instead of 32→32

# Block 3 (using 1x1)
self.conv5 = nn.Conv2d(48, 64, kernel_size=1)  # 48→64

# Output
self.fc = nn.Linear(64, 10)
```

This uses ~19K parameters and has better feature progression.

---

## Additional Improvements

### 5. Add Transition Layers (1x1 Convs)

**Why:** Efficient channel reduction/expansion between blocks.

```python
# Add between blocks
self.transition1 = nn.Conv2d(20, 24, kernel_size=1)  # After block 1
self.transition2 = nn.Conv2d(48, 48, kernel_size=1)  # After block 2
```

### 6. Missing Activation Before GAP

**Problem:**
```python
x = F.relu(self.bn5(self.conv5(x)))  # 2x2x32
x = self.dropout3(x)
x = self.gap(x)  # No activation after dropout
```

The dropout masks values but there's no final activation.

**Fix:**
```python
x = F.relu(self.bn5(self.conv5(x)))
x = F.relu(x)  # Ensure positive activations
x = self.gap(x)
```

Or better, remove dropout before GAP:
```python
x = F.relu(self.bn5(self.conv5(x)))
x = self.gap(x)
x = x.view(-1, 64)
x = self.dropout3(x)  # Apply dropout to flattened features
x = self.fc(x)
```

### 7. Consider Depthwise Separable Convolutions

**Why:** Save parameters while maintaining performance.

**Example:**
```python
# Instead of:
self.conv4 = nn.Conv2d(32, 48, kernel_size=3)  # 32*48*3*3 = 13,824 params

# Use depthwise separable:
self.conv4_dw = nn.Conv2d(32, 32, kernel_size=3, groups=32)  # 32*3*3 = 288
self.conv4_pw = nn.Conv2d(32, 48, kernel_size=1)             # 32*48 = 1,536
# Total: 1,824 params (87% reduction!)
```

---

## Recommended Architecture for Iteration 4

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1: 28x28 → 13x13
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28x16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 20, 3)             # 26x26x20
        self.bn2 = nn.BatchNorm2d(20)
        # pool → 13x13x20
        self.dropout1 = nn.Dropout2d(0.05)

        # Block 2: 13x13 → 6x6
        self.conv3 = nn.Conv2d(20, 32, 3)             # 11x11x32
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 48, 3)             # 9x9x48
        self.bn4 = nn.BatchNorm2d(48)
        # pool → 4x4x48
        self.dropout2 = nn.Dropout2d(0.10)

        # Block 3: 4x4 → 4x4 (no spatial reduction)
        self.conv5 = nn.Conv2d(48, 64, 1)             # 4x4x64 (1x1 conv)
        self.bn5 = nn.BatchNorm2d(64)

        # Output
        self.gap = nn.AdaptiveAvgPool2d(1)            # 1x1x64
        self.dropout3 = nn.Dropout(0.15)              # Apply to vector
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))

        # Output
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.dropout3(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)
```

**Estimated parameters:** ~18-19K
**Expected improvement:** Should reach 99.4%+ with proper training

---

## Summary of Key Changes

| Issue | Current | Recommended |
|-------|---------|-------------|
| Test normalization | (0.1407, 0.4081) | (0.1307, 0.3081) |
| Dropout placement | Before pool | After pool |
| Dropout rates | 0.1, 0.2, 0.3 | 0.05, 0.10, 0.15 |
| Channel progression | 16→16→32→32→32 | 16→20→32→48→64 |
| Last conv | 3x3 on 4x4 | 1x1 (no spatial loss) |
| Final dropout | On 2D (2x2) | On 1D (vector) |

These changes should improve accuracy by 0.5-1.0% and help you reach the 99.4% goal.
