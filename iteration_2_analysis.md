# Iteration 2 Analysis

## Results Summary

**Architecture:** Same as Iteration 1
- Parameters: 9,190 (10,810 under budget)
- Best Test Accuracy: 99.06% (Epoch 19)
- Best Train Accuracy: 98.36%
- Goal: 99.4% in ≤20 epochs
- **Status: FAILED (0.34% short of goal)**

## Key Findings

### What Changed from Iteration 1
- Trained for 20 epochs instead of 5
- Same architecture, optimizer, and hyperparameters

### Performance Analysis
- First epoch: 96.35% (worse than iter 1's 97.87%)
- Improvement from iter 1: 99.06% vs 98.90% (+0.16%)
- Learning plateaus after epoch 15
- Only 0.34% away from 99.4% goal

### Issues Identified
1. **Slow convergence**: First epoch accuracy dropped from 97.87% to 96.35%
2. **Training plateaus**: Minimal improvement after epoch 15
3. **Model capacity**: Still has 10,810 unused parameters
4. **Gap remains**: 0.34% short despite 20 epochs

## Suggestions for Iteration 3

### Priority 1: Increase Model Capacity
Current architecture is too small. Add more capacity:
- Increase filters: 10→16, 20→32 (add ~3-4K params)
- Add third conv layer: 32→32 (add ~9K params)
- Total would be ~16-20K params (within budget)

### Priority 2: Improve Learning Rate Schedule
Current schedule (StepLR step=15, gamma=0.1) causes sudden drop:
- Try ReduceLROnPlateau for adaptive reduction
- Or use OneCycleLR for better convergence
- Consider higher initial LR (0.03-0.05) with proper scheduling

### Priority 3: Better Regularization
- Add more dropout layers (after each conv block)
- Increase dropout rate to 0.3-0.4
- Add L2 regularization (weight_decay=1e-4)

### Priority 4: Enhanced Data Augmentation
Current augmentation is limited:
- Increase RandomRotation to ±20°
- Add RandomAffine for translation/shear
- Try MixUp or CutMix
- Adjust normalization values (test vs train mismatch: 0.1307/0.3081 vs 0.1407/0.4081)

### Architecture Suggestion for Iter 3

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 28x28
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3)            # 26x26
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout2d(0.1)
        # MaxPool -> 13x13

        # Block 2
        self.conv3 = nn.Conv2d(16, 32, 3)            # 11x11
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3)            # 9x9
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout2d(0.2)
        # MaxPool -> 4x4

        # Block 3
        self.conv5 = nn.Conv2d(32, 32, 3)            # 2x2
        self.bn5 = nn.BatchNorm2d(32)

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
```

**Estimated parameters:** ~18-19K (within budget)

### Recommended Training Config for Iter 3

```python
optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.05,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.2
)
```

## Expected Outcome
With these changes, iteration 3 should:
- Reach >99.4% accuracy by epoch 15-18
- Better first epoch performance (>98%)
- More stable training with OneCycleLR
- Better generalization with increased capacity and regularization
