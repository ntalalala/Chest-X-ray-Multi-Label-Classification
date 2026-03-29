# NIH Chest X-ray Multi-Label Classification: Model Comparison Report

## 1. Summary

This report presents a comprehensive comparison of six deep learning architectures for multi-label classification of chest X-ray images using the NIH ChestX-ray14 dataset. All models were trained with consistent methodology to ensure fair comparison.

### Key Findings

| Rank | Model | OOF AUC | Test AUC | Test F1 (Optimal) |
|------|-------|---------|----------|-------------------|
| 1 | **EfficientNet-B0** | **74.95%** | **71.88%** | 19.94% |
| 2 | InceptionV3 | 74.59% | 70.81% | **20.17%** |
| 3 | MobileNetV3-Large | 73.52% | 68.56% | 18.23% |
| 4 | DenseNet-121 | 71.84% | 69.06% | 13.39% |
| 5 | Xception | 69.90% | 66.71% | 14.74% |
| 6 | ResNet-50 | 69.18% | 63.07% | 13.64% |

**Best Overall Model**: EfficientNet-B0 achieved the highest OOF AUC and Test AUC. InceptionV3 achieved the highest Test F1.

---

## 2. Experimental Setup

### 2.1 Dataset

| Attribute | Value |
|-----------|-------|
| Dataset | NIH ChestX-ray14 (Subset) |
| Training Images | 15,155 |
| Test Images | 4,788 |
| Unique Patients (Train) | 5,000 |
| Unique Patients (Test) | 500 |
| Number of Classes | 14 diseases |

### 2.2 Disease Labels

The 14 disease categories:
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural_Thickening
13. Pneumonia
14. Pneumothorax

### 2.3 Common Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 5×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| Optimizer | AdamW |
| Loss Function | Focal Loss (α=1.0, γ=2.0) |
| Label Smoothing | 0.1 |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Cross-Validation | GroupKFold (K=3, grouped by Patient ID) |
| Early Stop Patience | 5 epochs |
| Early Stop Epochs | 15 |
| Full Training Epochs | 20 |

### 2.4 Preprocessing Pipeline

All models use identical preprocessing:

```
1. Adaptive Lung Crop + CLAHE
   - Convert to grayscale
   - Otsu's thresholding to find lung region
   - Crop to bounding box with padding (12px)
   - CLAHE contrast enhancement (clip_limit=2.0)

2. Training Augmentations:
   - Random Rotation (±15°)
   - Random Affine (translate=0.05, scale=0.95-1.05)
   - Color Jitter (brightness=0.15, contrast=0.15)

3. Normalization:
   - ImageNet mean: [0.485, 0.456, 0.406]
   - ImageNet std: [0.229, 0.224, 0.225]
```

---

## 3. Model Architectures

### 3.1 EfficientNet-B0

| Attribute | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (pretrained) |
| Input Size | 256×256 |
| Feature Dim | 1280 |
| Frozen Layers | features.0 through features.4 |
| Activation | SiLU (Swish) |
| Total Parameters | ~5.3M |
| Trainable Parameters | ~4.5M |

**Classifier Head:**
```
EfficientNet Features (1280)
    → Linear(1280, 512) → BN → SiLU → Dropout(0.3)
    → Linear(512, 256) → BN → SiLU → Dropout(0.18)
    → Linear(256, 14)
```

### 3.2 InceptionV3

| Attribute | Value |
|-----------|-------|
| Backbone | Inception v3 (pretrained, aux_logits=False) |
| Input Size | 299×299 |
| Feature Dim | 2048 |
| Frozen Layers | Early inception blocks |
| Activation | ReLU |
| Total Parameters | ~23.8M |
| Trainable Parameters | ~21.5M |

**Classifier Head:**
```
Inception Features (2048)
    → Linear(2048, 512) → BN → ReLU → Dropout(0.3)
    → Linear(512, 256) → BN → ReLU → Dropout(0.18)
    → Linear(256, 14)
```

### 3.3 MobileNetV3-Large

| Attribute | Value |
|-----------|-------|
| Backbone | MobileNetV3-Large (pretrained) |
| Input Size | 256×256 |
| Feature Dim | 960 |
| Frozen Layers | features.0 through features.7 |
| Activation | HardSwish |
| Total Parameters | ~3.6M |
| Trainable Parameters | ~0.7M |

**Classifier Head:**
```
MobileNet Features (960)
    → Linear(960, 512) → BN → HardSwish → Dropout(0.3)
    → Linear(512, 256) → BN → HardSwish → Dropout(0.18)
    → Linear(256, 14)
```

### 3.4 DenseNet-121

| Attribute | Value |
|-----------|-------|
| Backbone | DenseNet-121 (pretrained) |
| Input Size | 256×256 |
| Feature Dim | 1024 |
| Frozen Layers | denseblock1, denseblock2 |
| Activation | ReLU |
| Total Parameters | ~7.6M |
| Trainable Parameters | ~6.4M |

**Classifier Head:**
```
DenseNet Features (1024)
    → Linear(1024, 512) → BN → ReLU → Dropout(0.3)
    → Linear(512, 256) → BN → ReLU → Dropout(0.18)
    → Linear(256, 14)
```

### 3.5 Xception

| Attribute | Value |
|-----------|-------|
| Backbone | Xception (pretrained via timm) |
| Input Size | 299×299 |
| Feature Dim | 2048 |
| Frozen Layers | Blocks 0-7 |
| Activation | ReLU |
| Total Parameters | ~22.0M |
| Trainable Parameters | ~14.4M |

**Classifier Head:**
```
Xception Features (2048)
    → Linear(2048, 512) → BN → ReLU → Dropout(0.3)
    → Linear(512, 256) → BN → ReLU → Dropout(0.18)
    → Linear(256, 14)
```

### 3.6 ResNet-50

| Attribute | Value |
|-----------|-------|
| Backbone | ResNet-50 (pretrained) |
| Input Size | 256×256 |
| Feature Dim | 2048 |
| Frozen Layers | layer1, layer2 |
| Activation | ReLU |
| Total Parameters | ~24.7M |
| Trainable Parameters | ~23.3M |

**Classifier Head:**
```
ResNet Features (2048)
    → Linear(2048, 512) → BN → ReLU → Dropout(0.3)
    → Linear(512, 256) → BN → ReLU → Dropout(0.18)
    → Linear(256, 14)
```

---

## 4. Results

### 4.1 Overall Performance Comparison

| Model | OOF AUC | Test AUC | Test F1 (Fixed 0.5) | Test F1 (Optimal) | Test Precision | Test Recall |
|-------|---------|----------|---------------------|-------------------|----------------|-------------|
| **EfficientNet-B0** | **74.95%** | **71.88%** | 0.98% | 19.94% | 17.15% | 29.19% |
| InceptionV3 | 74.59% | 70.81% | 3.12% | **20.17%** | **18.67%** | 27.95% |
| MobileNetV3-Large | 73.52% | 68.56% | 3.22% | 18.23% | 14.26% | **29.07%** |
| DenseNet-121 | 71.84% | 69.06% | 1.62% | 13.39% | 19.96% | 20.10% |
| Xception | 69.90% | 66.71% | 0.86% | 14.74% | 11.97% | 23.31% |
| ResNet-50 | 69.18% | 63.07% | 1.67% | 13.64% | 12.34% | 21.70% |

### 4.2 Training Configuration Results

| Model | Best Mode | Best Fold |
|-------|-----------|-----------|
| EfficientNet-B0 | full_epoch | 1 |
| InceptionV3 | early_stop | 3 |
| MobileNetV3-Large | full_epoch | 3 |
| DenseNet-121 | full_epoch | 3 |
| Xception | full_epoch | 3 |
| ResNet-50 | early_stop | 3 |

### 4.3 Optimal Threshold Analysis

Per-class optimal threshold tuning (min_precision ≥ 15%) significantly improves F1-score:

| Model | F1 (Fixed 0.5) | F1 (Optimal) | Improvement |
|-------|----------------|--------------|-------------|
| InceptionV3 | 3.12% | 20.17% | +17.05% |
| EfficientNet-B0 | 0.98% | 19.94% | +18.96% |
| MobileNetV3-Large | 3.22% | 18.23% | +15.01% |
| Xception | 0.86% | 14.74% | +13.88% |
| ResNet-50 | 1.67% | 13.64% | +11.97% |
| DenseNet-121 | 1.62% | 13.39% | +11.77% |

**Key Insight**: Using a fixed threshold of 0.5 results in extremely low F1-scores due to severe class imbalance. Per-class optimal thresholds (typically ranging from 0.24 to 0.50) dramatically improve classification performance.

### 4.4 Model Efficiency Analysis Ranking

| Model | Total Params | Trainable Params | Input Size | Ranking|
|-------|--------------|------------------|------------|----------------|
| MobileNetV3-Large | 3.6M | 0.7M | 256×256 |1 (Fastest) |
| EfficientNet-B0 | 5.3M | 4.5M | 256×256 | 2 |
| DenseNet-121 | 7.6M | 6.4M | 256×256 | 3 |
| Xception | 22.0M | 14.4M | 299×299 | 4 |
| InceptionV3 | 23.8M | 21.5M | 299×299 | 4 |
| ResNet-50 | 24.7M | 23.3M | 256×256 | 4 |

**Best Efficiency-Performance Trade-off**:
- **EfficientNet-B0** offers the best performance with relatively small model size
- **MobileNetV3-Large** provides excellent efficiency (smallest trainable params) with competitive performance

---

## 5. Per-Disease Performance Analysis

### 5.1 Best Model Per Disease (Based on Test AUC)

| Disease | Best Model | Test AUC |
|---------|------------|----------|
| Atelectasis | EfficientNet-B0 / InceptionV3 | ~66-67% |
| Cardiomegaly | EfficientNet-B0 / InceptionV3 | ~68-70% |
| Consolidation | InceptionV3 | ~69% |
| Edema | EfficientNet-B0 | ~78-80% |
| Effusion | EfficientNet-B0 | ~74-75% |
| Emphysema | EfficientNet-B0 / InceptionV3 | ~73-74% |
| Fibrosis | EfficientNet-B0 | ~68-70% |
| Hernia | MobileNetV3 / InceptionV3 | ~71-76% |
| Infiltration | EfficientNet-B0 | ~65% |
| Mass | EfficientNet-B0 / MobileNetV3 | ~65% |
| Nodule | EfficientNet-B0 | ~60% |
| Pleural_Thickening | EfficientNet-B0 | ~69% |
| Pneumonia | Various | ~61-63% |
| Pneumothorax | InceptionV3 / MobileNetV3 | ~76-77% |

### 5.2 Most Challenging Diseases

1. **Nodule** - Consistently lowest AUC (~56-60%) across all models
2. **Pneumonia** - Low AUC (~61-67%) and near-zero F1 for most models
3. **Hernia** - Rare disease, difficult to detect

### 5.3 Best Detected Diseases

1. **Edema** - High AUC (~77-87%) across models
2. **Effusion** - Consistent performance (~71-85%) with reasonable F1
3. **Pneumothorax** - Good AUC (~72-77%)

---

## 6. Key Observations

### 6.1 Architecture Insights

1. **EfficientNet-B0** achieved the best overall results, validating its efficient compound scaling approach
2. **InceptionV3** performed comparably to EfficientNet, but requires larger input size (299×299)
3. **MobileNetV3-Large** offers the best parameter efficiency with competitive performance
4. **ResNet-50** underperformed despite its larger parameter count, suggesting diminishing returns for this task
5. **DenseNet-121** showed good generalization (small OOF-Test AUC gap) but lower absolute performance

### 6.2 Training Insights

1. **Early stopping** was beneficial for InceptionV3 and ResNet-50
2. **Full training** was preferred for EfficientNet, MobileNet, DenseNet, and Xception
3. **Fold 3** consistently performed best for most models, suggesting consistent data quality

### 6.3 Threshold Optimization Impact

1. Fixed threshold (0.5) results in **extremely poor F1-scores** (0.86-3.22%)
2. Per-class optimal thresholds improve F1 by **12-19 percentage points**
3. Optimal thresholds typically range from **0.24 to 0.50** depending on disease prevalence

### 6.4 Class Imbalance Challenges

1. Rare diseases (Hernia, Pneumonia, Fibrosis) remain difficult to detect
2. Focal Loss helps but cannot fully compensate for extreme imbalance
3. Label smoothing helps handle NLP-extracted label noise

---

## 7. Recommendations

### 7.1 For Production Deployment

| Scenario | Recommended Model | Rationale |
|----------|-------------------|-----------|
| High Accuracy Required | EfficientNet-B0 | Best overall performance |
| Resource Constrained | MobileNetV3-Large | Smallest model, fast inference |
| Balanced | EfficientNet-B0 | Good accuracy-efficiency trade-off |

### 7.2 For Further Improvement

1. **Ensemble Methods**: Combine top 3-4 models for improved robustness
2. **Data Augmentation**: Consider more aggressive augmentation for rare diseases
3. **Weighted Sampling**: Use WeightedRandomSampler to better handle class imbalance
4. **Larger Image Size**: Test EfficientNet-B0 with 384×384 input on better hardware
---

## 8. Conclusion

This comprehensive evaluation demonstrates that **EfficientNet-B0** is the best-performing architecture for NIH Chest X-ray multi-label classification, achieving:

- **Highest OOF AUC**: 74.95%
- **Highest Test AUC**: 71.88%
- **Test F1 (Optimal)**: 19.94%

The study also highlights the critical importance of **per-class optimal threshold tuning**, which improved F1-scores by an average of 15 percentage points across all models.

For resource-constrained deployments, **MobileNetV3-Large** offers an excellent alternative with only 3.6M parameters while maintaining competitive performance.

---

## Appendix A: Technical Environment

| Component | Specification |
|-----------|---------------|
| GPU | Tesla T4 (15.6 GB) |
| Framework | PyTorch |
| Mixed Precision | FP16 (via torch.amp) |
| CUDA Optimizations | cuDNN benchmark enabled |
| Random Seed | 42 |


