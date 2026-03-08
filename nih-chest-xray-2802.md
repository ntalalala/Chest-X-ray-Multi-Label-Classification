# NIH Chest X-ray Multi-Label Classification with Deep Learning

## 1. Introduction

Chest X-ray imaging is one of the most widely used and cost-effective diagnostic tools for screening and monitoring thoracic diseases. However, the interpretation of chest radiographs requires substantial clinical expertise and is subject to inter-observer variability, especially in large-scale clinical settings. With the rapid development of deep learning, automated analysis of chest X-ray images has emerged as a promising approach to assist radiologists in disease detection and decision support.

In this project, we focus on the problem of **automatic multi-label classification of chest X-ray images**, where a single image may present multiple pathological findings simultaneously. The objective is to develop and analyze deep learning models capable of identifying various thoracic diseases from chest radiographs while addressing key challenges such as:

- **Class imbalance** - Some diseases are much rarer than others
- **Label noise** - Labels extracted via NLP may contain errors
- **Variability in imaging conditions** - Different equipment and protocols

---

## 2. Dataset Description

### 2.1 Data Source

The dataset used in this study is the **NIH ChestX-ray14** dataset, released by the National Institutes of Health Clinical Center.

| Attribute | Value |
|-----------|-------|
| Total Images | 112,120 frontal-view chest X-rays |
| Unique Patients | 30,805 |
| Disease Labels | 14 pathological conditions + "No Finding" |
| Image Format | PNG |
| Label Extraction | Natural Language Processing from radiology reports |

### 2.2 Disease Categories

The 14 disease labels in the dataset are:

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
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax

### 2.3 Data Schema

| Column Name | Description |
|-------------|-------------|
| Image Index | Filename: `{patient_id}_{follow_up}.png` |
| Finding Labels | Disease labels separated by `\|` (multi-label) |
| Follow-up # | Sequential index for patient follow-up visits |
| Patient ID | Unique patient identifier |
| Patient Age | Age at time of X-ray |
| Patient Gender | M (Male) or F (Female) |
| View Position | PA (Posteroanterior) or AP (Anteroposterior) |
| OriginalImageWidth | Original image width in pixels |
| OriginalImageHeight | Original image height in pixels |
| OriginalImagePixelSpacing_x | Horizontal pixel spacing (mm/pixel) |
| OriginalImagePixelSpacing_y | Vertical pixel spacing (mm/pixel) |

---

## 3. Exploratory Data Analysis

### 3.1 Dataset Overview

- **Total images**: 112,120
- **Unique patients**: 30,805
- **Average images per patient**: ~3.6

### 3.2 Patient Demographics

#### Age Distribution at first X-ray visit
- Median age approximately 45-55 years
- Some outlier ages (>100 years) were treated as missing and imputed using patient-level forward/backward fill

#### Gender Distribution
- Dataset contains both male and female patients
- Slight imbalance between genders

#### Number of Images per Patient
- The dataset includes 30,805 unique patients, with an average of 3.64 X-ray images per patient (std = 7.27). The distribution is highly right-skewed, with a median of 1 image, and 75% of patients having three images or fewer

#### Number of Distinct Diseases per Patient
The majority of patients have no radiographic abnormality (No Finding) or a single pathology, while a small subset presents multiple comorbid conditions, with up to 13 distinct diseases per patient.

### 3.3 Disease Distribution

The dataset exhibits **severe class imbalance**, which is a common challenge in medical imaging:

| Category | Observation |
|----------|-------------|
| Most Common | "No Finding" dominates the dataset |
| High Prevalence | Infiltration, Effusion, Atelectasis |
| Low Prevalence | Hernia, Pneumonia (rare conditions) |

#### Multi-Label Statistics
- **Single disease**: ~25% of abnormal images
- **Multiple diseases**: ~15% of abnormal images show co-occurring conditions
- **No Finding**: ~60% of images

### 3.4 Disease Co-occurrence Analysis

A co-occurrence matrix reveals interesting patterns:
- **Infiltration** frequently co-occurs with Atelectasis and Effusion

### 3.5 Multivariate Analysis

#### Age vs Disease
- The data show that disease cases are more frequently observed in the 45–60 age range, indicating a central concentration rather than a linear increase with age.

#### Gender vs Disease
- Most diseases are observed more frequently in male patients than in female patients.

---

## 4. Methodology

### 4.1 Data Preprocessing

#### Train-Test Split
- Use `train_val_list.txt` and `test_list.txt` to construct the training/validation and test datasets with an approximate 80/20 split, ensuring that the class distribution between the two sets is similar and that no patient appears in both train and test sets

#### Image Transformations

**Training Augmentations:**
```
- Resize to 256×256 (EfficientNet) or 384×384 (DenseNet A100)
- Random Rotation (±10°)
- Convert to Tensor
- Normalization (ImageNet mean/std: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

**Test Transformations:**
```
- Resize to 256×256 (EfficientNet) or 384×384 (DenseNet A100)
- Convert to Tensor
- Normalization (ImageNet mean/std: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

### 4.2 Model Architecture

#### Model 1: DenseNet-121

**DenseNet-121** is chosen for:
- Efficient feature reuse through dense connections
- Strong performance on medical imaging tasks
- Fewer parameters than comparable architectures

**Custom Classifier Head:**
```
DenseNet Features (1024) 
    → Linear(1024, 512) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(512, 256) → BatchNorm → ReLU → Dropout(0.18)
    → Linear(256, 14)
```

**Transfer Learning Strategy**: Freeze early layers, fine-tune last 30 layers

#### Model 2: EfficientNet-B0

**EfficientNet-B0** is chosen for:
- Compound scaling for balanced network dimensions
- Superior accuracy-efficiency trade-off
- Optimized for mobile and edge deployment

**Custom Classifier Head:**
```
EfficientNet Features (1280) 
    → Linear(1280, 512) → BatchNorm → SiLU → Dropout(0.3)
    → Linear(512, 256) → BatchNorm → SiLU → Dropout(0.18)
    → Linear(256, 14)
```

**Transfer Learning Strategy**: Freeze features.0 through features.4, fine-tune remaining layers

#### Common Design Choices

- **BatchNorm**: Stabilizes training, acts as regularization
- **Progressive Dropout**: Moderate dropout early (0.3), lower later (0.18)
- **No final Sigmoid**: Using BCEWithLogitsLoss for numerical stability
- **Kaiming Initialization**: Applied to classifier head weights (EfficientNet)

### 4.3 Loss Function: Focal Loss with Label Smoothing

To handle severe class imbalance, we implement **Focal Loss** with **Label Smoothing**:

$$\mathcal{L}_{FL} = -\alpha (1 - p_t)^\gamma \log(p_t)$$

Where:
- $\alpha = 1$ (balancing factor)
- $\gamma = 2$ (focusing parameter)
- $p_t$ = probability of correct class

**Label Smoothing**: Targets are smoothed as $y' = y \times 0.9 + 0.05$ to handle noisy labels from NLP extraction and improve generalization.

**Note**: In the updated version, `pos_weight` has been **removed** from FocalLoss to avoid double penalty on rare classes when combined with WeightedRandomSampler.

### 4.4 Handling Class Imbalance: WeightedRandomSampler

To address the severe class imbalance in multi-label data, **WeightedRandomSampler** is implemented:

```python
def get_sample_weights(labels_df, disease_cols):
    class_counts = labels_df[disease_cols].sum(axis=0).values
    total_samples = len(labels_df)
    class_weights = total_samples / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(disease_cols)
    
    # Sample weight = max weight of its positive labels
    sample_weights = []
    for idx in range(len(labels_df)):
        row_labels = labels_df.iloc[idx][disease_cols].values
        positive_indices = np.where(row_labels == 1)[0]
        weight = max(class_weights[positive_indices]) if len(positive_indices) > 0 else 1.0
        sample_weights.append(weight)
    return sample_weights
```

This ensures samples containing rare diseases are sampled more frequently during training.

### 4.5 Training Configuration

| Hyperparameter | DenseNet-121 (A100) | EfficientNet-B0 |
|----------------|---------------------|------------------|
| Epochs | 30 | 20 |
| Batch Size | 256 | 128 |
| Image Size | 384×384 | 256×256 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 6×10⁻⁴ | 5×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ | 1×10⁻⁴ |
| Dropout Rate | 0.3 | 0.3 |
| Label Smoothing | 0.1 | 0.1 |
| LR Scheduler | CosineAnnealingWarmRestarts (per-iteration) | CosineAnnealingWarmRestarts (per-iteration) |
| Scheduler T_0 | iters_per_epoch × 10 | iters_per_epoch × 2 |
| Early Stopping | Patience = 5 epochs | Patience = 5 epochs |
| Random Rotation | ±10° | ±10° |

#### Cross-Validation
- **Strategy**: Group K-Fold (K=5, using first 3 folds)
- **Grouping**: By Patient ID to prevent data leakage
- **Best model selection**: Based on maximum validation AUC-ROC per fold (both models)

#### Optimization Techniques
- **Mixed Precision Training (FP16)**: 2x memory efficiency, faster training
- **TF32 for A100**: Enabled for DenseNet to leverage Tensor Cores
- **Gradient Clipping**: Max norm = 1.0 to prevent exploding gradients
- **CUDA Optimizations**: cuDNN benchmark mode enabled
- **Efficient Data Loading**: `prefetch_factor=2`, increased `num_workers`

### 4.6 Optimal Threshold Tuning

Instead of using a fixed threshold of 0.5, per-class optimal thresholds are computed:

```python
def find_optimal_thresholds(all_labels, all_preds, disease_cols, min_precision=0.15):
    for thresh in np.arange(0.1, 0.61, 0.02):
        # Find threshold that maximizes F1-score
        # while maintaining minimum precision constraint
```

**Benefits**:
- Rare diseases may benefit from lower thresholds (higher recall)
- Common diseases may use higher thresholds (higher precision)
- Constraint `min_precision=0.15` prevents excessive false positives

---

## 5. Results

### 5.1 Training Performance

Both DenseNet-121 and EfficientNet-B0 models were trained using 3-fold cross-validation with the following observations:

- **Convergence**: Models typically converged within 10-15 epochs
- **Learning Curves**: Validation loss closely tracked training loss, indicating good generalization
- **Early Stopping**: Triggered when validation loss showed no improvement for 5 consecutive epochs
- **Model Checkpointing**: Best model weights saved for each fold based on validation loss

### 5.2 Evaluation Metrics

We evaluate using multiple metrics suitable for multi-label classification:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Per-class binary accuracy |
| **AUC-ROC** | Area under ROC curve (ranking quality) |
| **F1-Score** | Harmonic mean of precision and recall |

### 5.3 Per-Disease Performance

Performance varies significantly across disease categories:

| Disease Category | Expected Performance |
|------------------|---------------------|
| High Prevalence (Infiltration, Effusion) | Higher accuracy due to more training samples |
| Low Prevalence (Hernia, Pneumonia) | Lower performance, class imbalance effects |
| Distinct Features (Cardiomegaly) | Better detection due to clear visual patterns |

---

## 6. Updates from Baseline (v0) to Improved Version (2802)

This section documents the key modifications from the baseline implementation (v0) to the improved version (2802) and explains the rationale behind each change.

### 6.1 Removal of pos_weight from FocalLoss

**Change**: The `pos_weight` parameter was **removed** from FocalLoss (commented out).

**Rationale**: 
- In v0, both `pos_weight` (in BCE loss) and `WeightedRandomSampler` were attempting to handle class imbalance simultaneously
- This caused **double penalty** on rare classes, leading to overprediction
- FocalLoss with α=1 and γ=2 already down-weights easy (majority) samples
- Removing `pos_weight` while keeping WeightedRandomSampler provides cleaner separation of concerns

**Impact**: More stable training and reduced overprediction for rare diseases.

### 6.2 Implementation of WeightedRandomSampler

**Change**: Added `get_sample_weights()` function and `WeightedRandomSampler` for data loading.

**Mechanism**:
- Calculate inverse class frequency as class weights
- Assign each sample the **maximum weight** among its positive labels
- Samples with rare diseases get higher sampling probability

**Rationale**:
- Ensures rare diseases are seen more frequently during training
- Works at the data loading level, not loss computation
- Compatible with multi-label setup where one sample may have multiple diseases

**Impact**: Better representation of minority classes during training batches.

### 6.3 Training Configuration Updates

| Parameter | v0 (DenseNet) | 2802 (DenseNet) | v0 (EfficientNet) | 2802 (EfficientNet) |
|-----------|---------------|-----------------|-------------------|---------------------|
| Batch Size | 32 | 256 | 64 | 128 |
| Learning Rate | 3×10⁻⁴ | 6×10⁻⁴ | 5×10⁻⁴ | 5×10⁻⁴ |
| Epochs | 15 | 30 | 30 | 20 |
| Dropout | 0.5 | 0.3 | 0.5 | 0.3 |
| Image Size | 256 | 384 | 256 | 256 |
| Rotation | ±15° | ±10° | ±15° | ±10° |

**Rationale**:
- **Larger batch size**: Better utilizes GPU parallelism (especially A100), more stable gradients
- **Higher learning rate**: Compensates for larger batches, faster convergence
- **Lower dropout (0.3)**: Retains more information during learning; v0's 0.5 was too aggressive
- **Larger image size for DenseNet**: A100's 40GB VRAM allows 384×384, capturing more details
- **Reduced rotation (±10°)**: Medical images have consistent orientation; ±15° was excessive

### 6.4 Label Smoothing Implementation

**Change**: Added `label_smoothing=0.1` to FocalLoss.

**Mechanism**:
```python
if label_smoothing > 0:
    targets = targets * (1 - label_smoothing) + label_smoothing / 2
    # targets = targets * 0.9 + 0.05
```

**Rationale**:
- NIH ChestX-ray14 labels are extracted via NLP from radiology reports
- Labels are inherently **noisy** (estimated 10-20% label error)
- Label smoothing prevents model from being overconfident on potentially incorrect labels
- Improves generalization by softening hard 0/1 targets

**Impact**: More robust model that handles label noise gracefully.

### 6.5 Per-Iteration Scheduler Update

**Change**: Modified scheduler to update per iteration instead of per epoch.

**Before (v0)**:
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)
scheduler.step()  # Called once per epoch
```

**After (2802)**:
```python
iters_per_epoch = len(loader_train)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=iters_per_epoch * 2, T_mult=2, eta_min=1e-6
)
scheduler.step(epoch + batch_idx / iters_per_epoch)  # Called every iteration
```

**Rationale**:
- Smoother learning rate transitions
- More precise warm restart timing
- Better suited for large datasets with many iterations per epoch

### 6.6 Optimal Threshold Tuning

**Change**: Added per-class threshold optimization instead of fixed 0.5.

**Implementation**:
- Search threshold range [0.1, 0.6] with step 0.02
- Maximize F1-score for each class
- Constraint: minimum precision ≥ 15% to prevent false positive explosion

**Rationale**:
- Different diseases have different prevalence and detection difficulty
- Rare diseases (e.g., Hernia, Pneumonia) may benefit from lower thresholds
- Common diseases may use higher thresholds for better precision
- Fixed 0.5 threshold is suboptimal for imbalanced multi-label problems

**Expected Improvement**:
- Macro AUC: Stable (threshold-independent metric)
- Macro F1: Improved due to per-class optimization
- Recall: Increased for rare diseases (lower thresholds)
- Precision: Controlled by min_precision constraint

### 6.7 Model Selection Criterion

**Change**: Both models now select best checkpoint based on **validation AUC-ROC** instead of validation loss.

**Rationale**:
- AUC-ROC is the primary evaluation metric for medical diagnosis
- Validation loss can be misleading with noisy labels
- AUC measures ranking quality, which is more important for clinical decision support


### 6.8 Summary of Expected Improvements

| Metric | v0 Baseline | 2802 Expected | Reason |
|--------|-------------|---------------|--------|
| Macro AUC | ~80-82% | ~82-84% | Better sampling, label smoothing |
| Macro F1 (fixed 0.5) | ~15-20% | ~20-25% | Reduced overprediction |
| Macro F1 (optimal) | N/A | ~25-35% | Per-class threshold tuning |
| Training Stability | Some oscillation | Smoother | Per-iteration scheduler, lower dropout |
| Rare Class Recall | Low | Improved | WeightedRandomSampler |

---