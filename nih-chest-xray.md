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
- **Split ratio**: 80% train, 20% test
- **Strategy**: Patient-level split to prevent data leakage
- Ensures no patient appears in both train and test sets

#### Image Transformations

**Training Augmentations:**
```
- Resize to 256×256
- Random Rotation (±15°)
- Convert to Tensor
- Normalization (ImageNet mean/std: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

**Test Transformations:**
```
- Resize to 256×256
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
    → Linear(1024, 512) → BatchNorm → ReLU → Dropout(0.5)
    → Linear(512, 256) → BatchNorm → ReLU → Dropout(0.3)
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
    → Linear(1280, 512) → BatchNorm → SiLU → Dropout(0.5)
    → Linear(512, 256) → BatchNorm → SiLU → Dropout(0.3)
    → Linear(256, 14)
```

**Transfer Learning Strategy**: Freeze features.0 through features.4, fine-tune remaining layers

#### Common Design Choices

- **BatchNorm**: Stabilizes training, acts as regularization
- **Progressive Dropout**: Higher dropout early (0.5), lower later (0.3)
- **No final Sigmoid**: Using BCEWithLogitsLoss for numerical stability
- **Kaiming Initialization**: Applied to classifier head weights (EfficientNet)

### 4.3 Loss Function: Focal Loss

To handle severe class imbalance, we implement **Focal Loss**:

$$\mathcal{L}_{FL} = -\alpha (1 - p_t)^\gamma \log(p_t)$$

Where:
- $\alpha = 1$ (balancing factor)
- $\gamma = 2$ (focusing parameter)
- $p_t$ = probability of correct class

**Class Weights**: Computed as $(N - N_{pos}) / N_{pos}$ to up-weight rare diseases.

### 4.4 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 15 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| LR Scheduler | Cosine Annealing with Warm Restarts |
| Early Stopping | Patience = 5 epochs |

#### Cross-Validation
- **Strategy**: Group K-Fold (K=5, using first 3 folds)
- **Grouping**: By Patient ID to prevent data leakage
- **Best model selection**: Based on minimum validation loss per fold

#### Optimization Techniques
- **Mixed Precision Training (FP16)**: 2x memory efficiency, faster training
- **Gradient Clipping**: Max norm = 1.0 to prevent exploding gradients
- **CUDA Optimizations**: cuDNN benchmark mode enabled

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