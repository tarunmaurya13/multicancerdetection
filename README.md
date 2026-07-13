<div align="center">

<!-- Animated Typing Header -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=FF4B91&center=true&vCenter=true&multiline=true&width=900&height=100&lines=%F0%9F%A7%A0+Multi-Cancer+Classification;CNN+%2B+CBAM+%2B+DropBlock+%2B+Mamba+SSM" alt="Typing SVG" /></a>

<br/>

<!-- Animated Wave Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:2b1a2e,100:ff4b91&height=200&section=header&text=CNN-Mamba%20Hybrid%20Architecture&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=24-Class%20Cancer%20Image%20Classification%20with%20State%20Space%20Modeling&descSize=16&descAlignY=55&descColor=f5a3c7" width="100%"/>

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Classes](https://img.shields.io/badge/Classes-24-orange?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</div>

---

## 📌 Project Overview

This project presents a novel **CNN-Mamba Hybrid deep learning architecture** for classifying **24 types of cancer** from medical images. By combining the local feature extraction power of Convolutional Neural Networks (CNNs) with the long-range sequential modeling capabilities of the **Mamba State Space Model (SSM)**, this architecture achieves superior performance over traditional CNN-only approaches.

The model integrates **CBAM (Convolutional Block Attention Module)** for channel and spatial attention, **DropBlock regularization** for robust generalization, and **dual Mamba blocks** to capture both mid-level and high-level spatial relationships — making it one of the most architecturally advanced approaches to multi-cancer image classification.

---

## 🎯 Objectives

- Design a hybrid CNN + Mamba architecture for medical image classification
- Classify 24 distinct cancer types from histopathological images
- Integrate attention mechanisms (CBAM) for improved feature focus
- Apply DropBlock regularization to reduce overfitting
- Evaluate model performance using Accuracy, Confusion Matrix, and ROC-AUC

---

## 🏗️ Model Architecture — CNNMambaHybrid

```
Input (3 × 224 × 224)
        │
   ┌────▼────┐
   │ Block 1  │  Conv2d(3→64) → BN → ReLU → MaxPool
   └────┬────┘
        │
   ┌────▼────┐
   │ Block 2  │  Conv2d(64→128) → BN → ReLU → CBAM → DropBlock(p=0.1) → MaxPool
   └────┬────┘
        │
   ┌────▼────┐
   │ Mamba 1  │  FeatureToSequence → Mamba SSM → LayerNorm → SequenceToFeature
   └────┬────┘     (captures mid-level spatial relationships)
        │
   ┌────▼────┐
   │ Block 3  │  Conv2d(128→256) → BN → ReLU → MaxPool
   └────┬────┘
        │
   ┌────▼────┐
   │ Block 4  │  Conv2d(256→512) → BN → ReLU → CBAM → DropBlock(p=0.2) → MaxPool
   └────┬────┘
        │
   ┌────▼────┐
   │ Mamba 2  │  FeatureToSequence → Mamba SSM → LayerNorm → SequenceToFeature
   └────┬────┘     (captures high-level spatial relationships)
        │
   ┌────▼────┐
   │   Head   │  GlobalAvgPool → Dropout(0.5) → Linear(512 → 24)
   └────┬────┘
        │
   Output (24 classes)
```

### Key Components

| Component | Role |
|---|---|
| **CNN Blocks** | Extract local spatial features at multiple scales |
| **CBAM** | Channel + Spatial attention — focuses on what and where |
| **DropBlock** | Structured regularization — drops contiguous regions |
| **Mamba SSM** | Sequential long-range dependency modeling |
| **Global Avg Pool** | Spatial aggregation before classification |
| **Dropout (0.5)** | Final regularization layer |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `PyTorch` | 2.4.0 | Deep learning framework |
| `TorchVision` | 0.19.0 | Image transforms & pretrained models |
| `Mamba-SSM` | 2.2.2 | State Space Model implementation |
| `causal-conv1d` | 1.4.0 | Efficient causal convolution for Mamba |
| `NumPy` | 2.0.2 | Numerical computations |
| `Matplotlib` | — | Training curves & visualization |
| `Seaborn` | — | Confusion matrix heatmap |
| `Scikit-learn` | — | Metrics — Accuracy, ROC-AUC, Classification Report |
| `tqdm` | — | Training progress bar |
| `PIL` | — | Image loading and preprocessing |
| `CUDA` | 12.1 | GPU acceleration |

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Task** | Multi-class cancer image classification |
| **Classes** | 24 cancer types |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Split** | Train / Validation / Test |
| **Format** | Image folders organized by class |

### Data Preprocessing Pipeline

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | Configurable |
| **Epochs** | Resumable (checkpoint-based) |
| **Device** | CUDA (GPU) / CPU fallback |
| **Checkpointing** | Best model saved per epoch |
| **History** | JSON-based training log |

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Classification Report** | Precision, Recall, F1 per class |
| **Confusion Matrix** | Heatmap of predicted vs actual |
| **ROC-AUC (Weighted, OvR)** | Multi-class discriminability score |
| **ROC Curves** | Per-class True Positive vs False Positive rate |

---

## 📁 File Structure

```
brain-cancer-classification/
│
├── brain.ipynb                  ← Main notebook (model + training + evaluation)
├── checkpoints/
│   ├── best_model.pth           ← Best saved model weights
│   └── checkpoint_epoch_N.pth  ← Per-epoch checkpoints
├── training_history.json        ← Loss & accuracy logs
└── README.md                    ← Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2
```

### 2. Prepare Dataset
```
dataset/
├── class_1/   ← Cancer type 1 images
├── class_2/   ← Cancer type 2 images
...
└── class_24/  ← Cancer type 24 images
```

### 3. Run the Notebook
Open `brain.ipynb` in Google Colab and run cells in order:
- Cell 1 — Install libraries
- Cell 2 — Import dependencies
- Cell 3 — Load & visualize dataset
- Cell 4 — Define model architecture
- Cell 5 — Train / Resume training
- Cell 6 — Evaluate on test set
- Cell 7 — Plot ROC curves

### 4. Resume Training
```python
resume_training_fresh(
    model, train_loader, val_loader,
    criterion, optimizer,
    additional_epochs=5,
    checkpoint_dir='checkpoints'
)
```

---

## 💡 Key Design Decisions

**Why Mamba over Transformer?**
Transformers have quadratic complexity with sequence length. Mamba's selective state space model achieves linear complexity — making it far more efficient for processing high-resolution medical images as sequences.

**Why CBAM?**
Medical images have highly localized diagnostic regions. CBAM's channel attention identifies *what* features matter, while spatial attention identifies *where* they are — critical for pathology detection.

**Why DropBlock over Dropout?**
Standard Dropout drops individual pixels — which are spatially correlated in CNNs and easy to reconstruct. DropBlock drops contiguous regions, forcing the model to learn from distributed representations.

---

## 👤 Author

<div align="center">

**Tarun Maurya**
Final-year BCA (Artificial Intelligence), Invertis University
[GitHub](https://github.com/tarunmaurya13) · tarunmaurya016@gmail.com

</div>

> Built as an advanced deep learning research project combining state-of-the-art architectures — CNN, CBAM Attention, DropBlock, and Mamba SSM — for real-world medical image classification.

---

## 📜 License

This project is open source and available for academic and research purposes.

---

<div align="center">
<i>"The art of medicine consists of amusing the patient while nature cures the disease." — Voltaire</i>
<br/>
<i>This model aims to assist, not replace, medical expertise.</i> 🧠
</div>
