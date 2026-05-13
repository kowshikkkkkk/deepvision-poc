# DeepVision PoC

> **Deep Learning Proof of Concept** — Image Classification & Signal Processing  
> CNN benchmarking with ResNet-18, VGG-16, and a custom SignalCNN using PyTorch

---

## 📊 Results

| Task | Model | Test Accuracy |
|---|---|---|
| 🔊 Synthetic Signals | SignalCNN | **96.20%** |
| 🖼️ MNIST | ResNet-18 | ~99% |
| 🖼️ CIFAR-10 | ResNet-18 | ~92–94% |
| 🖼️ CIFAR-10 | VGG-16 | ~90–93% |

---

## 🧠 What This Project Does

This PoC benchmarks deep learning models across two domains:

**1. Image Classification**
Trains ResNet-18 and VGG-16 on MNIST (handwritten digits) and CIFAR-10 (objects) using transfer learning from ImageNet pretrained weights.

**2. Signal Processing**
Generates synthetic audio waveforms (sine waves, chirps, FM signals etc.), converts them into spectrogram images using STFT, then classifies them using a lightweight custom CNN — the same pipeline used in real-world audio AI systems.

---

## 📦 Datasets

| Dataset | Size | How |
|---|---|---|
| **MNIST** | 11 MB | Auto-downloads via torchvision |
| **CIFAR-10** | 170 MB | Auto-downloads via torchvision |
| **Synthetic Signals** | 0 MB | Generated in memory — no download |

No manual dataset setup required.

---

## 📂 Project Structure

```
deepvision-poc/
├── src/
│   ├── image_classification/
│   │   ├── dataset.py       # MNIST + CIFAR-10 DataLoaders
│   │   ├── model.py         # ResNet-18, VGG-16 (transfer learning)
│   │   ├── train.py         # Training loop
│   │   └── evaluate.py      # Confusion matrix, training curves
│   ├── signal_processing/
│   │   ├── dataset.py       # Synthetic signal generator + STFT
│   │   ├── model.py         # SignalCNN + ResNet-18 for audio
│   │   ├── train.py         # Delegates to shared train loop
│   │   └── evaluate.py      # Signal evaluation + plots
│   └── utils/
│       ├── augmentation.py  # Data transforms for all datasets
│       └── stats.py         # Reproducibility + result saving
├── results/
│   ├── models/              # Saved .pth checkpoints
│   ├── plots/               # Confusion matrices + training curves
│   └── logs/                # JSON experiment results
├── run_experiments.py       # Master entry point
└── requirements.txt
```

---

## ⚙️ Setup

```bash
git clone https://github.com/kowshikkkkkk/deepvision-poc.git
cd deepvision-poc

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

---

## 🚀 Running Experiments

### Signal Processing (fastest — no download, runs on CPU)
```bash
python run_experiments.py --task signal --model signalcnn --epochs 15
```

### MNIST
```bash
python run_experiments.py --task mnist --model resnet18 --epochs 10
```

### CIFAR-10
```bash
python run_experiments.py --task cifar10 --model resnet18 --epochs 20
python run_experiments.py --task cifar10 --model vgg16 --epochs 20
```

### Run Everything
```bash
python run_experiments.py --task all --epochs 15
```

### Quick Smoke Test (2 epochs)
```bash
python run_experiments.py --task signal --model signalcnn --epochs 2
```

### All Arguments
| Argument | Options | Default |
|---|---|---|
| `--task` | `mnist`, `cifar10`, `signal`, `all` | `signal` |
| `--model` | `resnet18`, `vgg16`, `signalcnn` | `signalcnn` |
| `--mode` | `finetune`, `feature_extract` | `finetune` |
| `--epochs` | any integer | `15` |
| `--lr` | any float | `0.001` |

---

## 🔬 Key Concepts Explained

### 1. Convolutional Neural Networks (CNNs)
CNNs are deep learning models designed for image data. Instead of looking at every pixel individually, they use filters that slide across the image to detect patterns — edges, shapes, textures — at multiple levels of abstraction.

```
Input Image → Conv Layers (feature extraction) → FC Layers (classification) → Output
```

### 2. Transfer Learning
Training a CNN from scratch requires millions of images and days of compute. Transfer learning shortcuts this by starting with a model already trained on ImageNet (1.2M images, 1000 classes) and fine-tuning it for our task.

Two modes used in this project:
- **`finetune`** — all layers retrain on new data → best accuracy
- **`feature_extract`** — backbone frozen, only the final layer trains → faster, good for small datasets

### 3. ResNet-18
ResNet (Residual Network) introduced **skip connections** — shortcuts that let gradients flow directly through layers without vanishing. This solves the problem of deep networks becoming harder to train as they get deeper.

```
x → [Conv → BN → ReLU → Conv → BN] + x → ReLU
          ↑________________________________↑
                   skip connection
```

ResNet-18 has 18 layers and ~11M parameters. In this project, `conv1` is replaced with a 3×3 kernel (instead of 7×7) to handle small 32×32 CIFAR/MNIST inputs.

### 4. VGG-16
VGG-16 uses a simple but deep stack of 3×3 convolutions — 16 layers, ~138M parameters. Heavier than ResNet-18 but strong baseline for benchmarking. `AdaptiveAvgPool` is used here to handle variable input sizes.

### 5. Signal → Image (STFT Spectrogram)
Raw audio is a 1D waveform. To use image CNNs on audio, we convert it to a 2D spectrogram:

```
1D Signal (8000 samples) → STFT → Spectrogram (freq × time) → 64×64 RGB image → CNN
```

**STFT (Short-Time Fourier Transform)** slides a window across the signal and computes the frequency content at each time step. The result is a heatmap where:
- **X-axis** = time
- **Y-axis** = frequency
- **Pixel brightness** = energy at that frequency/time

This is how real-world audio AI works — the CNN "sees" sound as an image.

### 6. Synthetic Signal Classes
Since we generate signals mathematically, each class has distinct frequency characteristics that make them visually separable in spectrogram space:

| Class | Description |
|---|---|
| `pure_sine` | Single clean frequency — flat horizontal line in spectrogram |
| `noisy_sine` | Sine + Gaussian noise — blurry horizontal line |
| `square_wave` | Rich in odd harmonics — multiple horizontal lines |
| `sawtooth_wave` | All harmonics — dense stack of lines |
| `chirp` | Frequency sweep — diagonal line in spectrogram |
| `am_modulated` | Amplitude modulation — symmetric sidebands |
| `fm_modulated` | Frequency modulation — spread sidebands |
| `triangle_wave` | Odd harmonics, fast rolloff — faint upper lines |
| `random_noise` | All frequencies equally — uniform noise pattern |
| `mixed_harmonics` | Multiple tones — multiple bright lines |

### 7. Data Augmentation
Augmentation artificially expands the training set by applying random transformations:

| Dataset | Augmentations Applied |
|---|---|
| MNIST | Random rotation ±10°, random translation |
| CIFAR-10 | Random horizontal flip, random crop, color jitter |
| Signals | Normalization only (signal math handles variation) |

### 8. Training Pipeline
Every experiment uses the same loop:

```
for each epoch:
    → forward pass through model
    → compute CrossEntropyLoss
    → backpropagate gradients
    → AdamW optimizer updates weights
    → CosineAnnealingLR adjusts learning rate
    → save best model if val accuracy improves
    → stop early if no improvement for 7 epochs
```

**AdamW** — adaptive learning rate optimizer with weight decay (prevents overfitting)  
**CosineAnnealingLR** — smoothly reduces learning rate over training, avoids getting stuck  
**Early Stopping** — stops training if validation accuracy doesn't improve, saves time  

### 9. Reproducibility
All experiments use `set_seed(42)` which fixes:
- Python `random`
- NumPy random
- PyTorch CPU and CUDA seeds
- cuDNN deterministic mode

Same seed = same results every run.

### 10. Evaluation Metrics
After training, each model is evaluated with:

- **Accuracy** — overall % correct
- **Precision** — of all predicted class X, how many were actually X
- **Recall** — of all actual class X, how many were predicted correctly
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — grid showing which classes get confused with each other

---

## 📈 Output Files

After running an experiment you'll find:

```
results/
├── models/signal_signalcnn_best.pth      ← best checkpoint
├── plots/signal_signalcnn_confusion_matrix.png
├── plots/signal_signalcnn_training_curves.png
└── logs/signal_signalcnn_results.json
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Model building, training, inference |
| torchvision | Pretrained models, dataset loading, transforms |
| NumPy | Synthetic signal generation, STFT computation |
| scikit-learn | Classification report, confusion matrix |
| matplotlib + seaborn | Training curves, confusion matrix plots |
| TensorBoard | Live training visualization |
| scipy | Statistical validation |

---

