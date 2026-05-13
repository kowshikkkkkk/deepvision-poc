import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    for images, lbls in loader:
        out = model(images.to(device))
        preds.extend(out.max(1)[1].cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)

def evaluate(model, test_loader, device, class_names, model_name="model", save_dir="./results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    preds, labels = get_predictions(model, test_loader, device)
    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    print(f"\n[Evaluate] {model_name} → Test Accuracy: {accuracy:.2f}%")
    print(classification_report(labels, preds, target_names=class_names))
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{model_name} — Accuracy: {accuracy:.2f}%")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()
    return {"accuracy": accuracy}

def plot_training_history(history, model_name, save_dir="./results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history["train_loss"], "b-o", ms=3, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", ms=3, label="Val")
    ax1.set_title(f"{model_name} — Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"],  "b-o", ms=3, label="Train")
    ax2.plot(epochs, history["val_acc"],    "r-o", ms=3, label="Val")
    ax2.set_title(f"{model_name} — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_training_curves.png"), dpi=150)
    plt.close()
    print(f"  ✓ Training curves saved")