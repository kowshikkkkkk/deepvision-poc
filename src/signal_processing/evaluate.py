import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.image_classification.evaluate import get_predictions
from src.signal_processing.dataset import SIGNAL_CLASSES, _generate_signal, _signal_to_image

def evaluate(model, test_loader, device, model_name="signal_model", save_dir="./results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    preds, labels = get_predictions(model, test_loader, device)
    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    print(f"\n[Evaluate] {model_name} → Test Accuracy: {accuracy:.2f}%")
    print(classification_report(labels, preds, target_names=SIGNAL_CLASSES))
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=SIGNAL_CLASSES, yticklabels=SIGNAL_CLASSES, ax=ax)
    ax.set_title(f"{model_name} — Accuracy: {accuracy:.2f}%")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()
    return {"accuracy": accuracy}