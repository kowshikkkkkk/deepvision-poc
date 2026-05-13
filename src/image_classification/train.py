import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.stats import set_seed

def _run_epoch(model, loader, optimizer, criterion, device, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="  Train" if training else "  Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += images.size(0)
    return {"loss": total_loss/total, "acc": 100.0*correct/total}

def train(model, train_loader, val_loader, model_name="model",
          epochs=15, lr=1e-3, weight_decay=1e-4, patience=7,
          save_dir="./results/models", log_dir="./results/logs", seed=42):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    model  = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer    = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    history   = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc, no_improve = 0.0, 0

    for epoch in range(1, epochs+1):
        t0  = time.time()
        tr  = _run_epoch(model, train_loader, optimizer, criterion, device, True)
        val = _run_epoch(model, val_loader,   optimizer, criterion, device, False)
        scheduler.step()
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        writer.add_scalars("Loss",     {"train": tr["loss"],  "val": val["loss"]},  epoch)
        writer.add_scalars("Accuracy", {"train": tr["acc"],   "val": val["acc"]},   epoch)
        print(f"Epoch [{epoch:02d}/{epochs}] Train Loss: {tr['loss']:.4f} Acc: {tr['acc']:.2f}% | "
              f"Val Loss: {val['loss']:.4f} Acc: {val['acc']:.2f}% | {time.time()-t0:.1f}s")
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            no_improve   = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": best_val_acc},
                       os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f"  ✓ Best saved ({best_val_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStopping] Triggered.")
                break
    writer.close()
    print(f"\n[Train] Done. Best val acc: {best_val_acc:.2f}%")
    return history