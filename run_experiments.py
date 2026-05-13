"""
run_experiments.py
-------------------
DeepVision PoC — Master script

Usage:
  python run_experiments.py --task signal --model signalcnn --epochs 15
  python run_experiments.py --task mnist --model resnet18 --epochs 10
  python run_experiments.py --task cifar10 --model resnet18 --epochs 20
  python run_experiments.py --task all --epochs 2
"""

import argparse
import torch
from src.utils.stats import set_seed, save_results


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_best(model, name, device):
    import os
    path = f"./results/models/{name}_best.pth"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  ✓ Loaded: {path}")
    return model.to(device)


def run_mnist(args):
    from src.image_classification.dataset  import get_mnist_loaders, MNIST_CLASSES
    from src.image_classification.model    import get_model
    from src.image_classification.train    import train
    from src.image_classification.evaluate import evaluate, plot_training_history

    print("\n" + "="*50)
    print(f"  MNIST | Model: {args.model.upper()}")
    print("="*50)
    device = _get_device()
    train_loader, val_loader, test_loader = get_mnist_loaders()
    model   = get_model(args.model, num_classes=10, mode=args.mode)
    name    = f"mnist_{args.model}"
    history = train(model, train_loader, val_loader,
                    model_name=name, epochs=args.epochs, lr=args.lr)
    plot_training_history(history, name)
    model = _load_best(model, name, device)
    res   = evaluate(model, test_loader, device, MNIST_CLASSES, name)
    save_results({"task":"mnist","model":args.model,**res},
                 f"./results/logs/{name}_results.json")
    return res


def run_cifar10(args):
    from src.image_classification.dataset  import get_cifar10_loaders, CIFAR10_CLASSES
    from src.image_classification.model    import get_model
    from src.image_classification.train    import train
    from src.image_classification.evaluate import evaluate, plot_training_history

    print("\n" + "="*50)
    print(f"  CIFAR-10 | Model: {args.model.upper()}")
    print("="*50)
    device = _get_device()
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    model   = get_model(args.model, num_classes=10, mode=args.mode)
    name    = f"cifar10_{args.model}"
    history = train(model, train_loader, val_loader,
                    model_name=name, epochs=args.epochs, lr=args.lr)
    plot_training_history(history, name)
    model = _load_best(model, name, device)
    res   = evaluate(model, test_loader, device, CIFAR10_CLASSES, name)
    save_results({"task":"cifar10","model":args.model,**res},
                 f"./results/logs/{name}_results.json")
    return res


def run_signal(args):
    from src.signal_processing.dataset  import get_signal_loaders, SIGNAL_CLASSES
    from src.signal_processing.model    import get_signal_model
    from src.signal_processing.train    import train
    from src.signal_processing.evaluate import evaluate
    from src.image_classification.evaluate import plot_training_history

    print("\n" + "="*50)
    print(f"  SIGNAL | Model: {args.model.upper()}")
    print("="*50)
    device = _get_device()
    train_loader, val_loader, test_loader = get_signal_loaders()
    model   = get_signal_model(args.model, num_classes=10, mode=args.mode)
    name    = f"signal_{args.model}"
    history = train(model, train_loader, val_loader,
                    model_name=name, epochs=args.epochs, lr=args.lr)
    plot_training_history(history, name)
    model = _load_best(model, name, device)
    res   = evaluate(model, test_loader, device, name)
    save_results({"task":"signal","model":args.model,**res},
                 f"./results/logs/{name}_results.json")
    return res


def run_all(args):
    print("\n" + "="*50)
    print("  FULL BENCHMARK — ALL TASKS")
    print("="*50)
    args.model = "resnet18"
    r1 = run_mnist(args)
    r2 = run_cifar10(args)
    args.model = "signalcnn"
    r3 = run_signal(args)
    print("\n" + "-"*40)
    print("  RESULTS SUMMARY")
    print("-"*40)
    print(f"  MNIST     ResNet-18  → {r1['accuracy']:.2f}%")
    print(f"  CIFAR-10  ResNet-18  → {r2['accuracy']:.2f}%")
    print(f"  Signal    SignalCNN  → {r3['accuracy']:.2f}%")
    print("-"*40)


def main():
    parser = argparse.ArgumentParser(description="DeepVision PoC")
    parser.add_argument("--task",       choices=["mnist","cifar10","signal","all"], default="signal")
    parser.add_argument("--model",      choices=["resnet18","vgg16","signalcnn"],   default="signalcnn")
    parser.add_argument("--mode",       choices=["finetune","feature_extract"],     default="finetune")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)
    print(f"[Config] Task: {args.task} | Model: {args.model} | Epochs: {args.epochs}")

    if   args.task == "mnist":   run_mnist(args)
    elif args.task == "cifar10": run_cifar10(args)
    elif args.task == "signal":  run_signal(args)
    elif args.task == "all":     run_all(args)


if __name__ == "__main__":
    main()