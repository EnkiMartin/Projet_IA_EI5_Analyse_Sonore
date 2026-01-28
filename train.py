# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import os

from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='scratch', choices=['scratch', 'torchvision'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, valid_loader = get_data(batch_size=args.batch_size)
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    print("Classes:", classes)

    # Model
    if args.model == 'scratch':
        model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes).to(device)
        plot_name = 'resnet_scratch'
    else:
        model = build_model(pretrained=False, fine_tune=True, num_classes=num_classes).to(device)
        plot_name = 'resnet_torchvision'

    total_params, total_trainable_params = count_params(model)
    print(f"{total_params:,} total parameters.")
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer / Loss / Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ✅ Nouveau module AMP (corrigé)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Entraînement
    os.makedirs('outputs', exist_ok=True)
    best_acc = 0.0
    bad_epochs = 0
    train_loss_hist, valid_loss_hist = [], []
    train_acc_hist, valid_acc_hist = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"[INFO]: Epoch {epoch} of {args.epochs}")

        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion,
            device, scaler=scaler, use_amp=True
        )

        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion,
            device, use_amp=True
        )

        train_loss_hist.append(train_epoch_loss)
        valid_loss_hist.append(valid_epoch_loss)
        train_acc_hist.append(train_epoch_acc)
        valid_acc_hist.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        # Save best + early stop
        improved = valid_epoch_acc > best_acc
        if improved:
            best_acc = valid_epoch_acc
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'classes': classes, 'args': vars(args)}, 'outputs/best_model.pt')
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop: no val improvement in {args.patience} epochs.")
                break

        scheduler.step()

    save_plots(train_acc_hist, valid_acc_hist, train_loss_hist, valid_loss_hist, name=plot_name)
    print('TRAINING COMPLETE')

if __name__ == '__main__':
    main()
