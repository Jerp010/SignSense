"""
ml/train.py
===========
Training script for ASL MLP classifier.

Usage:
  python -m ml.train
  python -m ml.train --data ml/data/landmarks.csv --epochs 100 --lr 0.001 --batch-size 64

Pipeline:
  1. Load CSV, build label_map from unique labels
  2. Data augmentation during training: noise, x-flip, scale jitter
  3. Train/val split: 85/15 stratified
  4. Train with AdamW, CosineAnnealingLR scheduler
  5. Early stopping: patience=15 on val accuracy
  6. Save checkpoint: {'model_state', 'label_map', 'num_classes'}
  7. Print per-class accuracy table
  8. Plot training curves
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration system
from signsense.ml.config.config_loader import (
    load_config, get_model_config, get_data_config,
    create_parser, load_from_args, set_random_seed, get_device
)
from signsense.utils.logger import logger, TimingContext, log_milestone
from signsense.ml.model import SignMLP, LandmarkNormaliser
from signsense.ml.utils.checkpoint import CheckpointManager
from signsense.ml.utils.metrics import TrainingMetrics


class LandmarkDataset(Dataset):
    """PyTorch dataset for landmark data with optional augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        Args:
            X: features array (n_samples, 63)
            y: labels array (n_samples,)
            augment: whether to apply augmentations during iteration
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        
        if self.augment:
            # Gaussian noise: N(0, 0.005)
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            
            # X-flip with p=0.5 (simulates opposite hand orientation)
            if torch.rand(1).item() < 0.5:
                # Negate x-coordinates (every 3rd element starting from 0)
                x[0::3] = -x[0::3]
            
            # Scale jitter: multiply by Uniform(0.9, 1.1)
            scale_factor = torch.tensor(np.random.uniform(0.9, 1.1))
            x = x * scale_factor
        
        return x, self.y[idx]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    augmentation: bool = True,
    gradient_clip: float = 1.0,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(train_loader.dataset)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate on validation set. Returns (loss, accuracy, predictions, targets)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    
    return val_loss, val_acc, all_preds, all_targets


def main():
    # Parse arguments with config support
    parser = create_parser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Output model path"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load configuration (with CLI overrides)
    config = load_from_args(args)
    model_config = get_model_config(args.model)
    
    # Set random seed
    seed = args.seed if hasattr(args, 'seed') and args.seed else config.get('defaults', {}).get('random_seed', 42)
    set_random_seed(seed)
    
    # Get device
    device_str = args.device if hasattr(args, 'device') and args.device else 'auto'
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    
    # =====================================================================
    # 1. Load data
    # =====================================================================
    data_config = get_data_config('static')
    data_path_str = args.data if args.data else data_config.get('path', 'ml/data/landmarks.csv')
    data_path = Path(data_path_str)
    
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    if len(df) < 100:
        logger.warning(f"Only {len(df)} samples. Consider recording more data (target: 450+ per letter)")
    
    # =====================================================================
    # 2. Build label map
    # =====================================================================
    unique_labels = sorted(df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(label_map)
    
    logger.info(f"Labels: {unique_labels}")
    logger.info(f"Number of classes: {num_classes}")
    logger.debug(f"Label map: {label_map}")
    
    # =====================================================================
    # 3. Prepare features and targets
    # =====================================================================
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].map(label_map).values
    
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Validate data
    if X.shape[1] != 63:
        raise ValueError(f"Expected 63 features, got {X.shape[1]}")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Data contains NaN or Inf values")
    
    # =====================================================================
    # 4. Train/val split
    # =====================================================================
    val_split = data_config.get('validation_split', 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # =====================================================================
    # 5. Create dataloaders
    # =====================================================================
    train_batch_size = model_config.get('training', {}).get('batch_size', 64)
    
    train_dataset = LandmarkDataset(X_train, y_train, augment=True)
    val_dataset = LandmarkDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0
    )
    
    # =====================================================================
    # 6. Model, optimizer, scheduler
    # =====================================================================
    training_config = model_config.get('training', {})
    epochs = training_config.get('epochs', 100)
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    label_smoothing = training_config.get('label_smoothing', 0.1)
    gradient_clip = training_config.get('gradient_clip', 1.0)
    
    # Early stopping config
    early_stopping = training_config.get('early_stopping', {})
    patience = early_stopping.get('patience', 15)
    min_delta = early_stopping.get('min_delta', 0.001)
    
    logger.info(f"\nInitializing model on {device}...")
    model = SignMLP(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # =====================================================================
    # 7. Training loop with early stopping
    # =====================================================================
    log_frequency = 5  # Log every 5 epochs
    
    logger.info(f"\nTraining for {epochs} epochs (patience={patience})...\n")
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=gradient_clip
        )
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        if epoch % log_frequency == 0 or epoch == 1:
            logger.info(
                f"[{epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.6f}"
            )
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_preds = val_preds
            best_targets = val_targets
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break
    
    # =====================================================================
    # 8. Save checkpoint
    # =====================================================================
    output_path = Path(args.output) if args.output else Path("ml/models/sign_mlp.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state": best_model_state if 'best_model_state' in locals() else model.state_dict(),
        "label_map": label_map,
        "num_classes": num_classes,
        "training_date": datetime.now().isoformat(),
        "config": model_config,
    }
    
    torch.save(checkpoint, output_path)
    logger.info(f"\nModel saved to {output_path}")
    
    # =====================================================================
    # 9. Per-class accuracy table
    # =====================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Per-class Accuracy (Validation Set)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Letter':<10} {'Accuracy':<15} {'Samples':<10}")
    logger.info(f"{'-'*60}")
    
    inv_label_map = {v: k for k, v in label_map.items()}
    for class_idx in range(num_classes):
        mask = np.array(best_targets) == class_idx
        if mask.sum() == 0:
            continue
        
        class_acc = accuracy_score(
            np.array(best_targets)[mask],
            np.array(best_preds)[mask]
        )
        class_samples = mask.sum()
        letter = inv_label_map[class_idx]
        logger.info(f"{letter:<10} {class_acc*100:>6.1f}%        {class_samples:<10}")
    
    logger.info(f"{'='*60}")
    log_milestone("Training Complete", {
        "Best Val Accuracy": f"{best_val_acc*100:.2f}%",
        "Epoch": str(best_epoch),
        "Model": str(output_path)
    })
    
    # =====================================================================
    # 10. Plot training curves
    # =====================================================================
    plot_path = output_path.parent / "training_curves.png"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history["val_acc"], label="Val Accuracy", marker="o", markersize=3, color="green")
    ax2.axhline(y=best_val_acc, color="r", linestyle="--", label=f"Best ({best_val_acc*100:.1f}%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    print(f"Training curves saved to {plot_path}\n")


if __name__ == "__main__":
    main()
