"""
ml/dynamic_train.py
===================
Training script for dynamic ASL sign models.

Trains LSTM models to recognize multi-stage sign sequences.

Usage:
  python -m ml.dynamic_train J      # Train for J sign
  python -m ml.dynamic_train Z      # Train for Z sign

Pipeline:
  1. Load all sequences from ml/data/dynamic/<SIGN>/
  2. Group by stage, create labels for each frame
  3. Pad sequences to equal length, create attention masks
  4. Train/val split (80/20)
  5. Train LSTM with cross-entropy loss on frames
  6. Early stopping on validation accuracy
  7. Save checkpoint: ml/models/dynamic_<SIGN>.pt
     - Contains: model_state, num_stages, stage_names
  8. Print per-stage accuracy

Configuration:
  All training parameters are configured in ml/config/training.yaml
  Can be overridden via CLI arguments.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
import torch

# Import configuration system
sys.path.append(str(Path(__file__).parent.parent.parent))
from signsense.config.dynamic_config import get_config, get_all_sign_names, SignConfig
from signsense.ml.config.config_loader import (
    load_config, get_model_config, get_dynamic_sign_config,
    create_parser, load_from_args, set_random_seed, get_device
)
from signsense.utils.logger import logger, TimingContext, log_milestone

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from signsense.ml.dynamic_model import DynamicSignLSTM


class DynamicSignDataset(Dataset):
    """PyTorch dataset for dynamic sign sequences with per-frame labels."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, max_len: int):
        """
        Args:
            sequences: List of variable-length sequence arrays
            labels: List of stage labels for each frame in each sequence
            max_len: Maximum sequence length (for padding)
        """
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]  # shape: (seq_len, 63)
        lbl = self.labels[idx]       # shape: (seq_len,)
        
        # Pad sequence to max_len
        seq_len = len(seq)
        if seq_len < self.max_len:
            padding = np.zeros((self.max_len - seq_len, 63), dtype=np.float32)
            seq = np.vstack([seq, padding])
            
            # Extend labels (pad with -1 for masked frames)
            lbl = np.append(lbl, np.full(self.max_len - seq_len, -1, dtype=np.int64))
        
        # Create mask: 1 for real frames, 0 for padded
        mask = np.ones(self.max_len, dtype=np.float32)
        mask[seq_len:] = 0
        
        return {
            "sequence": torch.from_numpy(seq).float(),
            "labels": torch.from_numpy(lbl).long(),
            "mask": torch.from_numpy(mask).float()
        }


def load_sign_sequences(sign_dir: Path) -> tuple:
    """
    Load all sequences for a sign from disk.
    
    Returns:
        (sequences, labels, num_stages, stage_names)
        - sequences: list of np.ndarray (variable length, 63)
        - labels: list of np.ndarray (stage index per frame)
        - num_stages: total stages for this sign
        - stage_names: dict {stage_idx: stage_name} (currently numeric)
    """
    sequences = []
    labels = []
    
    # Find all stage files
    stage_files = {}
    for npy_file in sign_dir.glob("*.npy"):
        if "_s" not in npy_file.name:
            continue
        
        # Parse filename: SIGN_s0_0.npy -> stage 0
        parts = npy_file.name.split("_s")
        if len(parts) < 2:
            continue
        
        stage_num = int(parts[1].split("_")[0])
        
        if stage_num not in stage_files:
            stage_files[stage_num] = []
        stage_files[stage_num].append(npy_file)
    
    num_stages = len(stage_files)
    if num_stages == 0:
        raise ValueError(f"No stage sequences found in {sign_dir}")
    
    # Load sequences for each stage
    for stage_idx in sorted(stage_files.keys()):
        for npy_path in stage_files[stage_idx]:
            seq = np.load(npy_path)  # shape: (num_frames, 63)
            sequences.append(seq)
            
            # Label each frame with its stage
            stage_labels = np.full(len(seq), stage_idx, dtype=np.int64)
            labels.append(stage_labels)
    
    stage_names = {i: f"Stage {i}" for i in range(num_stages)}
    
    logger.info(f"  Loaded {len(sequences)} sequences for {num_stages} stages")
    logger.info(f"  Stage file distribution:")
    for stage_idx in sorted(stage_files.keys()):
        logger.info(f"    Stage {stage_idx}: {len(stage_files[stage_idx])} sequences")
    
    return sequences, labels, num_stages, stage_names


def train_dynamic_sign(
    sign_name: str, 
    data_dir: str = "ml/data/dynamic",
    output_dir: str = "ml/models",
    args: Optional[argparse.Namespace] = None
):
    """Train LSTM model for a dynamic sign using configuration."""
    
    # Normalize sign name
    sign_name = sign_name.upper()
    
    # Setup paths - resolve relative to project root, not current directory
    # Get project root (parent of signsense package)
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / data_dir / sign_name
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = get_config(sign_name)
    if config is None:
        # Try loading from unified config
        unified_config = get_dynamic_sign_config(sign_name)
        if unified_config:
            # Create config object from unified config
            from signsense.config.dynamic_config import TrainingConfig, DetectorConfig
            training_cfg = unified_config.get('training', {})
            config = type('SignConfig', (), {
                'training': TrainingConfig(**training_cfg),
                'detector': DetectorConfig(**unified_config.get('detector', {})),
                'stages': unified_config.get('stages', {}),
                'name': sign_name
            })()
        else:
            raise ValueError(f"Configuration for sign '{sign_name}' not found in dynamic_signs.yaml")
    
    # Get model config
    model_config = get_model_config('dynamic_lstm')
    training_config = model_config.get('training', {})
    
    # Log training start
    log_milestone(f"Training Dynamic Sign: {sign_name}", {
        "Data Directory": str(data_path),
        "Output Directory": str(output_dir)
    })
    
    # Load sequences
    logger.info(f"Loading sequences from {data_path}...")
    sequences, frame_labels, num_stages, stage_names = load_sign_sequences(data_path)
    
    if len(sequences) < 2:
        logger.error(f"Need at least 2 sequences, got {len(sequences)}")
        return
    
    # Find max sequence length (use config or actual max)
    max_len = max(len(seq) for seq in sequences)
    if hasattr(config.training, 'sequence_length') and config.training.sequence_length > max_len:
        max_len = config.training.sequence_length
    elif training_config.get('sequence', {}).get('max_length', 120) > max_len:
        max_len = training_config.get('sequence', {}).get('max_length', 120)
    logger.info(f"Max sequence length: {max_len} frames")
    
    # Create dataset
    dataset = DynamicSignDataset(sequences, frame_labels, max_len)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = training_config.get('batch_size', 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Setup model
    device = get_device()
    logger.info(f"Device: {device}")
    
    hidden_size = training_config.get('architecture', {}).get('hidden_size', 128)
    model = DynamicSignLSTM(num_stages, input_size=63, hidden_size=hidden_size)
    model.to(device)
    
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs', 50)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded frames
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    early_stopping = training_config.get('early_stopping', {})
    patience = early_stopping.get('patience', 15)
    
    best_val_acc = 0
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    
    logger.info(f"\nTraining for {epochs} epochs (patience={patience})...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            sequences_batch = batch["sequence"].to(device)
            labels_batch = batch["labels"].to(device)
            mask_batch = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            stage_logits, transition_probs = model(sequences_batch)
            # stage_logits: [batch, seq_len, num_stages]
            # labels_batch: [batch, seq_len]
            
            # Flatten for cross-entropy
            batch_size, seq_len = labels_batch.shape
            stage_logits_flat = stage_logits.reshape(batch_size * seq_len, num_stages)
            labels_flat = labels_batch.reshape(batch_size * seq_len)
            
            loss = criterion(stage_logits_flat, labels_flat)
            loss.backward()
            
            # Gradient clipping
            gradient_clip = training_config.get('gradient_clip', 1.0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences_batch = batch["sequence"].to(device)
                labels_batch = batch["labels"].to(device)
                
                stage_logits, _ = model(sequences_batch)
                
                # Get predictions
                preds = stage_logits.argmax(dim=-1)  # [batch, seq_len]
                
                # Flatten and filter out masked (-1) frames
                batch_size, seq_len = labels_batch.shape
                preds_flat = preds.cpu().numpy().reshape(-1)
                labels_flat = labels_batch.cpu().numpy().reshape(-1)
                
                mask_val = labels_flat != -1
                all_preds.extend(preds_flat[mask_val])
                all_labels.extend(labels_flat[mask_val])
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = output_path / f"dynamic_{sign_name}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "num_stages": num_stages,
                "stage_names": stage_names
            }, checkpoint_path)
            
            logger.info(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} * (saved)")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Print final stats
    log_milestone("Training Complete", {
        "Best Val Accuracy": f"{best_val_acc:.4f}",
        "Checkpoint": str(output_path / f"dynamic_{sign_name}.pt")
    })
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {sign_name}")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy - {sign_name}")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / f"training_curves_{sign_name}.png")
    logger.info(f"Saved training curves to training_curves_{sign_name}.png")


if __name__ == "__main__":
    # Create parser with common arguments
    parser = argparse.ArgumentParser(description="Train dynamic sign model")
    parser.add_argument("sign_name", help="Sign name to train (e.g., J, Z)")
    parser.add_argument("--data", default="ml/data/dynamic", help="Data directory")
    parser.add_argument("--output", default="ml/models", help="Output directory")
    
    # Add common training arguments
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], help="Device to use")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        from signsense.ml.config.config_loader import set_random_seed
        set_random_seed(args.seed)
    
    train_dynamic_sign(args.sign_name.upper(), args.data, args.output, args)
