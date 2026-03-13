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
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Import configuration system
sys.path.append(str(Path(__file__).parent.parent.parent))
from signsense.config.dynamic_config import get_config, get_all_sign_names, SignConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    
    print(f"  Loaded {len(sequences)} sequences for {num_stages} stages")
    print(f"  Stage file distribution:")
    for stage_idx in sorted(stage_files.keys()):
        print(f"    Stage {stage_idx}: {len(stage_files[stage_idx])} sequences")
    
    return sequences, labels, num_stages, stage_names


def train_dynamic_sign(sign_name: str, data_dir: str = "ml/data/dynamic", 
                      output_dir: str = "ml/models"):
    """Train LSTM model for a dynamic sign using configuration."""
    
    sign_name = sign_name.upper()
    data_path = Path(data_dir) / sign_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = get_config(sign_name)
    if config is None:
        raise ValueError(f"Configuration for sign '{sign_name}' not found in dynamic_signs.yaml")
    
    print(f"\n{'='*60}")
    print(f"Training Dynamic Sign: {sign_name}")
    print(f"{'='*60}")
    
    # Load sequences
    print(f"\nLoading sequences from {data_path}...")
    sequences, frame_labels, num_stages, stage_names = load_sign_sequences(data_path)
    
    if len(sequences) < 2:
        print(f"Error: Need at least 2 sequences, got {len(sequences)}")
        return
    
    # Find max sequence length (use config or actual max)
    max_len = max(len(seq) for seq in sequences)
    if config.training.sequence_length > max_len:
        max_len = config.training.sequence_length
    print(f"Max sequence length: {max_len} frames")
    
    # Create dataset
    dataset = DynamicSignDataset(sequences, frame_labels, max_len)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = DynamicSignLSTM(num_stages, input_size=63, hidden_size=config.training.hidden_size)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded frames
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    
    # Training loop
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    
    print("\nTraining...")
    for epoch in range(config.training.epochs):
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
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} ✓ (saved)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Print final stats
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {output_path / f'dynamic_{sign_name}.pt'}")
    print(f"{'='*60}\n")
    
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
    print(f"Saved training curves to training_curves_{sign_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dynamic sign model")
    parser.add_argument("sign_name", help="Sign name to train (e.g., J, Z)")
    parser.add_argument("--data", default="ml/data/dynamic", help="Data directory")
    parser.add_argument("--output", default="ml/models", help="Output directory")
    
    args = parser.parse_args()
    
    train_dynamic_sign(args.sign_name, args.data, args.output)
