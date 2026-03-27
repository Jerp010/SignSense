"""
ml/dynamic_train_enhanced.py
=============================
Training script for enhanced dynamic ASL sign models.

Uses the enhanced LSTM architecture with:
- Bidirectional LSTM
- Multi-head attention
- Residual connections
- Stage-specific features
- Confidence calibration

Usage:
  python -m ml.dynamic_train_enhanced J      # Train enhanced J model
  python -m ml.dynamic_train_enhanced Z      # Train enhanced Z model

Pipeline:
  1. Load all sequences from ml/data/dynamic/<SIGN>/
  2. Group by stage, create labels for each frame
  3. Pad sequences to equal length, create attention masks
  4. Train/val split (80/20)
  5. Train enhanced LSTM with multi-task loss
  6. Early stopping on validation accuracy
  7. Save checkpoint: ml/models/dynamic_<SIGN>_enhanced.pt
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict

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

from signsense.ml.dynamic_model_enhanced import EnhancedDynamicSignLSTM


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


class StageAwareLoss(nn.Module):
    """
    Multi-task loss with stage-specific weighting and transition detection.
    
    Improvements over basic cross-entropy:
    - Stage-specific weights (later stages are harder)
    - Transition detection loss
    - Consistency loss (penalizes stage skipping)
    """
    
    def __init__(self, num_stages: int, stage_weights: Optional[List[float]] = None):
        """
        Args:
            num_stages: Number of stages in the sign
            stage_weights: Optional weights for each stage (default: increasing weights)
        """
        super().__init__()
        self.num_stages = num_stages
        
        # Stage-specific weights (harder stages get higher weight)
        if stage_weights is None:
            # Default: later stages are harder, so weight them more
            self.stage_weights = torch.ones(num_stages)
            for i in range(num_stages):
                self.stage_weights[i] = 1.0 + (i * 0.2)
        else:
            self.stage_weights = torch.tensor(stage_weights)
        
        # Loss components
        self.stage_loss = nn.CrossEntropyLoss(
            weight=self.stage_weights,
            ignore_index=-1  # Ignore padded frames
        )
        self.transition_loss = nn.BCELoss()
        
        # Loss weights
        self.stage_weight = 1.0
        self.transition_weight = 0.3
        self.consistency_weight = 0.1
    
    def forward(
        self,
        stage_logits: torch.Tensor,
        transition_probs: torch.Tensor,
        confidence_scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            stage_logits: [batch, seq_len, num_stages]
            transition_probs: [batch, seq_len, 1]
            confidence_scores: [batch, seq_len, 1]
            labels: [batch, seq_len]
            mask: [batch, seq_len]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        batch_size, seq_len = labels.shape
        
        # Stage classification loss
        stage_logits_flat = stage_logits.reshape(-1, self.num_stages)
        labels_flat = labels.reshape(-1)
        stage_loss = self.stage_loss(stage_logits_flat, labels_flat)
        
        # Transition detection loss
        # Generate transition labels from stage changes
        transition_labels = torch.zeros_like(transition_probs)
        for i in range(batch_size):
            for j in range(1, seq_len):
                if mask[i, j] and mask[i, j-1]:
                    if labels[i, j] != labels[i, j-1]:
                        transition_labels[i, j] = 1.0
        
        transition_loss = self.transition_loss(transition_probs, transition_labels)
        
        # Consistency loss (penalize stage skipping)
        consistency_loss = self.compute_consistency_loss(stage_logits, labels, mask)
        
        # Combined loss
        total_loss = (
            self.stage_weight * stage_loss +
            self.transition_weight * transition_loss +
            self.consistency_weight * consistency_loss
        )
        
        loss_dict = {
            'stage_loss': stage_loss.item(),
            'transition_loss': transition_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def compute_consistency_loss(
        self,
        stage_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize predictions that skip stages.
        
        For example, going from Stage 0 directly to Stage 2 is penalized.
        """
        batch_size, seq_len, _ = stage_logits.shape
        loss = torch.tensor(0.0, device=stage_logits.device)
        
        for i in range(batch_size):
            preds = stage_logits[i].argmax(dim=-1)
            for j in range(1, seq_len):
                if mask[i, j] and mask[i, j-1]:
                    # Check if stage skipped
                    stage_diff = preds[j] - preds[j-1]
                    if stage_diff > 1:  # Skipped a stage
                        loss = loss + (stage_diff - 1)
        
        return (loss / batch_size).detach().clone()


def load_sign_sequences(sign_dir: Path) -> tuple:
    """
    Load all sequences for a sign from disk.
    
    Supports both multi-stage format (SIGN_s0_0.npy) and single-stage format (SIGN_0.npy).
    
    Returns:
        (sequences, labels, num_stages, stage_names)
        - sequences: list of np.ndarray (variable length, 63)
        - labels: list of np.ndarray (stage index per frame)
        - num_stages: total stages for this sign
        - stage_names: dict {stage_idx: stage_name} (currently numeric)
    """
    sequences = []
    labels = []
    
    # Check if multi-stage format exists (files with "_s" in name)
    multi_stage_files = list(sign_dir.glob("*_s*.npy"))
    single_stage_files = [f for f in sign_dir.glob("*.npy") if "_s" not in f.name]
    
    if multi_stage_files:
        # Multi-stage format: SIGN_s0_0.npy, SIGN_s1_0.npy, etc.
        stage_files = {}
        for npy_file in multi_stage_files:
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
        
        logger.info(f"  Loaded {len(sequences)} sequences for {num_stages} stages (multi-stage format)")
        logger.info(f"  Stage file distribution:")
        for stage_idx in sorted(stage_files.keys()):
            logger.info(f"    Stage {stage_idx}: {len(stage_files[stage_idx])} sequences")
    
    elif single_stage_files:
        # Single-stage format: SIGN_0.npy, SIGN_1.npy, etc.
        num_stages = 1
        for npy_path in sorted(single_stage_files):
            seq = np.load(npy_path)  # shape: (num_frames, 63)
            sequences.append(seq)
            
            # Label all frames as stage 0 (single stage)
            stage_labels = np.full(len(seq), 0, dtype=np.int64)
            labels.append(stage_labels)
        
        stage_names = {0: "Stage 0"}
        
        logger.info(f"  Loaded {len(sequences)} sequences for 1 stage (single-stage format)")
    
    else:
        raise ValueError(f"No sequences found in {sign_dir}")
    
    return sequences, labels, num_stages, stage_names


def train_enhanced_dynamic_sign(
    sign_name: str,
    data_dir: str = "ml/data/dynamic",
    output_dir: str = "ml/models",
    args: Optional[argparse.Namespace] = None
):
    """Train enhanced LSTM model for a dynamic sign."""
    
    # Normalize sign name
    sign_name = sign_name.upper()
    
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / data_dir / sign_name
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = get_config(sign_name)
    if config is None:
        unified_config = get_dynamic_sign_config(sign_name)
        if unified_config:
            from signsense.config.dynamic_config import TrainingConfig, DetectorConfig
            training_cfg = unified_config.get('training', {})
            config = type('SignConfig', (), {
                'training': TrainingConfig(**training_cfg),
                'detector': DetectorConfig(**unified_config.get('detector', {})),
                'stages': unified_config.get('stages', {}),
                'name': sign_name
            })()
        else:
            raise ValueError(f"Configuration for sign '{sign_name}' not found")
    
    # Get model config
    model_config = get_model_config('dynamic_lstm')
    training_config = model_config.get('training', {})
    
    # Log training start
    log_milestone(f"Training Enhanced Dynamic Sign: {sign_name}", {
        "Data Directory": str(data_path),
        "Output Directory": str(output_dir),
        "Model Type": "Enhanced LSTM (Bidirectional + Attention)"
    })
    
    # Load sequences
    logger.info(f"Loading sequences from {data_path}...")
    sequences, frame_labels, num_stages, stage_names = load_sign_sequences(data_path)
    
    if len(sequences) < 2:
        logger.error(f"Need at least 2 sequences, got {len(sequences)}")
        return
    
    # Find max sequence length
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
    model = EnhancedDynamicSignLSTM(
        num_stages=num_stages,
        input_size=63,
        hidden_size=hidden_size,
        num_lstm_layers=2,
        num_attention_heads=4,
        dropout=0.3
    )
    model.to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs', 50)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = StageAwareLoss(num_stages=num_stages)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    early_stopping = training_config.get('early_stopping', {})
    patience = early_stopping.get('patience', 15)
    
    best_val_acc = 0
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    loss_components = {'stage_loss': [], 'transition_loss': [], 'consistency_loss': []}
    
    logger.info(f"\nTraining for {epochs} epochs (patience={patience})...")
    logger.info("=" * 70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        epoch_loss_components = {'stage_loss': 0, 'transition_loss': 0, 'consistency_loss': 0}
        
        for batch in train_loader:
            sequences_batch = batch["sequence"].to(device)
            labels_batch = batch["labels"].to(device)
            mask_batch = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            stage_logits, transition_probs, confidence_scores = model(sequences_batch, mask_batch)
            
            # Compute loss
            loss, loss_dict = criterion(
                stage_logits, transition_probs, confidence_scores,
                labels_batch, mask_batch
            )
            
            loss.backward()
            
            # Gradient clipping
            gradient_clip = training_config.get('gradient_clip', 1.0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
            
            train_loss += loss_dict['total_loss']
            for key in epoch_loss_components:
                epoch_loss_components[key] += loss_dict[key]
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        for key in loss_components:
            loss_components[key].append(epoch_loss_components[key] / len(train_loader))
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences_batch = batch["sequence"].to(device)
                labels_batch = batch["labels"].to(device)
                mask_batch = batch["mask"].to(device)
                
                stage_logits, _, _ = model(sequences_batch, mask_batch)
                
                # Get predictions
                preds = stage_logits.argmax(dim=-1)
                
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
            checkpoint_path = output_path / f"dynamic_{sign_name}_enhanced.pt"
            torch.save({
                "model_state": model.state_dict(),
                "num_stages": num_stages,
                "stage_names": stage_names,
                "model_type": "enhanced",
                "hidden_size": hidden_size,
                "total_params": total_params
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
        "Checkpoint": str(output_path / f"dynamic_{sign_name}_enhanced.pt"),
        "Model Type": "Enhanced LSTM"
    })
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Total Loss')
    plt.plot(loss_components['stage_loss'], label='Stage Loss', linestyle='--')
    plt.plot(loss_components['transition_loss'], label='Transition Loss', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {sign_name} (Enhanced)")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Validation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy - {sign_name} (Enhanced)")
    plt.grid(True)
    
    # Plot 3: Learning Rate
    plt.subplot(1, 3, 3)
    lr_values = [optimizer.param_groups[0]['lr'] for _ in range(len(train_losses))]
    # Approximate LR schedule
    lr_values = [learning_rate * (0.5 ** (epoch / epochs)) for epoch in range(len(train_losses))]
    plt.plot(lr_values)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"Learning Rate Schedule - {sign_name}")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / f"training_curves_{sign_name}_enhanced.png")
    logger.info(f"Saved training curves to training_curves_{sign_name}_enhanced.png")


if __name__ == "__main__":
    # Create parser with common arguments
    parser = argparse.ArgumentParser(description="Train enhanced dynamic sign model")
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
    
    train_enhanced_dynamic_sign(args.sign_name.upper(), args.data, args.output, args)
