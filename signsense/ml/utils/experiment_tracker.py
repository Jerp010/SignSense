"""
ml/utils/experiment_tracker.py
=============================
Experiment tracking and logging for SignSense training.

Tracks experiments with:
- Configuration snapshots
- Metrics over time
- Model artifacts
- Training curves
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import pandas as pd
import torch


@dataclass
class Experiment:
    """
    Represents a single training experiment.
    
    Attributes:
        id: Unique experiment identifier
        name: Human-readable experiment name
        timestamp: Experiment creation timestamp
        config: Configuration dictionary
        metrics: Metrics recorded during training
        artifacts: Paths to saved artifacts
        status: Experiment status (running, completed, failed)
    """
    id: str
    name: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    status: str = "running"
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    @staticmethod
    def create(name: str, config: Dict[str, Any]) -> "Experiment":
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
        
        Returns:
            New Experiment instance
        """
        # Generate unique ID from config hash
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp}_{config_hash}"
        
        return Experiment(
            id=exp_id,
            name=name,
            timestamp=timestamp,
            config=config,
            status="running"
        )


class ExperimentTracker:
    """
    Tracks and manages training experiments.
    
    Features:
    - Create and manage experiments
    - Log metrics during training
    - Save and load experiment state
    - Generate comparison reports
    
    Usage:
        tracker = ExperimentTracker("ml/experiments")
        
        # Start new experiment
        exp = tracker.start_experiment(
            name="baseline",
            config={"lr": 0.001, "batch_size": 64}
        )
        
        # Log metrics
        tracker.log_metrics(exp.id, {
            "train_loss": 0.5,
            "val_acc": 0.9
        })
        
        # Complete experiment
        tracker.complete_experiment(exp.id)
        
        # List experiments
        experiments = tracker.list_experiments()
    """
    
    def __init__(self, experiment_dir: str = "ml/experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for active experiments
        self.experiments: Dict[str, Experiment] = {}
        
        # Load existing experiments
        self._load_experiments()
    
    def _get_experiment_path(self, exp_id: str) -> Path:
        """Get path to experiment directory."""
        return self.experiment_dir / exp_id
    
    def _load_experiments(self):
        """Load existing experiments from disk."""
        if not self.experiment_dir.exists():
            return
        
        for exp_dir in self.experiment_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_info_file = exp_dir / "experiment.json"
            if not exp_info_file.exists():
                continue
            
            try:
                with open(exp_info_file, 'r') as f:
                    data = json.load(f)
                    exp = Experiment(**data)
                    self.experiments[exp.id] = exp
            except Exception as e:
                print(f"Warning: Could not load experiment {exp_dir.name}: {e}")
    
    def start_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        overwrite: bool = False
    ) -> Experiment:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
            overwrite: Whether to overwrite existing experiment with same name
        
        Returns:
            New Experiment instance
        """
        # Check for existing experiment with same name
        existing = self.get_experiment_by_name(name)
        if existing and not overwrite:
            raise ValueError(
                f"Experiment with name '{name}' already exists. "
                f"Use overwrite=True to replace."
            )
        
        # Create experiment
        experiment = Experiment.create(name, config)
        
        # Create experiment directory
        exp_dir = self._get_experiment_path(experiment.id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        self._save_experiment(experiment)
        
        # Store in memory
        self.experiments[experiment.id] = experiment
        
        return experiment
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk."""
        exp_dir = self._get_experiment_path(experiment.id)
        
        # Save experiment metadata
        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(asdict(experiment), f, indent=2)
        
        # Save config
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(experiment.config, f, indent=2)
        
        # Save metrics as CSV if any
        if experiment.metrics:
            metrics_df = pd.DataFrame(experiment.metrics)
            metrics_df.to_csv(exp_dir / "metrics.csv", index=False)
    
    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(exp_id)
    
    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        for exp in self.experiments.values():
            if exp.name == name:
                return exp
        return None
    
    def log_metrics(
        self,
        exp_id: str,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None
    ):
        """
        Log metrics for an experiment.
        
        Args:
            exp_id: Experiment ID
            metrics: Dictionary of metric names and values
            step: Optional step number (epoch)
        """
        experiment = self.experiments.get(exp_id)
        if not experiment:
            print(f"Warning: Experiment {exp_id} not found")
            return
        
        # Initialize metric lists if needed
        for metric_name in metrics.keys():
            if metric_name not in experiment.metrics:
                experiment.metrics[metric_name] = []
        
        # Append metrics
        for metric_name, value in metrics.items():
            experiment.metrics[metric_name].append(float(value))
        
        # Save updated metrics
        self._save_experiment(experiment)
    
    def log_metric(
        self,
        exp_id: str,
        metric_name: str,
        value: Union[float, int],
        step: Optional[int] = None
    ):
        """
        Log a single metric for an experiment.
        
        Args:
            exp_id: Experiment ID
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number
        """
        self.log_metrics(exp_id, {metric_name: value}, step)
    
    def complete_experiment(
        self,
        exp_id: str,
        status: str = "completed",
        error: Optional[str] = None
    ):
        """
        Mark an experiment as complete.
        
        Args:
            exp_id: Experiment ID
            status: Final status (completed, failed, stopped)
            error: Optional error message
        """
        experiment = self.experiments.get(exp_id)
        if not experiment:
            return
        
        experiment.status = status
        experiment.completed_at = datetime.now().isoformat()
        experiment.error = error
        
        self._save_experiment(experiment)
    
    def fail_experiment(self, exp_id: str, error: str):
        """Mark an experiment as failed with an error message."""
        self.complete_experiment(exp_id, status="failed", error=error)
    
    def add_artifact(self, exp_id: str, name: str, path: str):
        """
        Add an artifact to an experiment.
        
        Args:
            exp_id: Experiment ID
            name: Artifact name
            path: Path to artifact
        """
        experiment = self.experiments.get(exp_id)
        if not experiment:
            return
        
        experiment.artifacts[name] = path
        self._save_experiment(experiment)
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Experiment]:
        """
        List experiments, optionally filtered by status.
        
        Args:
            status: Optional status filter
            limit: Maximum number of experiments to return
        
        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())
        
        # Filter by status
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def get_best_experiment(
        self,
        metric: str = "val_acc",
        mode: str = "max"
    ) -> Optional[Experiment]:
        """
        Get the experiment with the best metric value.
        
        Args:
            metric: Metric name to compare
            mode: "max" or "min"
        
        Returns:
            Best experiment or None
        """
        best_exp = None
        best_value = float('-inf') if mode == "max" else float('inf')
        
        for exp in self.experiments.values():
            if exp.status != "completed":
                continue
            
            if metric not in exp.metrics or not exp.metrics[metric]:
                continue
            
            # Get final value (last epoch)
            values = exp.metrics[metric]
            value = values[-1]
            
            if (mode == "max" and value > best_value) or \
               (mode == "min" and value < best_value):
                best_value = value
                best_exp = exp
        
        return best_exp
    
    def compare_experiments(
        self,
        metric: str = "val_acc",
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Compare experiments by a specific metric.
        
        Args:
            metric: Metric to compare
            limit: Number of experiments to include
        
        Returns:
            DataFrame with experiment comparison
        """
        experiments = self.list_experiments(limit=limit)
        
        comparison = []
        for exp in experiments:
            if metric not in exp.metrics or not exp.metrics[metric]:
                continue
            
            values = exp.metrics[metric]
            
            comparison.append({
                "experiment_id": exp.id,
                "name": exp.name,
                "status": exp.status,
                f"{metric}_final": values[-1],
                f"{metric}_best": max(values),
                f"{metric}_mean": sum(values) / len(values),
                "epochs": len(values)
            })
        
        return pd.DataFrame(comparison)
    
    def delete_experiment(self, exp_id: str):
        """
        Delete an experiment.
        
        Args:
            exp_id: Experiment ID to delete
        """
        if exp_id not in self.experiments:
            return
        
        # Remove from memory
        del self.experiments[exp_id]
        
        # Remove from disk
        exp_dir = self._get_experiment_path(exp_id)
        if exp_dir.exists():
            import shutil
            shutil.rmtree(exp_dir)
    
    def export_report(self, output_path: str = "experiment_report.md"):
        """
        Export a markdown report of all experiments.
        
        Args:
            output_path: Path to save report
        """
        experiments = self.list_experiments()
        
        lines = [
            "# SignSense Experiment Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"Total Experiments: {len(experiments)}",
            ""
        ]
        
        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Name | Status | Best Val Acc | Epochs |")
        lines.append("|------|--------|--------------|--------|")
        
        for exp in experiments:
            val_acc = exp.metrics.get("val_acc", [])
            best_acc = max(val_acc) if val_acc else "N/A"
            epochs = len(val_acc)
            
            lines.append(
                f"| {exp.name} | {exp.status} | {best_acc} | {epochs} |"
            )
        
        lines.append("")
        
        # Best experiment
        best = self.get_best_experiment("val_acc", "max")
        if best:
            lines.append("## Best Experiment")
            lines.append("")
            lines.append(f"**{best.name}** (ID: {best.id})")
            lines.append("")
            lines.append("### Configuration")
            lines.append("```json")
            lines.append(json.dumps(best.config, indent=2))
            lines.append("```")
        
        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"Report saved to {output_path}")


# Global experiment tracker instance
_experiment_tracker: Optional[ExperimentTracker] = None


def get_tracker(experiment_dir: str = "ml/experiments") -> ExperimentTracker:
    """Get or create the global experiment tracker."""
    global _experiment_tracker
    
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker(experiment_dir)
    
    return _experiment_tracker
