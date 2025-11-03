"""
LinearProbe - Single-layer linear probe for classification tasks.

This module provides LinearProbe class for training, evaluating, and visualizing
linear probes on hidden state activations.
"""

from typing import Dict, Optional
import json
import os
from datetime import datetime

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


class LinearProbe(nn.Module):
    """
    Single-layer linear probe for classification tasks.

    Supports variable loss functions and provides training, evaluation, and visualization.
    """

    def __init__(self, input_dim: int, num_classes: int, loss_fn: str = "cross_entropy"):
        """
        Initialize linear probe.

        Args:
            input_dim: Dimension of input features (hidden_dim * num_layers * num_hook_points)
            num_classes: Number of classes for classification
            loss_fn: Loss function ("cross_entropy" or "mse")
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.loss_fn_name = loss_fn

        # Linear layer
        self.linear = nn.Linear(input_dim, num_classes)

        # Loss function
        if loss_fn == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_fn == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear probe."""
        return self.linear(x)

    def train_probe(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train the linear probe.

        Args:
            activations: Input activations tensor (num_examples, feature_dim)
            labels: Class labels tensor (num_examples,)
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            optimizer_name: Optimizer name ("adam" or "sgd")
            weight_decay: L2 regularization weight
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training history
        """
        activations = activations.to(self.device)
        labels = labels.to(self.device).long()

        # Split into train/val
        n_train = int(len(activations) * (1 - validation_split))
        indices = torch.randperm(len(activations))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_activations = activations[train_indices]
        train_labels = labels[train_indices]
        val_activations = activations[val_indices]
        val_labels = labels[val_indices]

        # Create dataloaders
        train_dataset = TensorDataset(train_activations, train_labels)
        val_dataset = TensorDataset(val_activations, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_acc = 0.0
        best_epoch = 0

        self.train()
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_acts, batch_labels in train_loader:
                optimizer.zero_grad()
                logits = self(batch_acts)
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == batch_labels).sum().item()
                train_total += len(batch_labels)

            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_acts, batch_labels in val_loader:
                    logits = self(batch_acts)
                    loss = self.criterion(logits, batch_labels)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == batch_labels).sum().item()
                    val_total += len(batch_labels)

            self.train()

            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        history["best_val_acc"] = best_val_acc
        history["best_epoch"] = best_epoch

        return history

    def evaluate(self, activations: torch.Tensor, labels: torch.Tensor) -> Dict:
        """
        Evaluate the linear probe.

        Args:
            activations: Input activations tensor
            labels: Class labels tensor

        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        activations = activations.to(self.device)
        labels = labels.to(self.device).long()

        with torch.no_grad():
            logits = self(activations)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(labels_np, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average=None, zero_division=0
        )
        cm = confusion_matrix(labels_np, preds)

        # Per-class metrics
        num_classes = len(np.unique(labels_np))
        precision_dict = {i: float(precision[i]) for i in range(num_classes)}
        recall_dict = {i: float(recall[i]) for i in range(num_classes)}
        f1_dict = {i: float(f1[i]) for i in range(num_classes)}

        metrics = {
            "accuracy": float(accuracy),
            "precision": precision_dict,
            "recall": recall_dict,
            "f1_score": f1_dict,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(labels_np, preds),
        }

        return metrics

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Predict class labels for given activations."""
        self.eval()
        activations = activations.to(self.device)
        with torch.no_grad():
            logits = self(activations)
            preds = logits.argmax(dim=1)
        return preds.cpu()

    def save(self, save_dir: str, prefix: str = "probe") -> str:
        """
        Save the probe and metadata to disk.

        Args:
            save_dir: Directory to save the probe
            prefix: Prefix for saved files

        Returns:
            Path to saved checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(save_dir, f"{prefix}_{timestamp}.pt")
        metadata_path = os.path.join(save_dir, f"{prefix}_{timestamp}_metadata.json")

        # Save model state
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "loss_fn": self.loss_fn_name,
            },
            checkpoint_path,
        )

        # Save metadata
        metadata = {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "loss_fn": self.loss_fn_name,
            "timestamp": timestamp,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Probe saved to {checkpoint_path}")
        print(f"Metadata saved to {metadata_path}")

        return checkpoint_path

    @classmethod
    def load(cls, checkpoint_path: str) -> "LinearProbe":
        """
        Load a probe from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded LinearProbe instance
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        probe = cls(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            loss_fn=checkpoint["loss_fn"],
        )
        probe.load_state_dict(checkpoint["state_dict"])
        return probe

    def visualize(self, training_history: Dict, metrics: Optional[Dict] = None) -> None:
        """
        Visualize training history and evaluation metrics.

        Args:
            training_history: Training history dictionary
            metrics: Optional evaluation metrics dictionary
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"),
        )

        epochs = list(range(1, len(training_history["train_loss"]) + 1))

        # Loss plots
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history["train_loss"], name="Train Loss", mode="lines+markers"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history["val_loss"], name="Val Loss", mode="lines+markers"),
            row=1,
            col=2,
        )

        # Accuracy plots
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history["train_acc"], name="Train Acc", mode="lines+markers"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history["val_acc"], name="Val Acc", mode="lines+markers"),
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        fig.update_layout(
            title="Linear Probe Training History",
            height=800,
            showlegend=True,
        )

        fig.show()

        # Confusion matrix if metrics provided
        if metrics is not None:
            cm = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=[f"Class {i}" for i in range(len(cm))],
                y=[f"Class {i}" for i in range(len(cm))],
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig_cm.show()

            # Print metrics report
            print("\n" + "=" * 50)
            print("Evaluation Metrics")
            print("=" * 50)
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
            print("\nPer-class Metrics:")
            for i in range(len(metrics["precision"])):
                print(
                    f"Class {i}: Precision={metrics['precision'][i]:.4f}, "
                    f"Recall={metrics['recall'][i]:.4f}, F1={metrics['f1_score'][i]:.4f}"
                )
            print("\nClassification Report:")
            print(metrics["classification_report"])

