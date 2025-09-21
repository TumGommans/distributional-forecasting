# src/engine/trainer.py (Modified)

import torch
from torch.utils.data import DataLoader
from typing import Dict, Callable, List

class Trainer:
    """A generic trainer class to handle training and evaluation loops."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str
    ):
        """Initializes the Trainer."""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def _train_step(self, train_loader: DataLoader) -> float:
        """Performs a single training step over the entire training data loader."""
        self.model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            pi, mu, sigma = self.model(features)
            loss = self.loss_fn(pi, mu, sigma, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _eval_step(self, val_loader: DataLoader) -> float:
        """Performs a single evaluation step over the entire validation data loader."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                pi, mu, sigma = self.model(features)
                loss = self.loss_fn(pi, mu, sigma, targets)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """The main training loop."""
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            train_loss = self._train_step(train_loader)
            if val_loader:
                val_loss = self._eval_step(val_loader)
                history['val_loss'].append(val_loss)
            else:
                val_loss = float('nan')
                
            history['train_loss'].append(train_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )
        return history