# scripts/train.py

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import TimeSeriesSplit

from src.mdn.data.dataset import TabularDataset
from src.mdn.model.mdn import LogNormalMDN
from src.mdn.loss.nll import gaussian_mdn_loss, log_normal_mdn_loss
from src.mdn.loss.crps import calculate_crps
from src.mdn.utils.trainer import Trainer
from src.mdn.utils.config_parser import load_config

EDUCATION_MAPPING = {
    'Less Than High School': 0,
    'High School': 1,
    'Junior College': 2,
    'Bachelor': 3,
    'Graduate': 4
}

# --- 1. Load Data and Define Constants ---
# This would be your actual data loading step
# For demonstration, we create a dummy time-series dataset
print("Loading data...")
df_x = pd.read_csv("/workspace/data/X_trn.csv")
df_y = pd.read_csv("/workspace/data/y_trn.csv")
df = pd.concat([df_y, df_x], axis=1)

df['educcat'] = df['educcat'].map(EDUCATION_MAPPING)

# Define column names
# In a real scenario, you'd have actual categorical columns
CATEGORICAL_COLS = [
    'occrecode',
    'wrkstat',
    'gender',
    'maritalcat'
]

CONTINUOUS_COLS = [
    'year', 
    'age',
    'prestg10',
    'childs',
    'educcat'
]

TARGET_COL = 'realrinc'
temp_dataset = TabularDataset(df, CATEGORICAL_COLS, CONTINUOUS_COLS, TARGET_COL)
INPUT_DIM = temp_dataset.input_dim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Hyperparameter Search Space ---
print("Loading hyperparameter configuration...")
hp_config = load_config('/workspace/src/mdn/config/hyperparameters.yml')
space = {
    key: (
        # For 'choice', pass the list of options as a single argument
        getattr(hp, value['dist'])(key, value['args'])
        if value['dist'] == 'choice'
        # For everything else (quniform, etc.), unpack the arguments
        else getattr(hp, value['dist'])(key, *value['args'])
    )
    for key, value in hp_config.items()
    if isinstance(value, dict) and 'dist' in value
}

# --- 3. Define the Hyperopt Objective Function ---
def objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    The objective function for Hyperopt. It trains the model using
    time-series cross-validation and returns the mean CRPS.
    """
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=hp_config['cv_splits'])
    fold_scores = []

    for (train_idx, val_idx) in tscv.split(df):
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        # Create datasets and dataloaders
        train_dataset = TabularDataset(train_data, CATEGORICAL_COLS, CONTINUOUS_COLS, TARGET_COL)
        val_dataset = TabularDataset(val_data, CATEGORICAL_COLS, CONTINUOUS_COLS, TARGET_COL)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(params['batch_size']), shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(params['batch_size']))

        # Instantiate model and optimizer
        model = LogNormalMDN(
            input_dim=INPUT_DIM,
            n_gaussians=1, #int(params['n_gaussians']),
            core_hidden_dims=params['core_network_depth'],
            head_hidden_dims=params['head_network_depth']
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Train the model for one fold (silently)
        trainer = Trainer(model, optimizer, log_normal_mdn_loss, DEVICE)
        trainer.train(train_loader, val_loader, epochs=int(params['epochs']), verbose=False)

        # Evaluate using CRPS on validation set
        model.eval()
        val_features, val_targets = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))))
        val_features, val_targets = val_features.to(DEVICE), val_targets.to(DEVICE)
        
        with torch.no_grad():
            pi, mu, sigma = model(val_features)
            crps_score = calculate_crps(pi, mu, sigma, val_targets)
        
        fold_scores.append(crps_score.item())

    mean_crps = np.mean(fold_scores)
    print(f"--- Average CRPS for trial: {mean_crps:.4f} ---")

    return {'loss': mean_crps, 'status': STATUS_OK}

# --- 4. Run Hyperparameter Optimization ---
print("\nStarting hyperparameter optimization with Hyperopt...")
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=hp_config['max_evals'],
    trials=trials
)

print(f"\nOptimization finished. Best parameters found: {best_params}")

print("\nRetraining model on the entire dataset with optimal hyperparameters...")
best_params['core_network_depth'] = hp_config['core_network_depth']['args'][best_params['core_network_depth']]
best_params['head_network_depth'] = hp_config['head_network_depth']['args'][best_params['head_network_depth']]

# Create final dataset and loader
full_dataset = TabularDataset(df, CATEGORICAL_COLS, CONTINUOUS_COLS, TARGET_COL)
full_train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=int(best_params['batch_size']), shuffle=True)

# Instantiate final model and optimizer
final_model = LogNormalMDN(
    input_dim=INPUT_DIM,
    n_gaussians=1, #int(best_params['n_gaussians']),
    core_hidden_dims=best_params['core_network_depth'],
    head_hidden_dims=best_params['head_network_depth']
)
final_optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)

final_trainer = Trainer(final_model, final_optimizer, log_normal_mdn_loss, DEVICE)
final_trainer.train(full_train_loader, val_loader=None, epochs=int(best_params['epochs']), verbose=True)

print("\nSaving final model weights...")
os.makedirs("results", exist_ok=True)
torch.save(final_model.state_dict(), "results/mdn_weights.pth")
print("Model saved successfully to results/mdn_weights.pth")