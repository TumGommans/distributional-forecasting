"""Main script tuning and training the QRF model."""

import os
import json

import pandas as pd
import numpy as np
import scoringrules as sr

from typing import Any
from sklearn.model_selection import KFold
from quantile_forest import RandomForestQuantileRegressor
from hyperopt import fmin, tpe, STATUS_OK, Trials

from src.qrf.scripts.sample import sample
from src.utils.utils import (
    load_config, 
    fetch_space_from_config,
    encode_features
)

cfg = load_config(path="/workspace/src/qrf/config/config.yml")
hp_space = fetch_space_from_config(path="/workspace/src/qrf/config/config.yml")

QUANTILES = np.linspace(
    start=cfg['quantiles_to_predict']['low'],
    stop=cfg['quantiles_to_predict']['high'],
    num=cfg['quantiles_to_predict']['nr'],
)

df_x = pd.read_csv("data/X_trn.csv")
df_y = pd.read_csv("data/y_trn.csv")
df = pd.concat([df_y, df_x], axis=1)

df['educcat'] = df['educcat'].map(cfg['education_mapping'])
X = df.drop('realrinc', axis=1)
y = df['realrinc']

def objective(params: dict) -> dict[str, Any]:
    """Objective function for Hyperopt to minimize.

    Args:
        params: dictionary containing a hyperparameter config
    
    Returns
       dict[str, Any]: final loss object 
    """
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['min_samples_split'] = int(params['min_samples_split'])

    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    crps_scores = []

    for fold_num, (train_index, val_index) in enumerate(kfold.split(X)):
        print(f'Starting fold {fold_num + 1}/{n_folds}')
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_eng, X_val_eng = encode_features(X_train, X_val, cfg['all_cat_cols'])

        qrf = RandomForestQuantileRegressor(
            **params,
            random_state=42,
            max_features='sqrt'
        )
        
        qrf.fit(X_train_eng, y_train)
        
        y_pred_quantiles = qrf.predict(X_val_eng, quantiles=QUANTILES.tolist())
        forecasts = sample(
            quantiles=QUANTILES,
            quantile_preds=y_pred_quantiles,
        )

        fold_crps = sr.crps_ensemble(y_val.to_numpy(), forecasts).mean()
        crps_scores.append(fold_crps)

        print(f"Fold {fold_num + 1} CRPS: {fold_crps:.4f}")

    mean_crps = np.mean(crps_scores)
    print(f'Trial Mean CRPS score: {mean_crps:.4f}')
    print("-" * 30)

    return {'loss': mean_crps, 'status': STATUS_OK}

trials = Trials()
best_params = fmin(
    fn=objective,
    space=hp_space,
    algo=tpe.suggest,
    max_evals=cfg['max_evals'],
    trials=trials,
    rstate=np.random.default_rng(42)
)

best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])

best_loss = trials.best_trial['result']['loss']
results_to_save = best_params.copy()
results_to_save['best_loss'] = best_loss

output_dir = "./results/hparams/"
output_path = os.path.join(output_dir, "best_hparams_qrf.json")

os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results_to_save, f, indent=4)
