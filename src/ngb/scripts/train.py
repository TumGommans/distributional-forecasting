"""Main script tuning and training the NGBoost model."""

import os
import json

import pandas as pd
import numpy as np
import scoringrules as sr

from typing import Any
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ngboost.scores import LogScore
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, STATUS_OK, Trials

from src.utils.utils import (
    load_config, 
    fetch_space_from_config,
    encode_features
)

cfg = load_config(path="/workspace/src/ngb/config/config.yml")
hp_space = fetch_space_from_config(path="/workspace/src/ngb/config/config.yml")

df_x = pd.read_csv("data/X_trn.csv")
df_y = pd.read_csv("data/y_trn.csv")
df = pd.concat([df_y, df_x], axis=1)
df = df.sort_values(by='year').reset_index(drop=True)

df['educcat'] = df['educcat'].map(cfg['education_mapping'])
X = df.drop('realrinc', axis=1)
y = df['realrinc']

latest_year = df['year'].max()
cutoff_year = latest_year - cfg['window_length']
ROLLING_WINDOW_SIZE = df[df['year'] > cutoff_year].shape[0]

def objective(params: dict) -> dict[str, Any]:
    """Objective function for Hyperopt to minimize.

    Args:
        params: dictionary containing a hyperparameter config
    
    Returns
       dict[str, Any]: final loss object 
    """
    params['n_estimators'] = int(params['n_estimators'])

    n_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_folds, max_train_size=ROLLING_WINDOW_SIZE)
    crps_scores = []

    for fold_num, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f'Starting fold {fold_num + 1}/{n_folds}')
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_eng, X_val_eng = encode_features(X_train, X_val, cfg['all_cat_cols'])

        ngb = NGBRegressor(
            Dist=LogNormal,
            Score=LogScore,
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            col_sample=params['col_sample'],
            minibatch_frac=params['minibatch_frac'],
            random_state=42,
            verbose=False,
        )

        ngb.fit(X_train_eng, y_train)
        y_pred_dist = ngb.pred_dist(X_val_eng)

        s_params = y_pred_dist.params['s']
        scale_params = y_pred_dist.params['scale']

        mu_params = np.log(scale_params)
        sigma_params = s_params

        forecasts = np.random.lognormal(
            mean=mu_params, 
            sigma=sigma_params, 
            size=(10000, len(y_val))
        ).T

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

best_loss = trials.best_trial['result']['loss']
results_to_save = best_params.copy()
results_to_save['best_loss'] = best_loss

output_dir = "./results/hparams/"
output_path = os.path.join(output_dir, "best_hparams_ngb.json")

os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results_to_save, f, indent=4)
