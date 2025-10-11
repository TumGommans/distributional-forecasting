"""Main script tuning and training the NGBoost model."""

import os
import json
import pandas as pd
import numpy as np
import scoringrules as sr
from typing import Any

from ngboost import NGBRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from hyperopt import fmin, tpe, STATUS_OK, Trials

from src.ngb.scripts.custom_dist import CustomLogNormal
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

df['educcat'] = df['educcat'].map(cfg['education_mapping'])
X = df.drop('realrinc', axis=1)
y = df['realrinc']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

def objective(params: dict, X_data: pd.DataFrame, y_data: pd.Series) -> dict[str, Any]:
    """Objective function for Hyperopt to minimize using cross-validation."""
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    crps_scores = []

    for fold_num, (train_index, val_index) in enumerate(kfold.split(X_data)):
        print(f'Starting fold {fold_num + 1}/{n_folds}')
        X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
        y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        X_train_eng, X_val_eng = encode_features(X_train, X_val, cfg['all_cat_cols'])

        ngb = NGBRegressor(
            Dist=CustomLogNormal,
            Base=DecisionTreeRegressor(max_depth=params['max_depth']),
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            col_sample=params['col_sample'],
            minibatch_frac=params['minibatch_frac'],
            random_state=42,
            verbose=False,
        )

        ngb.fit(X_train_eng, y_train)
        y_pred_dist = ngb.pred_dist(X_val_eng)

        mu_params = np.log(y_pred_dist.params['scale'])
        sigma_params = y_pred_dist.params['s']
        forecasts = np.random.lognormal(
            mean=mu_params, sigma=sigma_params, size=(10000, len(y_val))
        ).T

        fold_crps = sr.crps_ensemble(y_val.to_numpy(), forecasts).mean()
        crps_scores.append(fold_crps)
        print(f"Fold {fold_num + 1} CRPS: {fold_crps:.4f}")

    mean_crps = np.mean(crps_scores)
    print(f'Trial Mean CRPS score: {mean_crps:.4f}\n' + "-" * 30)
    return {'loss': mean_crps, 'status': STATUS_OK}

trials = Trials()
best_params = fmin(
    fn=lambda params: objective(params, X_train_val, y_train_val),
    space=hp_space,
    algo=tpe.suggest,
    max_evals=cfg['max_evals'],
    trials=trials,
    rstate=np.random.default_rng(42)
)

best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

print("Best hyperparameters found:", best_params)

print("\nRetraining model on the full 90% training data with best hyperparameters...")
X_train_val_eng, X_test_eng = encode_features(X_train_val, X_test, cfg['all_cat_cols'])

final_ngb = NGBRegressor(
    Dist=CustomLogNormal,
    Base=DecisionTreeRegressor(max_depth=best_params['max_depth']),
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    col_sample=best_params['col_sample'],
    minibatch_frac=best_params['minibatch_frac'],
    random_state=42,
)

final_ngb.fit(X_train_val_eng, y_train_val)
y_pred_dist_test = final_ngb.pred_dist(X_test_eng)

mu_params_test = np.log(y_pred_dist_test.params['scale'])
sigma_params_test = y_pred_dist_test.params['s']
forecasts_test = np.random.lognormal(
    mean=mu_params_test, sigma=sigma_params_test, size=(10000, len(y_test))
).T

oos_crps = sr.crps_ensemble(y_test.to_numpy(), forecasts_test).mean()
print(f"Out-of-Sample CRPS on Test Set: {oos_crps:.4f}")

best_loss = trials.best_trial['result']['loss']
results_to_save = best_params.copy()
results_to_save['best_cv_loss'] = best_loss
results_to_save['oos_crps'] = oos_crps

output_dir = "./results/hparams/"
output_path = os.path.join(output_dir, "best_hparams_ngb.json")
os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results_to_save, f, indent=4)

print(f"\nResults saved to {output_path}")
