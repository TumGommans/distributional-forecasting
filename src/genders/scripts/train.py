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

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, STATUS_OK, Trials

from src.utils.utils import (
    load_config, 
    fetch_space_from_config
)

df_1980 = pd.DataFrame(
    [
        {'year': 1980, 'gender': 1},
        {'year': 1980, 'gender': 0},
    ]
)
df_2010 = pd.DataFrame(
    [
        {'year': 2010, 'gender': 1},
        {'year': 2010, 'gender': 0},
    ]
)

cfg = load_config(path="/workspace/src/genders/config/config.yml")
hp_space = fetch_space_from_config(path="/workspace/src/genders/config/config.yml")

df_x = pd.read_csv("data/X_trn.csv")
df_y = pd.read_csv("data/y_trn.csv")
df = pd.concat([df_y, df_x], axis=1)

X = df[['year', 'gender']]
X['gender'] = X['gender'].map(cfg['gender_mapping'])
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

    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    crps_scores = []

    for fold_num, (train_index, val_index) in enumerate(kfold.split(X)):
        print(f'Starting fold {fold_num + 1}/{n_folds}')
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        ngb = NGBRegressor(
            Dist=LogNormal,
            Score=LogScore,
            Base=DecisionTreeRegressor(max_depth=params['max_depth']),
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            col_sample=params['col_sample'],
            minibatch_frac=params['minibatch_frac'],
            random_state=42,
            verbose=False,
        )

        ngb.fit(
            X_train, 
            y_train
        )

        y_pred_dist = ngb.pred_dist(X_val)

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
best_params['max_depth'] = int(best_params['max_depth'])

best_loss = trials.best_trial['result']['loss']
results_to_save = best_params.copy()
results_to_save['best_loss'] = best_loss

output_dir = "./results/hparams/"
output_path = os.path.join(output_dir, "best_hparams_ngb_gender.json")

os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results_to_save, f, indent=4)

final_model = NGBRegressor(
    Dist=LogNormal,
    Score=LogScore,
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    col_sample=best_params['col_sample'],
    minibatch_frac=best_params['minibatch_frac'],
    random_state=42
)
final_model.fit(X, y)

gender_dist_1980 = final_model.pred_dist(df_1980)

s_params_1980 = gender_dist_1980.params['s']
scale_params_1980 = gender_dist_1980.params['scale']

mu_params_1980 = np.log(scale_params_1980)
sigma_params_1980 = s_params_1980

results_1980 = {
    'male': {
        'mu': mu_params_1980[0],
        'sigma': sigma_params_1980[0]
    },
    'female': {
        'mu': mu_params_1980[1],
        'sigma': sigma_params_1980[1]
    }
}

output_dir = "./results/genders/"
output_path = os.path.join(output_dir, "distribution_params_1980.json")

os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results_1980, f, indent=4)

gender_dist_2010 = final_model.pred_dist(df_2010)

s_params_2010 = gender_dist_2010.params['s']
scale_params_2010 = gender_dist_2010.params['scale']

mu_params_2010 = np.log(scale_params_2010)
sigma_params_2010 = s_params_2010

results_2010 = {
    'male': {
        'mu': mu_params_2010[0],
        'sigma': sigma_params_2010[0]
    },
    'female': {
        'mu': mu_params_2010[1],
        'sigma': sigma_params_2010[1]
    }
}

output_dir = "./results/genders/"
output_path = os.path.join(output_dir, "distribution_params_2010.json")

os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results_2010, f, indent=4)
