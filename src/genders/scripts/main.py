"""Script for training the best model only on (year, gender)."""

import os
import json

import pandas as pd
import numpy as np

from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ngboost.scores import LogScore

from src.utils.utils import load_json

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

df_x = pd.read_csv("data/X_trn.csv")
df_y = pd.read_csv("data/y_trn.csv")
df = pd.concat([df_y, df_x], axis=1)

X = df[['year', 'gender']]
X['gender'] = X['gender'].map(
    {
        'Male': 1,
        'Female': 0
    }
)
y = df['realrinc']

results_ngb = load_json(path="/workspace/results/hparams/best_hparams_ngb.json")
results_qrf = load_json(path="/workspace/results/hparams/best_hparams_qrf.json")

if results_ngb['best_loss'] <= results_qrf['best_loss']:
    
    final_model = NGBRegressor(
        Dist=LogNormal,
        Score=LogScore,
        n_estimators=results_ngb['n_estimators'],
        learning_rate=results_ngb['learning_rate'],
        col_sample=results_ngb['col_sample'],
        minibatch_frac=results_ngb['minibatch_frac'],
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
else:
    pass
