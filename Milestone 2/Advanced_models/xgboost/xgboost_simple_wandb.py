#!/usr/bin/env python3

import sys
sys.path.append('../')
import wandb
import pandas as pd
import numpy as np
import xgboost as xgb
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from plotting import roc_curve_and_auc, goal_rate_by_percentile, cumulative_proportion_of_goals, reliability_diagram


# Set paths
try:
    current_dirpath = pathlib.Path(__file__).parent.absolute().resolve()
except NameError:
    current_dirpath = pathlib.Path(os.path.curdir).absolute().resolve()

if not current_dirpath.parts[-3:] == ('ift6758', 'advanced_models', 'xgboost'):
    raise Exception(
        'It appears that this file is executed from the wrong location\n'
        'Expected path: (root-->)ift6758/advanced_models/xgboost/\n'
        f'Current path: {current_dirpath}'
    )
root_dirpath = current_dirpath.parents[1]

# Load the dataset
dataset_path = (root_dirpath / 'dataset' / 'complex_engineered' / 'augmented_data.csv')
if not (dataset_path.is_file() and dataset_path.match('*.csv')):
    raise Exception(
        'It appears that the dataset either does not exist or is not a valid CSV\n'
        f'Path: {dataset_path}'
    )
df = pd.read_csv(dataset_path)

# Initialize WandB run
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-xgboost-models")


# Feature selection (basic) and NaN values processing
features = ['distance_from_net', 'angle_from_net']
X = df[features]
X.fillna(X.mean(), inplace=True)
y = df['is_goal']
y.fillna(0, inplace=True)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model instance
xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train, y_train)

# XGBoost model predictions
y_pred = xg_clf.predict(X_val)
y_probas = xg_clf.predict_proba(X_val)

# Saving model to JSON
model_basename = 'xgb_classifier_simple.json'
model_path = current_dirpath / model_basename
xg_clf.save_model(fname=model_path)

# Log model to WandB run
run.log_model(path=model_path, name=model_basename)

# Log plots to WandB run
roc_curve_and_auc(y_val, y_probas[:, 1], log_to_run=run)
goal_rate_by_percentile(y_val, y_probas[:, 1], log_to_run=run)
cumulative_proportion_of_goals(y_val, y_probas[:, 1], log_to_run=run)
reliability_diagram(y_val, y_probas[:, 1], log_to_run=run)

# Random Baseline generation and log to WandB
# y_probas_random = np.random.uniform(0, 1, len(y_val))

# roc_curve_and_auc(y_val, y_probas_random, log_to_run=run)
# goal_rate_by_percentile(y_val, y_probas_random, log_to_run=run)
# cumulative_proportion_of_goals(y_val, y_probas_random, log_to_run=run)

# End run
run.finish()


'''
roc_curve_and_auc(y_val, y_probas[:, 1])
goal_rate_by_percentile(y_val, y_probas[:, 1])
cumulative_proportion_of_goals(y_val, y_probas[:, 1])
'''