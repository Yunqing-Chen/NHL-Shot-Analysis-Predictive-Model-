#!/usr/bin/env python3

#TODO:
# -Figure out how to deal with NaN values in X (see lines 79-80)

import sys
sys.path.append('../')
import wandb
import pandas as pd
import numpy as np
import xgboost as xgb
import pathlib
import os
from sklearn import preprocessing
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-xgboost-models", name="xgboost_complex_no_fillna")

# Feature selection (basic) and NaN values processing
features = [
    'distance_from_net', # f0
    'angle_from_net', # f1  #
    'game_seconds', # f2  #
    'period', # f3  ###
    'x_coord', # f4  ###
    'y_coord', # f5
    'angle_from_net', # f6  #
    'shot_type', # comment this out when selecting features
    'last_event_type', # comment this out when selecting features
    'last_x_coord', # f7  ###
    'last_y_coord', # f8  ##
    'time_from_last_event', # f9
    'distance_from_last_event', # f10  ##
    'rebound', # f11
    'change_in_shot_angle', # f12  ###
    'speed' # f13  ##
]

df['distance_from_last_event'] = df['distance_from_last_event'].fillna(df['distance_from_last_event'].mean())
df['shot_type'] = df['shot_type'].fillna(df['shot_type'].mode()[0])
df['x_coord'] = df['x_coord'].fillna(df['x_coord'].mean())
df['y_coord'] = df['y_coord'].fillna(df['y_coord'].mean())
df['last_event_type'] = df['last_event_type'].fillna(df['last_event_type'].mode()[0])
df['time_from_last_event'] = df['time_from_last_event'].fillna(df['time_from_last_event'].mean())
df['angle_from_net'] = df['angle_from_net'].fillna(df['angle_from_net'].mean())
df['distance_from_net'] = df['distance_from_net'].fillna(df['distance_from_net'].mean())

X = df[features]
# Check if some features have more than 1% (arbritrary) of NaN values
df_percent_nan = (df.isnull().sum() / df.count()).sort_values(ascending=False).loc[lambda x : x > 1.0]
if not df_percent_nan.empty:
    print('[WARNING] The following features have lots of NaN values')
    print(df_percent_nan)
    print('Current method of inputation is replacing with mean')

# Let XGBoost handle NaN values
if X.isnull().sum().sum() > 0:
    print('[INFO] XGBoost will handle NaN values natively.')

y = df['is_goal']
y.fillna(0, inplace=True)

X = pd.get_dummies(X, columns=['shot_type', 'last_event_type'], drop_first=True) # comment this out when selecting features

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Transform data to Dmatrix data structure
#data_dmatrix = xgb.DMatrix(data=X, label=y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data to DMatrix data structure so that we can use xgb.cv
# Reference: https://datascience.stackexchange.com/questions/12799/pandas-dataframe-to-dmatrix
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dval = xgb.DMatrix(data=X_val, label=y_val)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'alpha': 10,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Cross-validation
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=5,
    early_stopping_rounds=10,
    metrics="logloss",
    seed=42
)

# Log best score and iteration
best_iteration = len(cv_results)
best_score = cv_results['test-logloss-mean'].min()
print(f"Best Iteration: {best_iteration}, Best Log Loss: {best_score:.4f}")

# Train final model with best iteration
params['n_estimators'] = best_iteration
xgb_clf = xgb.train(params, dtrain, num_boost_round=best_iteration)

# Save the model
model_basename = 'xgb_classifier_cv.json'
model_path = pathlib.Path.cwd() / model_basename
xgb_clf.save_model(fname=model_path)

# After training the XGBoost model, predict probabilities for the validation set
y_val_pred_prob = xgb_clf.predict(dval)  # dval is the validation set in DMatrix format

# Plotting and logging to WandB
roc_curve_and_auc(y_val, y_val_pred_prob, log_to_run=run)
goal_rate_by_percentile(y_val, y_val_pred_prob, log_to_run=run)
cumulative_proportion_of_goals(y_val, y_val_pred_prob, log_to_run=run)
reliability_diagram(y_val, y_val_pred_prob, log_to_run=run)

xgb.plot_importance(xgb_clf, importance_type='gain')  # Change to 'weight' or 'cover' if needed
plt.title("Feature Importance by Gain")
plt.show()

xgb.plot_importance(xgb_clf, importance_type='weight')  
plt.title("Feature Importance by Weight")
plt.show()

xgb.plot_importance(xgb_clf, importance_type='cover')  
plt.title("Feature Importance by Cover")
plt.show()

# Log model to WandB run
run.log_model(path=model_path, name=model_basename)

# End run
run.finish()
