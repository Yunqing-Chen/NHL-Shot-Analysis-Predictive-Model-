#!/usr/bin/env python

import wandb
import pandas as pd
import numpy as np
import pathlib
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple
import matplotlib.pyplot as plt



## 1. ROC Curve and AUC
def roc_curve_and_auc(y_val, y_prob, log_to_run=None, commit=True):
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"Decision Trees Classifier (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if log_to_run:
        if wandb.run is not None:
            run.log({'roc_auc': wandb.Image(plt)}, commit=commit)
            return
    plt.show()

## 2. Goal Rate by Percentile (Binned by 5%)
def goal_rate_by_percentile(y_val, y_prob, log_to_run=None, commit=True):
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    df_val['percentile'] = pd.qcut(df_val['y_prob'], 100, labels=False, duplicates='drop') + 1  # Percentiles from 1 to 100
    goal_rate_by_percentile = df_val.groupby('percentile')['y_val'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(goal_rate_by_percentile.index, goal_rate_by_percentile, marker='o')
    ax.set_title("Goal Rate by Percentile")
    ax.set_xlabel("Shot Probability Model Percentile")
    ax.set_ylabel("Goal Rate (#goals / (#goals + #no_goals))")
    ax.set_ylim(0, 1)  # Set the y-axis range from 0 to 1
    ax.grid(True)
    ax.invert_xaxis()  # Reverse the x-axis

    if log_to_run:
        log_to_run.log({'Goal Rate by Percentile': wandb.Image(fig)}, commit=commit)
    plt.show()
    plt.close(fig)

## 3. Cumulative Proportion of Goals by Percentile
def cumulative_proportion_of_goals(y_val, y_prob, log_to_run=None, commit=True):
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    df_val = df_val.sort_values('y_prob', ascending=False).reset_index(drop=True)  # Sort by descending probability

    # Calculate cumulative goals and proportion
    cumulative_goals = df_val['y_val'].cumsum()
    total_goals = df_val['y_val'].sum()
    cumulative_goal_percentage = cumulative_goals / total_goals

    # Percentile bins (from 100% to 0%)
    percentiles = np.linspace(100, 0, len(cumulative_goal_percentage))

    # Plot cumulative proportion of goals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percentiles, cumulative_goal_percentage, marker='o', label="Cumulative Proportion")
    ax.set_title("Cumulative Proportion of Goals by Model Percentile")
    ax.set_xlabel("Shot Probability Model Percentile")
    ax.set_ylabel("Cumulative Proportion of Goals")
    ax.grid(True)
    ax.invert_xaxis()  # Reverse the x-axis

    if log_to_run:
        log_to_run.log({'Cumulative Proportion of Goals': wandb.Image(fig)}, commit=commit)
    plt.show()
    plt.close(fig)


# 4. Reliability Diagram (Calibration Curve)
def reliability_diagram(y_val, y_prob, log_to_run=None, commit=True):
    CalibrationDisplay.from_predictions(y_val, y_prob, n_bins=10, strategy='uniform')
    plt.title("Reliability Diagram (Calibration Curve)")

    if log_to_run:
        log_to_run.log({f"Reliability Diagram": wandb.Image(plt)}, commit=commit)

    plt.show()
    plt.close()


def detect_nan_values(df):
    """
    Detects NaN values in the DataFrame and prints details about their occurrence.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze for NaN values.

    Returns:
        None
    """
    # Check for NaN values in each column
    nan_counts = df.isna().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        print(f"Total NaN values in dataset: {total_nan}")
        print("Columns with NaN values:")
        print(nan_counts[nan_counts > 0])

        # Find rows with NaN values
        nan_rows = df[df.isna().any(axis=1)]
        print(f"\nRows with NaN values (showing up to 10):")
        print(nan_rows.head(10))

        # Identify specific cells with NaN
        print("\nSpecific locations of NaN values:")
        nan_locations = np.where(pd.isna(df))
        for row, col in zip(nan_locations[0], nan_locations[1]):
            print(f"NaN at Row: {row}, Column: {df.columns[col]}")
    else:
        print("No NaN values detected in the dataset.")


def filter_feature_and_NaN(df):
    # Define features and target
    features = [
        'distance_from_net',
        'angle_from_net',
        'game_seconds',
        'period',
        'x_coord',
        'y_coord',
        'angle_from_net',
        'shot_type',
        'last_event_type',
        'last_x_coord',
        'last_y_coord',
        'time_from_last_event',
        'distance_from_last_event',
        'rebound',
        'change_in_shot_angle',
        'speed'
    ]

    df['change_in_shot_angle'] = df['change_in_shot_angle'].fillna(0)
    df['speed'] = df['speed'].fillna(0)
    df['distance_from_last_event'] = df['distance_from_last_event'].fillna(df['distance_from_last_event'].mean())
    df['shot_type'] = df['shot_type'].fillna(df['shot_type'].mode()[0])
    df['x_coord'] = df['x_coord'].fillna(df['x_coord'].mean())
    df['y_coord'] = df['y_coord'].fillna(df['y_coord'].mean())
    df['last_event_type'] = df['last_event_type'].fillna(df['last_event_type'].mode()[0])
    df['time_from_last_event'] = df['time_from_last_event'].fillna(df['time_from_last_event'].mean())
    df['angle_from_net'] = df['angle_from_net'].fillna(df['angle_from_net'].mean())
    df['distance_from_net'] = df['distance_from_net'].fillna(df['distance_from_net'].mean())

    # Assuming `df` is your dataset
    X = df[features]
    y = df['is_goal']

    detect_nan_values(X)

    #assert not np.isnan(X).any(), "NaN values found in training data"
    #assert not np.isinf(X['speed']).any(), "Inf values found in training data"

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['shot_type', 'last_event_type'], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_dataset(return_paths: Tuple[str, str] = False):
    try:
        current_dirpath = pathlib.Path(__file__).parent.absolute().resolve()
    except NameError:
        current_dirpath = pathlib.Path(os.path.curdir).absolute().resolve()

    if not current_dirpath.parts[-3:] == ('ift6758', 'advanced_models', 'decisiontrees'):
        raise Exception(
            'It appears that this file is executed from the wrong location\n'
            'Expected path: (root-->)ift6758/advanced_models/neuro_network/\n'
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
    if return_paths:
        return df, (current_dirpath, root_dirpath)
    return df



# Load the dataset
df, (current_dirpath, _) = load_dataset(return_paths=True)

# Initialize WandB run
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-decisiontrees-models")

# Replace possible `inf` values with -1
df.loc[:, 'speed'] = df['speed'].replace(np.inf, -1)

# Split features and target
X, y = filter_feature_and_NaN(df)

# Split data
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=97)
y_val = y_val.reset_index()['is_goal']

# Build a DecisionTree Classifier
model = DecisionTreeClassifier()

# Define parameters ranges to do GridSearch
params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 5, 10, 20, None],
}

gs = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
gs.fit(X_train, y_train)

# Log Cross Validation results
#run.log({'cv_results': pd.DataFrame(gs.cv_results_)}, commit=False)

# Get predictions
y_pred = gs.predict(X_val)
y_probas = gs.predict_proba(X_val)[:, 1]

# Return best parameters
run.log({'best_params': gs.best_params_}, commit=True)

# Saving model to Pickle
model_basename = 'decisiontrees_simple.pkl'
model_path = current_dirpath / model_basename
with open(model_path,'wb') as model_file:
    pickle.dump(gs, model_file)

# Log model to WandB run
run.log_model(path=model_path, name=model_basename)

# Plots and log to WandB
roc_curve_and_auc(y_val, y_probas, log_to_run=run, commit=False)
goal_rate_by_percentile(y_val, y_probas, log_to_run=run, commit=False)
cumulative_proportion_of_goals(y_val, y_probas, log_to_run=run, commit=False)
reliability_diagram(y_val, y_probas, log_to_run=run, commit=True)

# End run
run.finish()
