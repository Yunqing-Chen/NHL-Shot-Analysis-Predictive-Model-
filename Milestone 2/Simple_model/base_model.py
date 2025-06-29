import wandb
import joblib
import pandas as pd
import numpy as np
import os
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

os.sys.path.append(str(pathlib.Path(__file__).absolute().resolve().parents[1]))
from path_utils import get_git_root_path


# Load the dataset
current_file_path = pathlib.Path(__file__)
current_dir_path = current_file_path.parent
base_root_path = get_git_root_path(current_file_path)
if base_root_path is None:
    raise Exception('Could not locate root git directory for processing dataset')
df = pd.read_csv((base_root_path / 'ift6758' / 'dataset' / 'complex_engineered' / 'augmented_data.csv'))

features = ['distance_from_net', 'angle_from_net']
X = df[features]
y = df['is_goal']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values separately to avoid data leakage
X_train.fillna(X_train.mean(), inplace=True)
X_val.fillna(X_train.mean(), inplace=True)
y_train.fillna(0, inplace=True)
y_val.fillna(0, inplace=True)

# Logistic Regression Models
models = {
    "Distance Only": LogisticRegression().fit(X_train[['distance_from_net']], y_train),
    "Angle Only": LogisticRegression().fit(X_train[['angle_from_net']], y_train),
    "Distance + Angle": LogisticRegression().fit(X_train, y_train),
}

# Iterate through models and log results for each
results = {}

for model_name, (model, features_subset) in {
    "Distance Only": (models["Distance Only"], ['distance_from_net']),
    "Angle Only": (models["Angle Only"], ['angle_from_net']),
    "Distance and Angle": (models["Distance + Angle"], features),
}.items():
    # Initialize W&B run for the model
    wandb.init(
        project="logistic_regression_comparison",
        entity="IFT6758_2024-B01",
        name=model_name,
        tags=[model_name.replace(" ", "_")],  # Example: "Distance_Only"
    )

    # Predictions
    y_prob = model.predict_proba(X_val[features_subset])[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Store results
    results[model_name] = {"roc_auc": roc_auc, "y_prob": y_prob}

    model_save_path = (current_dir_path / f"{model_name.replace(' ', '_')}_logistic_model.pkl")
    joblib.dump(model, model_save_path)
    print(f"Saved model: {model_name} to {str(model_save_path)}")

    # Log ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.50)")
    plt.title(f"ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    wandb.log({f"{model_name} ROC Curve": wandb.Image(plt)})
    plt.close()

    # Goal Rate by Percentile
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    df_val['percentile'] = pd.qcut(df_val['y_prob'], 100, labels=False, duplicates='drop') + 1
    goal_rate_by_percentile = df_val.groupby('percentile')['y_val'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(goal_rate_by_percentile.index, goal_rate_by_percentile, marker='o')
    plt.title(f"Goal Rate by Percentile: {model_name}")
    plt.xlabel("Shot Probability Model Percentile")
    plt.ylabel("Goal Rate (#goals / (#goals + #no_goals))")
    plt.ylim(0, 1)  # Set the y-axis range from 0 to 1
    plt.grid(True)
    plt.gca().invert_xaxis()  # Reverse the x-axis
    wandb.log({f"{model_name} Goal Rate by Percentile": wandb.Image(plt)})
    plt.close()

    # Cumulative Proportion of Goals
    df_val = df_val.sort_values('y_prob', ascending=False).reset_index(drop=True)
    cumulative_goals = df_val['y_val'].cumsum()
    total_goals = df_val['y_val'].sum()
    cumulative_goal_percentage = cumulative_goals / total_goals
    percentile_range = np.linspace(100, 0, len(cumulative_goal_percentage))

    plt.figure(figsize=(10, 6))
    plt.plot(percentile_range, cumulative_goal_percentage, marker='o', label="Cumulative Goals")
    plt.title(f"Cumulative Proportion of Goals: {model_name}")

    plt.xlabel("Shot Probability Model Percentile")
    plt.ylabel("Cumulative Proportion of Goals")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis() # Reverse the x-axis
    wandb.log({f"{model_name} Cumulative Proportion of Goals": wandb.Image(plt)})
    plt.close()

    # Reliability Diagram (Calibration Curve)
    CalibrationDisplay.from_predictions(y_val, y_prob, n_bins=10, strategy='uniform')
    plt.title(f"Reliability Diagram (Calibration Curve): {model_name}")
    wandb.log({f"{model_name} Reliability Diagram": wandb.Image(plt)})
    plt.close()

    # Finish W&B run for the model
    wandb.finish()

# Log a summary table for all models in the final run
wandb.init(
    project="logistic_regression_comparison",
    entity="IFT6758_2024-B01",
    name="Summary Table",
)

metrics_table = wandb.Table(columns=["Model Name", "ROC AUC"])
for model_name, result in results.items():
    metrics_table.add_data(model_name, result["roc_auc"])
wandb.log({"Model Comparison Table": metrics_table})

wandb.finish()
