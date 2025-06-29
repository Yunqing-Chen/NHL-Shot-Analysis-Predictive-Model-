import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

## 1. ROC Curve and AUC
def roc_curve_and_auc(y_val, y_prob, log_to_run=None):
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"XGBoost Classifier (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if log_to_run:
        log_to_run.log({'ROC Curve and AUC': wandb.Image(fig)})
    plt.show()
    plt.close(fig)
       
## 2. Goal Rate by Percentile (Binned by 5%)
def goal_rate_by_percentile(y_val, y_prob, log_to_run=None):
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
        log_to_run.log({'Goal Rate by Percentile': wandb.Image(fig)})
    plt.show()
    plt.close(fig)
    
## 3. Cumulative Proportion of Goals by Percentile
def cumulative_proportion_of_goals(y_val, y_prob, log_to_run=None):
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
        log_to_run.log({'Cumulative Proportion of Goals': wandb.Image(fig)})
    plt.show()
    plt.close(fig)

# 4. Reliability Diagram (Calibration Curve)
def reliability_diagram(y_val, y_prob, log_to_run=None):
    CalibrationDisplay.from_predictions(y_val, y_prob, n_bins=30, strategy='uniform')
    plt.title("Reliability Diagram (Calibration Curve)")
    if log_to_run:
        log_to_run.log({f"Reliability Diagram": wandb.Image(plt)})
    '''
    if log_to_run:
        if wandb.run is not None:
            run.log({'reliability_diagram': plt})
            return
    '''
    plt.show()
