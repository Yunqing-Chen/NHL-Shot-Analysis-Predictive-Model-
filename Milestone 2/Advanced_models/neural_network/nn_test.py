import sys
sys.path.append('../')
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pathlib
from plotting import roc_curve_and_auc, goal_rate_by_percentile, cumulative_proportion_of_goals, reliability_diagram

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


def load_dataset():
    try:
        current_dirpath = pathlib.Path(__file__).parent.absolute().resolve()
    except NameError:
        current_dirpath = pathlib.Path(os.path.curdir).absolute().resolve()

    if not current_dirpath.parts[-3:] == ('ift6758', 'advanced_models', 'neural_network'):
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
    return df


def filter_feature_and_NaN(df):
    # Define features and target
    features = [
        'distance_from_net',
        'angle_from_net',
        'game_seconds',
        'period',
        'x_coord',
        'y_coord',
        'shot_type',
        'last_event_type',
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
    #assert not np.isinf(X).any(), "Inf values found in training data"

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['shot_type', 'last_event_type'], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def split_and_load_data(X, y):
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for training and validation
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor


# Define the neural network
class GoalPredictionNet(nn.Module):
    def __init__(self, input_dim):
        super(GoalPredictionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )
    
    def forward(self, x):
        return self.net(x)

def train(train_loader, val_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs = 50, lr=0.0001):
    # Instantiate the model
    input_dim = X_train_tensor.shape[1]
    model = GoalPredictionNet(input_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2 regularization with weight_decay

    # Training loop
    best_val_auc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0
        y_val_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                val_loss += loss.item()
                y_val_pred.append(y_pred)
        
        val_loss /= len(val_loader)
        y_val_pred = torch.cat(y_val_pred).cpu().numpy()
        
        # Calculate validation metrics
        y_val_pred_binary = (y_val_pred >= 0.5).astype(int)
        val_accuracy = accuracy_score(y_val_tensor.numpy(), y_val_pred_binary)
        val_auc = roc_auc_score(y_val_tensor.numpy(), y_val_pred)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model, "best_model.pth")
        
    return model




def evaluate_model_and_plot(model, X_val, y_val, log_to_run=None):
    """
    Evaluate a trained model on the validation set and plot performance metrics:
    - ROC Curve and AUC
    - Goal Rate by Percentile
    - Cumulative Proportion of Goals
    - Reliability Diagram

    Parameters:
        model (sklearn-like or PyTorch model): The trained model to evaluate.
        X_val (pd.DataFrame or torch.Tensor): Validation feature set.
        y_val (pd.Series or torch.Tensor): Validation target values.
    """
    model.eval()
    with torch.no_grad():
        y_prob = model(X_val).numpy().flatten()

    # Ensure y_val is numpy array
    y_val = y_val.values if isinstance(y_val, pd.Series) else y_val
    y_val = y_val.flatten()

    # 1. ROC Curve and AUC
    roc_curve_and_auc(y_val, y_prob, log_to_run=log_to_run)

    # 2. Goal Rate by Percentile
    goal_rate_by_percentile(y_val, y_prob, log_to_run=log_to_run)

    # 3. Cumulative Proportion of Goals by Percentile
    cumulative_proportion_of_goals(y_val, y_prob, log_to_run=log_to_run)

    # 4. Reliability Diagram
    reliability_diagram(y_val, y_prob, log_to_run=log_to_run)

num_epochs=50
lr=0.0001
# record the run with the epoch and learning rate
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-neural-network", name=f"neural-network-epoch{num_epochs}-lr{lr}")

X, y = filter_feature_and_NaN(load_dataset())
train_loader, val_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = split_and_load_data(X, y)
model = train(train_loader, val_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=num_epochs, lr=lr)
evaluate_model_and_plot(model, X_val_tensor, y_val_tensor, log_to_run=run)
torch.save(model, "best_model.pth")

run.log_model(path="best_model.pth", name=f"neural-network-epoch{num_epochs}-lr{lr}")
              
run.finish()