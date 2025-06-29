import sys
sys.path.append('../')
from plotting import roc_curve_and_auc, goal_rate_by_percentile, cumulative_proportion_of_goals, reliability_diagram
import wandb
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

# regular
#phase = "02"

# playoff
phase = "03"

'''
# 1. Logistic Regression
# Load the dataset
df = pd.read_csv("../../dataset/simple_engineered/augmented_test_data.csv")
df = df[df['game_id'].astype(str).str.startswith('2020' + phase)].copy()

features = ['distance_from_net', 'angle_from_net']
X_test = df[features]
y_test = df['is_goal']

# Fill missing values
X_test.fillna(X_test.mean(), inplace=True)
y_test.fillna(0, inplace=True)

# Distance only
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-logistic-distance", name="logistic-distance-only-test")

model_name = "Distance Only"
model_path = f"{model_name.replace(' ', '_')}_logistic_model.pkl"
model = joblib.load(model_path)
print(f"Loaded model: {model_name} from {model_path}")

features_subset = ['distance_from_net']

# Predictions
y_prob = model.predict_proba(X_test[features_subset])[:, 1]

# Plot graphs
roc_curve_and_auc(y_test, y_prob, log_to_run=run)
goal_rate_by_percentile(y_test, y_prob, log_to_run=run)
cumulative_proportion_of_goals(y_test, y_prob, log_to_run=run)
reliability_diagram(y_test, y_prob, log_to_run=run)

run.log_model(path=model_path, name=model_name.replace(' ', '_'))
run.finish()

# Angle only
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-logistic-angle", name="logistic-angle-only-test")

model_name = "Angle Only"
model_path = f"{model_name.replace(' ', '_')}_logistic_model.pkl"
model = joblib.load(model_path)
print(f"Loaded model: {model_name} from {model_path}")

features_subset = ['angle_from_net']

# Predictions
y_prob = model.predict_proba(X_test[features_subset])[:, 1]

# Plot graphs
roc_curve_and_auc(y_test, y_prob, log_to_run=run)
goal_rate_by_percentile(y_test, y_prob, log_to_run=run)
cumulative_proportion_of_goals(y_test, y_prob, log_to_run=run)
reliability_diagram(y_test, y_prob, log_to_run=run)

run.log_model(path=model_path, name=model_name.replace(' ', '_'))
run.finish()

# Distance + Angle
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-logistic-distance-angle", name="logistic-distance-angle-test")

model_name = "Distance and Angle"
model_path = f"{model_name.replace(' ', '_')}_logistic_model.pkl"
model = joblib.load(model_path)
print(f"Loaded model: {model_name} from {model_path}")

features_subset = ['distance_from_net', 'angle_from_net']

# Predictions
y_prob = model.predict_proba(X_test[features_subset])[:, 1]

# Plot graphs
roc_curve_and_auc(y_test, y_prob, log_to_run=run)
goal_rate_by_percentile(y_test, y_prob, log_to_run=run)
cumulative_proportion_of_goals(y_test, y_prob, log_to_run=run)
reliability_diagram(y_test, y_prob, log_to_run=run)

run.log_model(path=model_path, name=model_name.replace(' ', '_'))
run.finish()
'''

# 2. XGBoost
def evaluate_xgboost_model(model_path, X_test, y_test):
    """
    Loads an XGBoost model from a JSON file, evaluates it on a test dataset, and plots relevant metrics.

    Parameters:
        model_path (str): Path to the XGBoost model file in JSON format.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target values.
    Returns:
        None
    """
    run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-xgb", name="xgb-test")

    # Load the XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Convert test data to DMatrix
    dtest = xgb.DMatrix(X_test)

    # Predict probabilities
    y_prob = model.predict(dtest)

    # Evaluate metrics
    y_pred = (y_prob >= 0.5).astype(int)  # Convert probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")

    # Plot
    roc_curve_and_auc(y_test, y_prob, log_to_run=run)
    goal_rate_by_percentile(y_test, y_prob, log_to_run=run)
    cumulative_proportion_of_goals(y_test, y_prob, log_to_run=run)
    reliability_diagram(y_test, y_prob, log_to_run=run)

    run.log_model(path=model_path, name="xgb_classifier")
    run.finish()

model_path = "xgb_classifier_cv.json"
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

# Load test data
df = pd.read_csv("../../dataset/complex_engineered/augmented_test_data.csv")
df = df[df['game_id'].astype(str).str.startswith('2020' + phase)].copy()

df['distance_from_last_event'] = df['distance_from_last_event'].fillna(df['distance_from_last_event'].mean())
df['shot_type'] = df['shot_type'].fillna(df['shot_type'].mode()[0])
df['x_coord'] = df['x_coord'].fillna(df['x_coord'].mean())
df['y_coord'] = df['y_coord'].fillna(df['y_coord'].mean())
df['last_event_type'] = df['last_event_type'].fillna(df['last_event_type'].mode()[0])
df['time_from_last_event'] = df['time_from_last_event'].fillna(df['time_from_last_event'].mean())
df['angle_from_net'] = df['angle_from_net'].fillna(df['angle_from_net'].mean())
df['distance_from_net'] = df['distance_from_net'].fillna(df['distance_from_net'].mean())

X = df[features]

# Let XGBoost handle NaN values
if X.isnull().sum().sum() > 0:
    print('[INFO] XGBoost will handle NaN values natively.')

y = df['is_goal']
y.fillna(0, inplace=True)

X = pd.get_dummies(X, columns=['shot_type', 'last_event_type'], drop_first=True) # comment this out when selecting features

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

evaluate_xgboost_model(model_path, X, y)


# 3. Neural Network

def filter_feature_and_NaN(df):
    # Define features and target
    features = [
        'game_id',
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

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['shot_type', 'last_event_type'], drop_first=True)
    
    # Filter after one hot encoding
    X = X[X['game_id'].astype(str).str.startswith('2020' + phase)].copy()
    df = df[df['game_id'].astype(str).str.startswith('2020' + phase)].copy()
    y = df['is_goal']
    
    #drop game_id
    X = X.drop(columns=['game_id'])

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Define the Neural Network Class (same as used during training)
class GoalPredictionNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(GoalPredictionNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Evaluation Function
def evaluate_neural_network(model_path, X_test, y_test):
    """
    Evaluates a neural network model on a test dataset and computes metrics.

    Parameters:
        model_path (str): Path to the saved PyTorch model (.pth file).
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target values.

    Returns:
        None
    """
    run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-neural-network", name=f"neural-network-test")

    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Load the saved model
    model = torch.load(model_path)
    model.eval()
    print(f"Loaded model from {model_path}")

    # Predict probabilities
    with torch.no_grad():
        y_prob = model(X_test_tensor).numpy().flatten()

    # 1. ROC Curve and AUC
    roc_curve_and_auc(y_test, y_prob, log_to_run=run)

    # 2. Goal Rate by Percentile
    goal_rate_by_percentile(y_test, y_prob, log_to_run=run)

    # 3. Cumulative Proportion of Goals by Percentile
    cumulative_proportion_of_goals(y_test, y_prob, log_to_run=run)

    # 4. Reliability Diagram
    reliability_diagram(y_test, y_prob, log_to_run=run)

    run.log_model(path="best_model.pth", name=f"neural-network-test")
              
    run.finish()

df = pd.read_csv("../../dataset/complex_engineered/augmented_test_data.csv")
model_path = "best_model.pth"
X, y = filter_feature_and_NaN(df)
evaluate_neural_network(model_path, X, y)