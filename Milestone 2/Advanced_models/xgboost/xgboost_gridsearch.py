import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns

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

# Define the parameter grid for hyperparameter search
param_grid = {
    'max_depth': [3, 4, 5, 6],           # Depth of the trees
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'n_estimators': [50, 100, 200],     # Number of trees
    'subsample': [0.6, 0.8, 1.0],       # Subsampling ratio of training instances
    'colsample_bytree': [0.6, 0.8, 1.0] # Subsampling ratio of columns
}

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

# Define GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=make_scorer(log_loss, greater_is_better=False, needs_proba=True),  # Log-loss as metric
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1  # Use all available processors
)

# Perform grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Log Loss:", -grid_search.best_score_)

# Get results in a DataFrame
results = grid_search.cv_results_

# Convert to DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Sort by mean test score
results_df = results_df.sort_values(by="mean_test_score", ascending=False)

# Save grid search results
results_df.to_csv("grid_search_results.csv", index=False)
