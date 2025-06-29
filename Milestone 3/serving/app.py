"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import wandb
import json
# from helper import download_game_data, parse_game_events, augment_data, update_tracker, get_unprocessed_events, load_model
os.sys.path.append(str(( Path(__file__) / '..' / '..' ).absolute().resolve()))
print(os.sys.path)
from ift6758.helper_ms3.helper import download_game_data, parse_game_events, augment_data, update_tracker, get_unprocessed_events, load_model


# import ift6758


# LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

# Logger setup
LOG_FILE = "flask.log"
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)
# Global variables
current_model = None
current_model_name = None
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
processed_events_tracker = {}

#@app.before_first_request
@app.before_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    app.before_request_funcs[None].remove(before_first_request)

    # TODO: setup basic logging configuration
    logger.info("Flask app initialized.")

    # TODO: any other initialization before the first request (e.g. load default model)
    global current_model, current_model_name
    
    # Load default model ("Distance Only")
    default_model_name = "Distance_Angle"
    default_model_version = "v0"
    default_model_path = os.path.join(model_dir, f"{default_model_name}_model.pkl")

    # Check if the default model is already downloaded
    if not os.path.exists(default_model_path):
        logger.info(f"Default model {default_model_name} not found locally. Downloading from WandB.")
        try:
            logger.info(f"Downloading model {default_model_name} version {default_model_version} from WandB registry.")
            wandb.init(project="ms2-logistic-regression", mode="online")
            model_artifact = wandb.use_artifact(f"{default_model_name}:{default_model_version}", type="model")
            model_artifact.download(root=model_dir)
            wandb.finish()
        except Exception as e:
            logger.error(f"Failed to download default model {default_model_name}: {e}")
            return 

    # Load the default model
    current_model = load_model(default_model_path, logger)
    if current_model:
        current_model_name = f"{default_model_name}_{default_model_version}"
        logger.info(f"Default model {default_model_name} version {default_model_version} loaded successfully.")
    else:
        logger.error(f"Failed to load default model {default_model_name} version {default_model_version}.")


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    try:
        with open(LOG_FILE, "r") as log_file:
            log_data = log_file.read()
        return f"<pre>{log_data}</pre>"
    except Exception as e:
        logger.error(f"Failed to fetch logs: {e}")
        return jsonify({"error": "Could not retrieve logs."}), 500  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """

    global current_model, current_model_name
    try:
        # Get POST json data
        json = request.get_json()
        app.logger.info(json)
        # Parse request data
        workspace = json.get("workspace")
        model_name = json.get("model")
        version = json.get("version")

        if not all([workspace, model_name, version]):
            raise ValueError("Missing required arguments: workspace, model, or version.")

        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
        if os.path.exists(model_path):
            logger.info(f"Model {model_name} version {version} already exists locally. Loading it.")
        else:
            logger.info(f"Downloading model {model_name} version {version} from WandB registry.")
            wandb.init(project=workspace, entity="IFT6758_2024-B01", mode="online")
            logger.info(f"Initialized a run in project {workspace}")
            model_artifact = wandb.use_artifact(f"{model_name}:{version}", type="model")
            logger.info(f"Downloading artifact {model_artifact.name}")
            model_download_path = model_artifact.download(root=model_dir)
            logger.info(f'Downloaded {model_artifact.name} into {model_download_path}')
            wandb.finish()

        # Load the downloaded model
        current_model = load_model(model_path, logger)
        if current_model:
            current_model_name = f"{model_name}_{version}"
            logger.info(f"Successfully loaded model: {model_name} version {version}")
            return jsonify({"message": f"Model {model_name} version {version} loaded successfully."})
        else:
            raise Exception("Failed to load model after downloading.")
    except Exception as e:
        logger.error(f"Failed to download or load model: {e}")
        return jsonify({"error": str(e)}), 500

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    global current_model
    if current_model is None:
        logger.error("No model loaded.")
        return jsonify({"error": "No model loaded."}), 400
    
    # Get POST json data
    try:
        json = request.get_json()
        if not json:
            raise ValueError("No JSON data received.")

        features_df = pd.DataFrame.from_dict(json)
        predictions = current_model.predict_proba(features_df)[:, 1]  # Probability for the positive class
        logger.info("Predictions made successfully.")
        return jsonify(predictions.tolist())
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500  # response must be json serializable!
    

# Endpoint: /process_game
@app.route("/process_game", methods=["POST"])
def process_game():
    """
    Processes game data for a given game_id, augments it, and predicts probabilities for unprocessed events.
    """
    global current_model
    if current_model is None:
        logger.error("No model loaded.")
        return jsonify({"error": "No model loaded."}), 400

    try:
        data = request.get_json()
        game_id = data.get("game_id")
        if not game_id:
            return jsonify({"error": "Game ID is required."}), 400

        # Download and parse game data
        game_data = download_game_data(game_id)
        if not game_data:
            return jsonify({"error": "Failed to download game data."}), 400
        events_df = parse_game_events(game_data)
        unprocessed_events = get_unprocessed_events(game_id, events_df, processed_events_tracker)

        if unprocessed_events.empty:
            return jsonify({"message": "No new events to process."})

        # Augment data for unprocessed events
        augmented_events = augment_data(unprocessed_events)

        # Predict probabilities for augmented events
        if current_model_name == 'Distance_Angle':
            input_features = ["distance_from_net", "angle_from_net"]
        elif current_model_name == 'Distance_Only':
            input_features = ["distance_from_net"]
        elif current_model_name == 'Angle_Only':
            input_features = ["angle_from_net"]

        augmented_events["predicted_probabilities"] = current_model.predict_proba(
            augmented_events[["distance_from_net", "angle_from_net"]]
        )[:, 1]

        # Update tracker
        update_tracker(game_id, unprocessed_events["event_id"].tolist(), processed_events_tracker)

        # Log processed events
        logger.info(f"Processed {len(unprocessed_events)} events for game ID {game_id}.")
        return augmented_events.to_json(orient="records")

    except Exception as e:
        logger.error(f"Failed to process game: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
