import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        url = f"{self.base_url}/predict"
        try:
            response = requests.post(url, json=X)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error in predict: {e}")
            return None

    def logs(self) -> dict:
        """Get server logs"""
        url = f"{self.base_url}/logs"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error in logs: {e}")
            return None

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        url = f"{self.base_url}/download_registry_model"
        payload = {"workspace": workspace, "model": model, "version": version}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error in download_registry_model: {e}")
            return {"error": str(e)}
        
    def process_game(self, game_id: str) -> pd.DataFrame:
        """
        Submits a game ID to the server's /process_game endpoint for processing.
        Retrieves processed events, including their predicted probabilities.
        
        Args:
            game_id (str): The game ID to process.
            
        Returns:
            pd.DataFrame: A DataFrame with the processed events and predictions, or None if no new data to process.
        """
        url = f"{self.base_url}/process_game"
        payload = {"game_id": game_id}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # Handle response based on content type
            if isinstance(response.json(), dict):
                # If it's a message, check for "No new events" case
                message = response.json().get("message", "")
                if message == "No new events to process.":
                    logger.info(f"No new events to process for game ID: {game_id}.")
                    return None
                else:
                    logger.warning(f"Unexpected message received: {message}")
                    return None
            
            # If response is not a dict, assume it's a DataFrame
            logger.info(f"Successfully processed events for game ID: {game_id}.")
            return pd.DataFrame(json.loads(response.text))
        
        except requests.RequestException as e:
            logger.error(f"Error in process_game: {e}")
            return None