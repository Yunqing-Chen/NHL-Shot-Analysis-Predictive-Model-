import os
import json
import pathlib
import re
import pandas as pd
import requests
import numpy as np
import joblib


# Helper: Update tracker
def update_tracker(game_id, processed_event_ids, processed_events_tracker):
    """
    Updates the tracker with the processed event IDs for the given game.
    """
    if game_id not in processed_events_tracker:
        processed_events_tracker[game_id] = set()
    processed_events_tracker[game_id].update(processed_event_ids)

# Helper: Get unprocessed events
def get_unprocessed_events(game_id, events_df, processed_events_tracker):
    """
    Filters out already processed events for the given game.
    """
    processed_ids = processed_events_tracker.get(game_id, set())
    unprocessed_events = events_df[~events_df["event_id"].isin(processed_ids)]
    return unprocessed_events


# Helper: Load a model
def load_model(model_path, logger):
    global current_model, current_model_name
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def download_game_data(game_id):
    """
    Downloads game data for a given game ID and saves it locally. If the data already exists,
    it loads the cached data.

    Parameters:
        game_id (str): The ID of the game to download.

    Returns:
        dict: The game data as a JSON object.
    """
    # Create the season folder based on the game ID
    base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"

    # Download the data
    url = base_url.format(game_id)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        game_data = response.json()
        if 'error' in game_data:
            print(f"Error in response for game ID {game_id}: {game_data['error']}")
            return None

        print(f"Successfully downloaded data for game ID: {game_id}")

        return game_data

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for game ID {game_id}: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred for game ID {game_id}: {err}")
        return None



def parse_game_events(game_data: dict) -> pd.DataFrame:
    """
    Parses the JSON response of a game to extract 'shot-on-goal' and 'goal' events and converts them into a Pandas DataFrame.
    """
    events = game_data.get('plays', [])
    if events == []:
        raise Exception('No play-by-play data has been found')
    game_id = game_data.get('id', '')

    # List to store parsed events
    event_data = []

    for event in events:
        event_type = event.get('typeDescKey', '')
        if event_type in ['shot-on-goal', 'goal']:
            event_data.append({
                'game_id': game_id,
                'event_id': event.get('eventId'),
                'event_type': event_type,
                'time_in_period': event.get('timeInPeriod'),
                'time_remaining': event.get('timeRemaining'),
                'period': event.get('periodDescriptor', {}).get('number'),
                'team_id': event.get('details', {}).get('eventOwnerTeamId'),
                'x_coord': event.get('details', {}).get('xCoord'),
                'y_coord': event.get('details', {}).get('yCoord'),
                'shooter_id': event.get('details', {}).get('shootingPlayerId'),
                'goalie_id': event.get('details', {}).get('goalieInNetId'),
                'shot_type': event.get('details', {}).get('shotType'),
                'zone_code': event.get('details', {}).get('zoneCode'),
            })

    return pd.DataFrame(event_data)


def segment_shot_data(season_data):
    """
    segments the shot data by team_id, game_id, and period.

    Parameters:
        season_data (pd.DataFrame): DataFrame containing all shot data for the season.

    Returns:
        dict: Nested dictionary where data is segmented by team, game, and period.
    """
    segmented_data = {}

    # Group by team_id, game_id, and period
    for team_id, team_data in season_data.groupby('team_id'):
        segmented_data[team_id] = {}
        for game_id, game_data in team_data.groupby('game_id'):
            segmented_data[team_id][game_id] = {}
            for period, period_data in game_data.groupby('period'):
                segmented_data[team_id][game_id][period] = calculate_new_metrics(period_data)

    return segmented_data


def determine_goal_location(shots_data):
    """
    Determines the goal location for a team based on the offensive zone.

    Parameters:
        shots_data (pd.DataFrame): DataFrame containing shots for a team in a period.

    Returns:
        tuple: Coordinates (x, y) of the goal.
    """
    for index, row in shots_data.iterrows():
        zone_code = row['zone_code']
        x_coord = row['x_coord']
        
        # If we find an offensive or defensive zone shot, we determine the goal location
        if zone_code == 'O':  # Offensive zone
            return (90, 0) if x_coord > 0 else (-90, 0)
        elif zone_code == 'D':  # Defensive zone
            return (-90, 0) if x_coord > 0 else (90, 0)

    # Default to offensive goal at (0, 90) if no offensive/defensive zone shots found
    return (90, 0)


def calculate_new_metrics(period_data):
    """
    Calculates new metrics (distance and angle) for shots in a period.

    Parameters:
        period_data (pd.DataFrame): DataFrame containing shots for a team in a period.

    Returns:
        pd.DataFrame: DataFrame with adjusted shot coordinates.
    """
    # Filter for SHOT and GOAL events
    df_shots_goals = period_data[period_data['event_type'].isin(['shot-on-goal', 'goal'])].copy()

    # Determine the goal location based on the shots in the period
    goal_location = determine_goal_location(period_data)

    # Calculate distance, angle, is_goal, and empty_net
    distances, angles = zip(*df_shots_goals.apply(
        lambda row: calculate_distance_and_angle(row['x_coord'], row['y_coord'], net_x=goal_location[0], net_y=goal_location[1]), axis=1
    ))
    df_shots_goals['distance_from_net'] = distances
    df_shots_goals['angle_from_net'] = angles
    df_shots_goals['is_goal'] = (df_shots_goals['event_type'] == 'goal').astype(int)
    df_shots_goals['net_x'] = goal_location[0]
    df_shots_goals['net_y'] = goal_location[1]

    df_shots_goals.drop(df_shots_goals.columns[df_shots_goals.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    return df_shots_goals


def aggregate_data(segmented_data):
    """
    Aggregates data into a single DataFrame either for the entire league or for individual teams.

    Parameters:
        segmented_data (dict): Nested dictionary where data is segmented by team, game, and period.

    Returns:
        pd.DataFrame: Combined DataFrame containing the aggregate data.
    """
    aggregate = []

    for team_id, games in segmented_data.items():
        for game_id, periods in games.items():
            for period_data in periods.values():
                aggregate.append(period_data)

    # Combine all data into a single DataFrame
    return pd.concat(aggregate, ignore_index=True)


def calculate_distance_and_angle(x, y, net_x=90, net_y=0):
    """
    Calculate the distance and angle from the shot position to the net.

    Parameters:
        x (float): x-coordinate of the shot.
        y (float): y-coordinate of the shot.
        net_x (float): x-coordinate of the net (default: offensive zone net at x=89).
        net_y (float): y-coordinate of the net (default: center of net at y=0).

    Returns:
        tuple: (distance, angle) to the net.
    """
    # Distance calculation
    distance = np.sqrt((x - net_x) ** 2 + (y - net_y) ** 2)

    # Angle calculation
    angle = np.arctan2(abs(y - net_y), abs(x - net_x)) * (180 / np.pi)

    # Left side of goal = positive, right side of goal = negative
    if (y < net_y and net_x > 0) or (y > net_y and net_x < 0):
        angle = -angle

    return distance, angle


def augment_data(all_data: pd.DataFrame):
    """
    Augments the raw data with new features and saves the augmented data to a CSV file.
    """
    # Segment the data by team, game, and period
    segmented_data = segment_shot_data(all_data)

    # Aggregate the data and calculate new metrics
    df_aggregate = aggregate_data(segmented_data)

    return df_aggregate
