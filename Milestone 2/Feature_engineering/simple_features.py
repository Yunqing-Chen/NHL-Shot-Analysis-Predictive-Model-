import os
import pandas as pd
import numpy as np
import pathlib


def load_season_data(data_folder):
    """
    Reads all CSV files for a given season into a single DataFrame.

    Parameters:
        data_folder (str): Path to the folder containing the season's game CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame containing all shots from the season.
    """
    all_data = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            game_data = pd.read_csv(file_path)
            all_data.append(game_data)

    # Combine all game data into a single DataFrame
    season_data = pd.concat(all_data, ignore_index=True)
    return season_data


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


def augment_data(input_path, output_path):
    """
    Augments the raw data with new features and saves the augmented data to a CSV file.

    Parameters:
        input_path (pathlib.Path): Path to the directory containing the raw data CSV files.
        output_path (pathlib.Path): Path to the directory where augmented data will be saved.

    Returns:
        None
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "augmented_data.csv"

    if output_file.exists():
        print(f'File {output_file} already exists. Skipping')
        df_aggregate = pd.read_csv(output_file)
    else:
        print(f'Processing {input_path}..')
        all_data = load_season_data(input_path)
        # Segment the data by team, game, and period
        segmented_data = segment_shot_data(all_data)
        # Aggregate the data and calculate new metrics
        df_aggregate = aggregate_data(segmented_data)
        df_aggregate.to_csv(output_file, index=False)
        print(f"Augmented data saved to {output_file}")

    return df_aggregate


# Augment and combine all the data from 2016 - 2019
def augment_dataset():
    # Paths for input and output directories
    years = [2016, 2017, 2018, 2019]
    df_aggregate = pd.DataFrame()
    all_data = []
    input_directory = "../dataset/processed"
    output_directory = "../dataset/simple_engineered"

    for year in years:
        input_path = os.path.join(input_directory, str(year))
        output_path = os.path.join(output_directory, str(year))
        # Process the data
        all_data.append(augment_data(input_path, output_path))

    df_aggregate = pd.concat(all_data, ignore_index=True)
    output_file_path = pathlib.Path(output_directory)
    output_file = output_file_path / "augmented_data.csv"
    df_aggregate.to_csv(output_file, index=False)
    print(f"Combined augmented data saved to {output_file}")

    return df_aggregate
