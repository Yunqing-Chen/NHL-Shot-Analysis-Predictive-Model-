# TODO
# Still need to include calculations for when the goal is at even strength, shorthanded, or power play

import os
import json
import pathlib
import re
import pandas as pd

def parse_game_events(game_data: dict) -> pd.DataFrame:
    """
    Parses the JSON response of a game to extract 'shot-on-goal' and 'goal' events and converts them into a Pandas DataFrame.

    Parameters:
        game_data (dict): JSON response of a single game's events.

    Returns:
        pd.DataFrame: A dataframe containing the filtered and formatted events data.
    """
    events = game_data.get('plays', [])
    game_id = game_data.get('id', '')

    # List to store parsed events
    event_data = []

    # Loop through each event in the game data
    for event in events:
        # Filter for shot-on-goal or goal
        event_type = event.get('typeDescKey', '')
        if event_type in ['shot-on-goal', 'goal']:
            # Extract time, period, and event fields
            time_in_period = event.get('timeInPeriod', None)
            time_remaining = event.get('timeRemaining', None)
            period = event.get('periodDescriptor', None).get('number', None)
            event_id = event.get('eventId', None)

            # Team that took the shot
            team_id = event.get('details', {}).get('eventOwnerTeamId', None)

            # Coordinates of the shot
            x_coord = event.get('details', {}).get('xCoord', None)
            y_coord = event.get('details', {}).get('yCoord', None)

            # Shooter and goalie info
            shooter_id = event.get('details', {}).get('shootingPlayerId', None)
            scoring_player_id = event.get('details', {}).get('scoringPlayerId', None)
            goalie_id = event.get('details', {}).get('goalieInNetId', None)

            # Shot type
            shot_type = event.get('details', {}).get('shotType', None)

            # Zone Code for figuring out which side the team is on
            zone_code = event.get('details', {}).get('zoneCode', None)


            # Check if it was on an empty net (no goalie present)
            empty_net = goalie_id is None

            # Append the event data to the list
            event_data.append({
                'game_id': game_id,
                'event_id': event_id,
                'event_type': event_type,
                'period': period,
                'time_in_period': time_in_period,
                'time_remaining': time_remaining,
                'team_id': team_id,
                'x_coord': x_coord,
                'y_coord': y_coord,
                'shooter_id': shooter_id or scoring_player_id,  # Use scoring player ID if it's a goal
                'goalie_id': goalie_id,
                'shot_type': shot_type,
                'empty_net': empty_net,
                'zone_code': zone_code
            })

    # Convert the list of events into a Pandas DataFrame
    df = pd.DataFrame(event_data)

    return df


def process_and_save_json_file(DATA_INPUT_PATH : pathlib.Path, DATA_OUTPUT_PATH : pathlib.Path) -> None:
    """
    Process all .json files found in DATA_INPUT_PATH, convert to Pandas DataFrame and save them to csv in DATA_OUTPUT_PATH

    Parameters:
        DATA_INPUT_PATH (pathlib.Path) : Input directory of unprocessed data
                                         Assumes the following hierarchy : DATA_INPUT_PATH/{season_folder}/{gameid}*.json
        DATA_OUTPUT_PATH (pathlib.Path) : Output directory of processed (DataFrames) data
                                          Copies hierarchy of DATA_INPUT_PATH for naming output csvs

    Returns:
        None
    """
    for game_json_file in DATA_INPUT_PATH.rglob("**/game*.json"):
        game_title = game_json_file.parts[-1]
        game_title_csv = re.sub('json$', 'csv', game_title)
        season_folder = game_json_file.parts[-2]
        output_file = DATA_OUTPUT_PATH.joinpath(season_folder, game_title_csv)
        #Check if processed file already exists
        if output_file.exists():
            print(f'File {output_file} already exists. Skipping')
        else:
            #Check if DATA_OUTPUT_PATH/season_folder exists, else create it
            if not output_file.parent.exists():
                os.mkdir(output_file.parent)
            with open(game_json_file, 'r') as open_file:
                print(f'Processing {game_json_file}..')
                game_dict = json.load(open_file)
                df_game = parse_game_events(game_dict)
                df_game.to_csv(output_file)
                print(f'Saved csv of dataframe to {output_file}')



def main():
  DATA_INPUT_PATH, DATA_OUTPUT_PATH = gather_and_check_paths()
  process_and_save_json_file(DATA_INPUT_PATH, DATA_OUTPUT_PATH)

if __name__ == '__main__' :
		#This file is now part of pipeline ../fetch_and_tidy.py
		#Utils functions such as gather_and_check_paths have been moved to ../fetch_and_tidy.py
		#Executing as a main script will fail
		raise RuntimeError('This file is not meant to be executed as a standalone script')
