import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO process all the data required in the questions and regenerate the plots

# QUESTION 1

def get_all_data(data_folder):
    """
    Reads all CSV files in a folder, transforms them into DataFrames, and concatenates them into a single DataFrame.
    
    Parameters:
        data_folder (str): The path to the folder containing the processed CSV files.

    Returns:
        pd.DataFrame: A single DataFrame containing all the data from the CSV files.
    """
    # Initialize an empty list to store DataFrames
    all_data = []

    # Loop through all files in the specified directory
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            # Read each CSV file and append it to the list
            df = pd.read_csv(file_path)
            all_data.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    season_data = pd.concat(all_data, ignore_index=True)

    return season_data


def tally_shots_and_goals(season_data):
    """
    Reads the DataFrame of an entire season, and tallies up the shots and goals by shot type.
    
    Parameters:
        pd.DataFrame: A single DataFrame containing all the data from the CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the tally of shots and goals for each shot type.
    """
    # Filter for shot-on-goal and goal events
    filtered_data = season_data[season_data['event_type'].isin(['shot-on-goal', 'goal'])]

    # Group by shot type and tally shots and goals
    tally = filtered_data.groupby('shot_type').agg(
        shots=('event_id', 'count'),
        goals=('event_type', lambda x: (x == 'goal').sum())
    ).reset_index()

    return tally


def plot_shot_types_vs_goals(tally_data):
    """
    Plots a stacked bar graph showing shot types, with the number of goals and shots-on-goal for each shot type.

    Parameters:
        tally_data (pd.DataFrame): DataFrame containing the tally of shots and goals for each shot type.
    """
    # Calculate the number of shot-on-goals (excluding goals)
    tally_data['shots_on_goal_only'] = tally_data['shots'] - tally_data['goals']

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))

    # Bar for goals (bottom part)
    plt.bar(tally_data['shot_type'], tally_data['goals'], color='red', label='Goals')

    # Bar for shot-on-goals (top part of the bar)
    plt.bar(tally_data['shot_type'], tally_data['shots_on_goal_only'], bottom=tally_data['goals'], color='blue', label='Shots (excluding goals)')

    # Add labels and title
    plt.title('Comparison of Shot Types: Goals vs Shots-on-Goal', fontsize=14)
    plt.xlabel('Shot Type', fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()



# QUESTION 2


def determine_goal(team_shots):
    """
    Determines the goal location for a team in a specific period based on their shots.

    Parameters:
        team_shots (pd.DataFrame): DataFrame containing all the shots for the team in the period.

    Returns:
        tuple: Coordinates (x, y) of the goal. If no valid goal can be determined, returns None.
    """
    # Loop through the shots to find the first where zone_code isn't 'N'
    for index, row in team_shots.iterrows():
        x_coord = row['x_coord']
        zone_code = row['zone_code']
        
        if zone_code == 'O':  # Offensive zone
            return (90, 0) if x_coord > 0 else (-90, 0)
        elif zone_code == 'D':  # Defensive zone
            return (-90, 0) if x_coord > 0 else (90, 0)
    
    # If no valid offensive or defensive zone shot is found, return None (e.g., all shots in neutral zone)
    return None
    

def calculate_distance_to_goal(x_coord, y_coord, goal_coords):
    """
    Calculates the Euclidean distance from the shot position to the goal.
    
    Parameters:
        x_coord (float): The x-coordinate of the player when taking the shot.
        y_coord (float): The y-coordinate of the player when taking the shot.
        goal_coords (tuple): The (x, y) coordinates of the goal.

    Returns:
        float: The distance between the shot and the goal.
    """
    if goal_coords is None:
        return None  # No valid distance if no goal coordinates are defined
    goal_x, goal_y = goal_coords
    return np.sqrt((x_coord - goal_x) ** 2 + (y_coord - goal_y) ** 2)


def process_game(file_path):
    """
    Processes a single game, divides shots by period and team, and calculates the distance for each shot.
    
    Parameters:
        file_path (str): Path to the game CSV file.

    Returns:
        pd.DataFrame: DataFrame containing shots, distances, and goal outcomes for each team in the game.
    """
    # Read the game data
    game_data = pd.read_csv(file_path)
    
    # Initialize list to store processed data
    processed_data = []

    # Group by period first, then by team within each period
    
    for period, period_data in game_data.groupby('period'):
        for team_id, team_data in period_data.groupby('team_id'):
            # Determine the goal once for the team in this period
            goal_coords = determine_goal(team_data)
            
            # Process each shot for the team in this period
            for index, row in team_data.iterrows():
                x_coord = row['x_coord']
                y_coord = row['y_coord']
                
                # Calculate the distance to the goal
                distance = calculate_distance_to_goal(x_coord, y_coord, goal_coords)
                
                # Check if the event was a goal
                is_goal = 1 if row['event_type'] == 'goal' else 0
                
                # Append the shot's information to the processed list
                processed_data.append({
                    'game_id': row['game_id'],
                    'team_id': team_id,
                    'event_id': row['event_id'],
                    'distance': distance,
                    'is_goal': is_goal,
                    'period_number': period,
                    'zone_code': row['zone_code'],
                    'shot_type': row['shot_type'],
                })
    
    # Convert processed data into a DataFrame
    return pd.DataFrame(processed_data)


def process_all_games(data_folder):
    """
    Processes all games in a folder, calculating distances and goal outcomes for every shot.

    Parameters:
        data_folder (str): Path to the folder containing all the game CSV files.

    Returns:
        pd.DataFrame: DataFrame containing processed shot data for all games.
    """
    all_processed_data = []

    # Loop through all files in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            # Process the game and append the data
            game_data = process_game(file_path)
            all_processed_data.append(game_data)
    
    # Concatenate all game data into a single DataFrame
    return pd.concat(all_processed_data, ignore_index=True)


def plot_shot_outcomes_by_distance(distill_data):
    """
    Plots a bar graph showing the number of goals vs misses for different shot distance intervals.

    Parameters:
        distill_data (pd.DataFrame): DataFrame containing shot distances and goal outcomes.
    """
    # Create 5-unit distance bins (intervals)
    distill_data['distance_bin'] = pd.cut(distill_data['distance'], bins=range(0, int(distill_data['distance'].max()) + 5, 5))

    # Group by distance bin and tally up the number of goals and misses
    distance_data = distill_data.groupby('distance_bin').agg(
        total_shots=('is_goal', 'count'),
        total_goals=('is_goal', 'sum')
    ).reset_index()

    # Calculate the number of misses
    distance_data['total_misses'] = distance_data['total_shots'] - distance_data['total_goals']

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    
    # Bar for goals (bottom part of the stack)
    plt.bar(distance_data['distance_bin'].astype(str), distance_data['total_goals'], color='green', label='Goals')

    # Bar for misses (top part of the stack)
    plt.bar(distance_data['distance_bin'].astype(str), distance_data['total_misses'], bottom=distance_data['total_goals'], color='blue', label='Misses')

    # Add labels and title
    plt.title('Shot Outcomes by Distance Intervals', fontsize=14)
    plt.xlabel('Distance from Goal (5-Unit Intervals)', fontsize=12)
    plt.ylabel('Number of Shots', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def prepare_goal_percentage_data(shots_data):
    """
    Prepares the data for the goal percentage plot, grouping by shot type and distance.

    Parameters:
        shots_data (pd.DataFrame): The shot data for a given season.

    Returns:
        pd.DataFrame: Pivot table with shot types as rows, distance bins as columns, and goal percentage as values.
    """
    # Bin distances into 5-unit intervals
    shots_data['distance_bin'] = pd.cut(shots_data['distance'], bins=range(0, int(shots_data['distance'].max()) + 5, 5))

    # Group by shot type and distance bin
    grouped_data = shots_data.groupby(['shot_type', 'distance_bin']).agg(
        total_shots=('event_id', 'count'),
        total_goals=('is_goal', 'sum')  # Sum up the goals
    ).reset_index()

    # Calculate goal percentage
    grouped_data['goal_percentage'] = grouped_data['total_goals'] / grouped_data['total_shots']

    # Pivot the table for plotting
    pivot_table = grouped_data.pivot(index='shot_type', columns='distance_bin', values='goal_percentage')

    return pivot_table


# QUESTION 3


def plot_goal_percentage_heatmap(pivot_table):
    """
    Plots a heatmap of goal percentage as a function of shot type and distance using a logarithmic color scale.

    Parameters:
        pivot_table (pd.DataFrame): Pivot table with goal percentages.
    """
    # Apply logarithmic transformation to the values in the pivot table, adding a small value to avoid log(0)
    pivot_table_log = np.log10(pivot_table + 1e-3)  # Adding a small constant to avoid log(0)

    # Create the heatmap with logarithmic scale
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table_log, 
        annot=pivot_table,  # Show the original values in the heatmap
        cmap="coolwarm", 
        fmt=".2f",  # Format the annotations to show two significant digits in decimal
        linewidths=.5,
        cbar_kws={'label': 'Log Goal Percentage'}
    )

    # Add labels and title
    plt.title('Goal Percentage by Shot Type and Distance (Logarithmic Scale)', fontsize=16)
    plt.xlabel('Distance from Net (Binned)', fontsize=12)
    plt.ylabel('Shot Type', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

