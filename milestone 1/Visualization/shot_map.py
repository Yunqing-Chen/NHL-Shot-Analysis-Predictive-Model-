import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from PIL import Image



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
                segmented_data[team_id][game_id][period] = calculate_offensive_zone(period_data)

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
            return (0, 90) if x_coord > 0 else (-90, 0)
        elif zone_code == 'D':  # Defensive zone
            return (-90, 0) if x_coord > 0 else (0, 90)
    
    # Default to offensive goal at (0, 90) if no offensive/defensive zone shots found
    return (0, 90)


def calculate_offensive_zone(shots_data):
    """
    Adjusts shot coordinates to place the offensive zone goal at (0, 90).
    If the goal is at (-90, 0), flip all shot coordinates about the y-axis.

    Parameters:
        shots_data (pd.DataFrame): DataFrame containing shots for a team in a period.

    Returns:
        pd.DataFrame: DataFrame with adjusted shot coordinates.
    """
    adjusted_shots = []
    
    # Determine the goal location based on the shots in the period
    goal_location = determine_goal_location(shots_data)
    
    # If the goal is at (-90, 0), we will flip all shots along the y-axis
    flip = goal_location == (-90, 0)

    for index, row in shots_data.iterrows():
        x_coord, y_coord = row['x_coord'], row['y_coord']

        # Adjust shots by flipping if necessary
        if flip:
            adjusted_x = -x_coord
            adjusted_y = -y_coord
        else:
            adjusted_x = x_coord
            adjusted_y = y_coord

        adjusted_shots.append({
            'game_id': row['game_id'],
            'team_id': row['team_id'],
            'x_coord': adjusted_x,
            'y_coord': adjusted_y,
            'event_type': row['event_type'],
            'shot_type': row['shot_type']
        })

    return pd.DataFrame(adjusted_shots)


def aggregate_data(segmented_data, team=None):
    """
    Aggregates data into a single DataFrame either for the entire league or for individual teams.

    Parameters:
        segmented_data (dict): Nested dictionary where data is segmented by team, game, and period.

    Returns:
        pd.DataFrame: Combined DataFrame containing the aggregate data.
    """
    aggregate = []

    for team_id, games in segmented_data.items():
        if team is None or team == team_id: # Aggregate all teams or a specific team
            for game_id, periods in games.items():
                for period_data in periods.values():
                    aggregate.append(period_data)

    # Combine all data into a single DataFrame
    return pd.concat(aggregate, ignore_index=True)


def calculate_kde(segmented_data, grid_size=100, bw_adjust=0.5, team_id=None):
    """
    Calculates the average KDE.

    Parameters:
        segmented_data (dict): Nested dictionary where data is segmented by team, game, and period.
        grid_size (int): The resolution of the grid on which the KDE is evaluated.
        bw_adjust (float): Bandwidth adjustment for KDE.
        team_id (int): The ID of the team to calculate the KDE for. If None, the league KDE is calculated.

    Returns:
        tuple: A grid of x, y coordinates and the KDE normalized by the number of games.
    """
    # Aggregate the data
    data = aggregate_data(segmented_data, team_id)
    
    # Extract shot coordinates
    x = data['x_coord']
    y = data['y_coord']
    
    # Set up the grid
    x_grid = np.linspace(-100, 100, grid_size)
    y_grid = np.linspace(-42.5, 42.5, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    
    coords = np.vstack([x, y])

    # Filter out invalid coordinates (NaN and inf entries)
    valid_mask = ~np.isnan(coords).any(axis=0) & ~np.isinf(coords).any(axis=0)
    valid_coords = coords[:, valid_mask]

    # Print how many entries were removed (if any)
    if len(valid_coords[0]) < len(x):
        print(f"Removed {len(x) - len(valid_coords[0])} invalid entries containing NaN or inf.")

    # Perform the KDE for the league
    kde = gaussian_kde(valid_coords, bw_method=bw_adjust)
    kde_values = np.reshape(kde(positions).T, x_mesh.shape)
    
    return x_grid, y_grid, kde_values

 
def calculate_percentage_kde_difference(team_kde, league_kde):
    """
    Calculates the percentage difference between the team's KDE and the league-wide KDE.

    Parameters:
        team_kde (numpy.ndarray): The team's KDE values.
        league_kde (numpy.ndarray): The league-wide KDE values.

    Returns:
        numpy.ndarray: The percentage difference in KDE values.
    """
    # Avoid division by zero by adding a small constant to the league KDE
    league_kde_safe = np.where(league_kde == 0, 1e-6, league_kde)
    
    # Calculate the percentage difference
    percentage_difference = ((team_kde - league_kde) / league_kde_safe) * 100
    
    return percentage_difference


def calculate_kde_differences_for_teams(segmented_data, grid_size=100, bw_adjust=0.5):
    """
    Calculates the KDE differences (team KDE - league KDE) for all teams.

    Parameters:
        segmented_data (dict): Nested dictionary where data is segmented by team, game, and period.
        grid_size (int): Resolution of the grid for KDE.
        bw_adjust (float): Bandwidth adjustment for the KDE.

    Returns:
        dict: A dictionary containing the KDE differences for each team (team_id as the key).
        numpy.ndarray: The x_grid and y_grid for the KDE plot.
    """
    # Calculate the league KDE
    x_grid, y_grid, league_kde = calculate_kde(segmented_data, grid_size, bw_adjust)

    kde_differences = {}

    # Loop through each team and calculate the KDE difference
    for team_id in segmented_data.keys():
        _, _, team_kde = calculate_kde(segmented_data, grid_size, bw_adjust, team_id)
        # kde_diff = calculate_percentage_kde_difference(team_kde, league_kde) # Calculate percentage difference
        kde_diff = team_kde - league_kde # Calculate raw difference

        kde_differences["Team " + str(team_id) + " Diff W/ League"] = kde_diff
        kde_differences["Team " + str(team_id) + " Shot Map"] = team_kde

    kde_differences["League Average Shot Map"] = league_kde

    return kde_differences, x_grid, y_grid


def create_interactive_kde_plot(kde_differences, x_grid, y_grid, rink_image):
    """
    Creates an interactive KDE shot map plot with a dropdown menu to switch between teams, showing differences.
    Handles the offensive zone filtering, axis rotation, and adds a hockey rink image to the background.

    Parameters:
        kde_differences (dict): Dictionary containing the KDE differences (not yet rotated) for each team.
        x_grid (numpy.ndarray): The x-axis grid values for the KDE plot (not yet rotated).
        y_grid (numpy.ndarray): The y-axis grid values for the KDE plot (not yet rotated).
        rink_image_path (str): Path to the hockey rink image.
    """
    fig = go.Figure()

    # Initialize first team_id
    first_team_id = list(kde_differences.keys())[0]
    first_kde_diff = kde_differences[first_team_id]

    # Step 1: Rotate axes for visualization (goal at the top)
    x_new = y_grid  # Old Y-axis becomes new X-axis
    y_new = x_grid  # Old X-axis becomes new Y-axis

    # Step 2: Add the first KDE difference as the initial heatmap
    fig.add_trace(go.Heatmap(
        z=first_kde_diff.T,  # Transposed to match the new orientation
        x=x_new,
        y=y_new,
        colorscale='RdBu',
        zmin=-np.max(np.abs(first_kde_diff)),
        zmax=np.max(np.abs(first_kde_diff)),
        colorbar=dict(title="Shot Density Difference"),
        opacity=0.6
    ))

    # Step 3: Add the hockey rink image to the background
    fig.update_layout(
        images=[{
            'source': rink_image,
            'xref': 'x',
            'yref': 'y',
            'x': -42.5,  # Align with the new X-axis
            'y': 100,    # Align with the new Y-axis
            'sizex': 85,  # The width of the rink
            'sizey': 100,  # The length of the rink
            'sizing': 'stretch',
            'opacity': 1.0,  # Slightly transparent
            'layer': 'below'
        }]
    )


    # Step 4: Create the dropdown menu for selecting teams
    dropdown_buttons = []
    for team_id, kde_diff in kde_differences.items():

        dropdown_buttons.append({
            'args': [{'z': [kde_diff.T], 'zmin': -np.max(np.abs(kde_diff)), 'zmax': np.max(np.abs(kde_diff))}],
            'label': f'{team_id}',
            'method': 'restyle'
        })

    # Step 5: Update layout for dropdown and axis labels, also filters out the defensive zone
    fig.update_layout(
        title='KDE Shot Map Difference (Team vs League Average)',
        xaxis=dict(title='X Coordinate (Feet)', range=[-42.5, 42.5]),
        yaxis=dict(title='Y Coordinate (Feet)', range=[0, 100]),
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        # Set aspect ratio to 85:100 for rink proportions
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1.1765  # 100 / 85 = 1.1765
    )

    # Step 6: Show the plot
    fig.show()
