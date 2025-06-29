# implement interactive debugging tool

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
from data_acquisition import NHLDataFetcher
import pprint
import matplotlib.image as mpimg
from typing import Union
from textwrap import dedent
from pathlib import Path
import os
import json

class NHLDataViewer:
    def __init__(self, save_dir = None):
          self.save_dir = save_dir or self._get_save_dir()
          self.nhl_data = NHLDataFetcher(base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play", save_dir=self.save_dir)
          self.select_season = ''
          self.select_game_type = ''
          self.event_intslider = widgets.IntSlider(
                value=1,
                min=1,
                max=1000
        )

    def _get_save_dir(self):
        """
        Returns `save_dir` for setting NHLDataFetcher save_dir argument
        save_dir is the path to unprocessed raw JSON of game data
        Assumed to be in {ROOT_DIR}/dataset/unprocessed/
        """
        #Hacky way of getting file's path
        #Seems robust when ran as __main__ and imported as module
        if __file__:
            file_path = Path(__file__).absolute()
            #Assumes current file one folder (data/) away from root path
            #root path containing folder dataset/ with unprocessed JSON
            #in dataset/unprocessed/
            root_dir = file_path.parents[1]
            save_dir = root_dir.joinpath('dataset', 'unprocessed').absolute()
            assert save_dir.is_dir(), f'Specified dataset dir {save_dir} does not exists'
            return save_dir

    def visualize_events(self,season, game_type, game_id, i):

        if(game_type == 'regular'):
            game_type_value = '02'
            game_id =  f"game_{season}{game_type_value}{str(game_id).zfill(4)}"
        else:
            game_type_value = '03'
            # Get a list of files in the current directory that start with "2013"
            file_prefix = f"game_{season}{game_type_value}"
            files = [f for f in os.listdir(self.nhl_data.save_dir.joinpath(season)) if f.startswith(file_prefix)]
            # Sort the list to maintain order
            files.sort()
            # Get the ith file
            ith_file=files[game_id-1]
            # remove suffix
            game_id= os.path.splitext(ith_file)[0]


        # get game data through the file name
        game_id = game_id[5:]
        game_data = self.nhl_data.get_game_data(game_id)

        if( game_data[0] != None):
            if type(game_data[0]) is dict:
                game_data = game_data[0]
            elif type(game_data[0]) is str:
                game_data = json.loads(game_data[0])
            # time_label = widgets.Label(value=game_data['startTimeUTC'])
            # title_label = widgets.Label(value=f"Game ID: {game_data['id'] }; {game_data['homeTeam']['abbrev']}(home) VS {game_data['awayTeam']['abbrev']}(away)")
            # display(time_label,title_label)

            print(game_data['startTimeUTC'])
            print(f"Game ID: {game_data['id']}; {game_data['homeTeam']['abbrev']} (home) vs {game_data['awayTeam']['abbrev']} (away)")

            col1 = ['', 'Teams', 'Goals', 'SoG']
            col2 = ["Home",f"{game_data['homeTeam']['abbrev']}", f"{game_data['homeTeam']['score']}", f"{game_data['homeTeam']['sog']}"]
            col3 = ["Away",f"{game_data['awayTeam']['abbrev']}", f"{game_data['awayTeam']['score']}", f"{game_data['awayTeam']['sog']}"]
            print('')
            for c1, c2, c3 in zip(col1, col2, col3):
                print(f'{c1:<18} {c2:<18} {c3:<18}')


            event_count = len(game_data['plays'])

            self.event_intslider.max=event_count

            event_id = i
            event_data = game_data['plays'][event_id-1]


            fig, ax = plt.subplots()
            image_path = 'patinoire.png'
            img = mpimg.imread(image_path)
            img_height, img_width = img.shape[0], img.shape[1]


            ax.imshow(img, extent=[-100, 100, -42.5, 42.5], origin='lower')


            ax.spines['left'].set_position(('axes', 0))
            ax.spines['bottom'].set_position(('axes', 0))

            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            if('details' in event_data and 'xCoord' in event_data['details'] and 'yCoord' in event_data['details']):
                ax.scatter(event_data['details']['xCoord'], event_data['details']['yCoord'], color="blue", s=100, zorder=5)

            y_min, y_max = plt.ylim()
            home_team_position_x = 40
            away_team_position_x = -60
            if 'homeTeamDefendingSide' in event_data:
                if event_data['homeTeamDefendingSide'] == 'right':
                    home_team_position_x = 40
                    away_team_position_x = -60
                else:
                    home_team_position_x = -60
                    away_team_position_x = 40

            plt.text(home_team_position_x, y_max, game_data['homeTeam']['abbrev'], fontsize=12, verticalalignment='bottom')
            plt.text(away_team_position_x, y_max, game_data['awayTeam']['abbrev'], fontsize=12, verticalalignment='bottom')
            plt.show()
            pprint.pprint(event_data)


    def create_interactive_viewer(self):
        season_dropdown = widgets.Dropdown(
            options=[str(year) for year in range(2015,2024)],
            description='Season:',
        )

        game_type_dropdown = widgets.Dropdown(
            options=['regular','playoff'],
            description='Game Type:',
        )

        game_intslider = widgets.IntSlider(
            value=1,
            min=1,
            max=1,
            description ='Game ID',
            continuous_update=True
        )
        max_game_label = widgets.Label(value="Max Game ID:")

        def update_game_dropdown(change):
            season = change['new']
            game_type = game_type_dropdown.value
            game_type_value = '03'
            if(game_type_dropdown.value == 'regular'):
                game_type_value = '02'
            file_prefix = f"game_{season}{game_type_value}"
            max_number = len([f for f in os.listdir(self.nhl_data.save_dir.joinpath(season)) if f.startswith(file_prefix)])
            if(max_number <= 0):
                game_intslider.disabled = True
            else:
                game_intslider.disabled = False
                game_intslider.max = max_number

            max_game_label.value = f"Max Game ID: {max_number}"

        def update_game_dropdown_event_type(change):
            update_game_dropdown({'new': season_dropdown.value})

        season_dropdown.observe(update_game_dropdown, names='value')
        game_type_dropdown.observe(update_game_dropdown_event_type, names='value')
        update_game_dropdown({'new': season_dropdown.value})
        display(max_game_label)

        widgets.interact(self.visualize_events,season=season_dropdown, game_type=game_type_dropdown, game_id=game_intslider, i=self.event_intslider)
