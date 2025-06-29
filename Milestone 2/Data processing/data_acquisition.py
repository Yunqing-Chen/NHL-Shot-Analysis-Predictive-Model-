# TODO
# Reading into the API and automatically generate the game_ids
# _add_ functionality to be able to aggregate data across multiple seasons

import os
import requests
import time
import json
from sys import argv

class NHLDataFetcher:
    def __init__(self, base_url, save_dir=None):
        # Use environment variable for file path if provided
        self.get_root_path = self._get_root_path()
        self.save_dir = os.getenv('NHL_DATA_INPUT_PATH') or save_dir or f'{self.get_root_path}/dataset/unprocessed/'
        self.base_url = base_url

    def _get_exec_path(self):
        return os.path.abspath(argv[0]).split('/')[:-1]

    def _get_root_path(self):
        self.exec_path = self._get_exec_path()
        return '/'.join(self.exec_path[:-1])

    def get_playoffs_game_data(self, game_id):
        """
        Transform game_id by adding necessary offset before sending
        to get_game_data()
        Ex (Querying the first game of 2022 playoffs)
             game_id 202203[0001] --> 202203[0111]
             game_id 202203[0010] --> 202203[0123]
        for round_number in range(1, 5):  # Playoffs have 4 rounds
        for matchup_number in range(1, 9):  # Each round has up to 8 matchups
        for game_number in range(1, 8):  # Each matchup has up to 7 games
        game_id = f"{year}030{round_number}{matchup_number}{game_number}"
        """
        nth_game = int(game_id[-4:])

        def playoffs_gameid_generator(game: int):
            round = 1
            matchup = 1
            res = 0
            while True:
                res = game - 7
                if res < 1:
                    break
                else:
                    if matchup < 8:
                        matchup += 1
                    else:
                        matchup = 1
                        if round < 4:
                            round += 1
                        else:
                            print(f'{game}th playoff game does not exist')
                            return None
                game = res
            return f'0{round}{matchup}{game}'

        while True:
            #Match nth_game to potential playoff_game_id
            playoff_game_id = playoffs_gameid_generator(nth_game)
            if playoff_game_id is None:
                return
            #Truncate last 4 char corresponding to nth_game wanted
            game_id = game_id[:-4]
            #Replace with potential playoff_game_id
            game_id = game_id + playoff_game_id
            #Try downloading corresponding game
            ret, _ = self.get_game_data(game_id)
            if ret is not None:
                break
            else :
                nth_game += 1
       # while ret is None:
       #     #If not available, increment nth_game and fetch new potential
       #     #playoff_game_id
       #     nth_game += 1
       #     playoff_game_id = playoffs_gameid_generator(nth_game)
       #     game_id = game_id[:-4]
       #     game_id = game_id + playoff_game_id


    def get_game_data(self, game_id):
        """
        Downloads a single game for the given game_id.
        """
        # Create file path
        season = game_id[:4]
        season_folder = os.path.join(self.save_dir, season)
        filename = os.path.join(season_folder, f'game_{game_id}.json')


        # Check if the data is already cached
        if os.path.exists(filename):
            print(f"Loading cached data for game ID: {game_id}")
            with open(filename, 'r') as file:
                #Return JSON object (dict) to feed to ipywidget debugger
                #Can also return str with file.read() in other cases
                json_obj = json.load(file)
                return json_obj, False

        # Otherwise download the data
        url = self.base_url.format(game_id)
        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            if 'error' in data:
                print(f"Error found in response for game ID {game_id}: {data['error']}")
                return None, True

            os.makedirs(season_folder, exist_ok=True)  # Ensure directory exists
            with open(filename, 'w') as file:
                file.write(response.text)

            print(f"Successfully downloaded and cached data for game ID: {game_id}")
            return data, False

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred for game ID {game_id}: {http_err}")
            return None, True
        except Exception as err:
            print(f"An error occurred for game ID {game_id}: {err}")
            return None, True

    def get_season_data(self, year, types: list[str]):
        """
        Downloads all games for a given NHL season (preseason, regular season, playoffs, and all-star).
        """
        if int(year) >= 2021:
            num_games = 1353
        elif int(year) >= 2017:
            num_games = 1271
        else:
            num_games = 1230

        # Preseason games
        def _get_season_preseason():
            for game_number in range(1, num_games+1):
                game_id = f"{year}01{game_number:04d}"
                data, error = self.get_game_data(game_id)
                if error:
                    print(f"Stopping download of Preseason Games due to an error with game ID: {game_id}")
                    break
                time.sleep(1)

        # Regular season games
        def _get_season_regular():
            for game_number in range(1, num_games+1):
                game_id = f"{year}02{game_number:04d}"
                data, error = self.get_game_data(game_id)
                if error:
                    print(f"Stopping download of Regular Season Games due to an error with game ID: {game_id}")
                    break
                time.sleep(1)

        # Playoff games
        def _get_season_playoffs():
            for round_number in range(1, 5):  # Playoffs have 4 rounds
                for matchup_number in range(1, 9):  # Each round has up to 8 matchups
                    for game_number in range(1, 8):  # Each matchup has up to 7 games
                        game_id = f"{year}030{round_number}{matchup_number}{game_number}"
                        data, error = self.get_game_data(game_id)
                        if error:
                            print(f"Stopping download of Playoff Games due to an error with game ID: {game_id}")
                            break
                        time.sleep(1)

        # All-star game
        def _get_season_allstar():
            data, error = self.get_game_data(f"{year}040001")
            if error:
                print(f"Stopping download of All-Star Game due to an error with game ID: {game_id}")

            print(f"Completed downloading data for the {year}-{int(year)+1} NHL season.")

        type_dict = {
            '01': _get_season_preseason,
            '02': _get_season_regular,
            '03': _get_season_playoffs,
            '04': _get_season_allstar
        }
        for _type in types:
            type_dict[_type]()

if __name__ == '__main__':
  raise RuntimeError('This file is not meant to be executed as main script')
  # Example usage
  base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
  fetcher = NHLDataFetcher(base_url)

  game_ids = ['2022030411']

  # Fetch data for each game
  for game_id in game_ids:
      data = fetcher.get_game_data(game_id)
      time.sleep(1)  # Delay to avoid hitting the API too hard

  year = "2022"
  # Fetch data for a full season
  fetcher.get_season_data(year)
