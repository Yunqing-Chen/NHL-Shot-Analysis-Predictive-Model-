#!/usr/bin/env python3

from data_acquisition import NHLDataFetcher
from typing import Union
from textwrap import dedent
import argparse
import pathlib
import os
import re
import sys

#Instanciate NHLDataFetcher parser
def init_parser():
  parser = argparse.ArgumentParser(
    prog='NHLDataFetcher',
    description="Query and store NHL API requests",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog = dedent(f"""\
    EXAMPLES
    =========
    {sys.argv[0]} -y 2022 -t 3 -g 1-10 :
      Try downloading plyaoffs games 1 to 10 from season 2022

    {sys.argv[0]} -y 2014,2020 -t 2 -g 20,100 :
      Try downloading regular season games 20 and 100 from season 2014 and 2020

    {sys.argv[0]} :
      Try downloading all games from all seasons
    """)
  )

  parser.add_argument(
    '-y', '--year',
    type=str,
    default='all',
    help="Year of games wanted"
  )

  parser.add_argument(
    '-t', '--type',
    type=str,
    default="all",
    choices=['2', '3', '2,3', '2-3'],
    help="Type of games wanted ('regular': 2, 'playoffs': 3)"
  )

  parser.add_argument(
    '-g', '--games',
    type=str,
    default='all',
    help='Games ID wanted as interval (ex : 0-100)'
  )

  parser.add_argument(
    '-o', '--output',
    type=pathlib.Path,
    default=None,
    help=dedent(f"""Directory in which output should go to.\nDefaults to `../dataset/unprocessed`""")
  )

  parser.add_argument(
    '--parse-args',
    default=None,
    action='store_true',
    help='Check validity of passed arguments without executing the data acquisition'
  )

  return parser

#Function that look for REGEX patterns in arguments fed to parser
#Splits them based on their delimiter (`-`, `,`)
def reg_match_arg(arg: str) -> list[str]:
  try:
    #Match NUM,...,NUM series (ex: 1,3,5,9 --> ['1', '3', '5', '9'])
    if re.match(
     r"""
      \d+,(\d+,?){1,} #2+ patterns of digits with optional ',' after 1 occurrence
     """,
     arg,
     re.VERBOSE) :
      match_list = arg.split(',')
      for i in range(len(match_list) - 1 ):
        assert int(match_list[i+1]) > int(match_list[i]), f'{match_list[i+1]} !> {match_list[i]}'
      return match_list
    #Match NUM-NUM intervals (ex: 5-8 --> ['5', '6', '7', '8', '9'])
    elif re.match(
     r"""
      ^\d+    #4digits for gid_min
      \-        #dash as interval
      \d+$    #4 digits for gid_max
     """,
     arg,
     re.VERBOSE) :
      val_min, val_max = arg.split('-')
      assert int(val_max) > int(val_min), f'{val_max} !> {val_min}'
      return [ f'{val}' for val in range(int(val_min), int(val_max)+1) ]
    #Parse NUM single (ex: 1 --> ['1'])
    elif re.match(r'^\d+$', arg):
      return [arg]
  except AssertionError as e:
    print(e)

#Function that verify that arguments fed to parser are valid
#Format as appropriate strings of leading-zeros integers for API URL format
def verify_args_parser(parser, args : argparse.Namespace = None):
  if args == None:
    args = parser.parse_args()
  try:
    assert args.year, 'Missing argument year'
    assert args.type, 'Missing argument type'
    #Argument `year`
    if args.year == parser.get_default('year'):
      q_year = [ f'{g_year}' for g_year in range(2015,2024) ]
    else:
      q_year = reg_match_arg(args.year)
    #Argument `type`
    if args.type == parser.get_default('type'):
      q_type = [ f'{g_type:02d}' for g_type in (2,3) ]
    else:
      q_type = reg_match_arg(args.type)
      q_type = [ f'{int(type):02d}' for type in q_type ]
    #Argument games
    if args.games == parser.get_default('games'):
      q_games = ['1231']
    else:
      q_games = reg_match_arg(args.games)
      q_games = [ f'{int(game):04d}' for game in q_games ]
    #Global variable that controls if NHLDataFetcher will use class method `get_game_data`
    #or `get_season_data` for downloading games
    #Boolean switch logic is set at number of games > 1230 as this will be a full season
    #no matter which year it is
    #TODO : Implement stronger logic for different seasons-games pairs
    global DOWNLOAD_GAMES_DATA
    DOWNLOAD_GAMES_DATA = True
    if int(q_games[-1]) > 1230:
      DOWNLOAD_GAMES_DATA = False
    #Argument `output`
    if args.output != parser.get_default('output'):
      assert pathlib.Path.is_dir(args.output), f'{args.output} is not a directory'
      assert os.getenv('NHL_DATA_OUTPUT_PATH') is None, 'Cannot use -o|--output with set environment variable NHL_DATA_OUTPUT_PATH'
    q_dir = args.output
    return q_year, q_type, q_games, q_dir

  except (AssertionError, ValueError) as err:
    print(err)
    os.sys.exit()

def main():
  #Instanciate parser
  parser = init_parser()
  #Parse arguments
  sys_args = sys.argv[1:]
  args = parser.parse_args(sys_args)
  #Check if valid arguments
  q_year, q_type, q_games, q_dir = verify_args_parser(parser, args)
  #If called with arg `--parse-args`, then return after successfully checking passed args
  if args.parse_args:
    return
  #Instanciate fetcher
  base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
  fetcher = NHLDataFetcher(base_url, save_dir = q_dir or None)
  #Execute fetcher
  if DOWNLOAD_GAMES_DATA:         #Fetch game-by-game if less than 1230 games required
    for season in q_year:
      for type in q_type:
        for game in q_games:
          if type == '03':
            fetcher.get_playoffs_game_data(f'{season}{type}{game}')
          else:
            fetcher.get_game_data(f'{season}{type}{game}')
  else:                           #Fetch season-by-season if more than 1230 games required
    for season in q_year:
      fetcher.get_season_data(season, q_type)


if __name__ == '__main__':
  main()
