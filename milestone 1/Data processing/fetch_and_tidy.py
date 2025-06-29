#!/usr/bin/env python3

import subprocess
import os
import pathlib
import json
import random
import features.data_formatting
from re import match
from time import sleep
from typing import Tuple

def subprocess_popen_cmanager(args: list, timeout=30, verbose=False):
    """
    Passes `args` to subprocess.Popen and defines a context manager around it

    Parameters:
        args: list, list of command arguments
        timeout: int, timeout in seconds

    Return:
        p: subprocess.Popen, finished process of args
    """
    if verbose:
        print(f'Opening process for command {args}')
        print(f'Setting timeout to {timeout} seconds')
    try:
      p = subprocess.Popen(args)
      p.communicate(timeout=timeout)
    except KeyboardInterrupt:
      print('CTRL-C catched, terminating program')
      p.terminate()
      sleep(5)
      #If SIGTERM does not do the trick, force raise TimeoutExpired
      if not p.poll():
          raise p.TimeoutExpired()
    except subprocess.TimeoutExpired:
        print('Timeout for program execution reached, killing program')
        p.kill()
    finally:
        return p


def _gather_and_check_paths(DATA_INPUT_PATH: str = None, DATA_OUTPUT_PATH: str = None, return_root_dir=False) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Gather DATA_INPUT_PATH and DATA_OUTPUT_PATH in the following order:
      Argument fed in the function, else
      System Environment Varaible as 'NHL_DATA_[INPUT|OUTPUT]_PATH', else
      Default directory '{ROOT_DIR}/dataset/[unprocessed|processed]' with ROOT_DIR as root directory of currently executed script

      Parameters:
          DATA_INPUT_PATH·(str) : Input directory of unprocessed data
          DATA_OUTPUT_PATH·(str) : Output directory of processed (DataFrames) data
          return_root_dir (bool) : Whether to return ROOT_DIR or not
      Returns:
          DATA_INPUT_PATH (pathlib.Path)
          DATA_OUTPUT_PATH (pathlib.Path)
          ROOT_DIR (pathlib.Path)
    """

    #Get absolute path of currently run script
    ##From https://stackoverflow.com/a/595317: sys.argv[0] => will not work if import file as module
    ##Experimenting with __file__ which seems to be more consistant
    EXEC_PY_PATH = pathlib.Path(__file__).absolute()
    #Setting dir 'ift6758' as ROOT_DIR
    ROOT_DIR = EXEC_PY_PATH.parents[0]
    #Check if DATA_INPUT_PATH and DATA_OUTPUT_PATH have been set and get their absolute paths
    DATA_INPUT_PATH = DATA_INPUT_PATH or os.getenv('NHL_DATA_INPUT_PATH', f'{ROOT_DIR}/dataset/unprocessed/')
    DATA_INPUT_PATH = pathlib.Path(DATA_INPUT_PATH).absolute()
    DATA_OUTPUT_PATH = DATA_OUTPUT_PATH or os.getenv('NHL_DATA_OUTPUT_PATH', f'{ROOT_DIR}/dataset/processed/')
    DATA_OUTPUT_PATH = pathlib.Path(DATA_OUTPUT_PATH).absolute()
    #Check that those paths exist
    ##DATA_INPUT_PATH & DATA_OUTPUT_PATH could not exist, create then
    if not DATA_INPUT_PATH.exists():
            print(f'Could not find input directory {DATA_INPUT_PATH}')
            print('Creating it..')
            os.makedirs(DATA_INPUT_PATH)
    if not DATA_OUTPUT_PATH.exists():
            print(f'Could not find output directory {DATA_OUTPUT_PATH}')
            print('Creating it..')
            os.makedirs(DATA_OUTPUT_PATH)
    if return_root_dir:
        return DATA_INPUT_PATH, DATA_OUTPUT_PATH, ROOT_DIR
    else:
        return DATA_INPUT_PATH, DATA_OUTPUT_PATH

def _assert_game_json(pathlib_gen):
    """
    Asserts that `pathlib_gen` files respect NHL games JSON structures
    Ensure validity of future processing of data

    Parameters:
        pathlib_gen (pathlib generator object) : Generator issued from pathlib.Path.glob or pathlib.Path.rglob
    Returns:
        True if all files in pathlib_gen respect assertions (type dict, `id`&`plays` keys)
        None and early exit if any file does not respect assertions
    """
    faulty_files = []
    while True:
        try:
            file_item = next(pathlib_gen)
            with file_item.open(mode='r') as file:
                json_file = json.load(file)
                assert type(json_file) is dict
                assert json_file.get("id")
                assert json_file.get("plays")
            return True
        except StopIteration:
            if len(faulty_files) > 0:
                print(f'Found {len(faulty_files)} faulty files in path')
                print(f'Such as {random.choice(faulty_files)}')
                inpt = input('See full list ? [y/N]')
                if inpt == 'y' or inpt == 'Y':
                    print(list(map(str,faulty_files)))
                print('Make sure that DATA_INPUT_PATH contains only NHL game data')
                os.sys.exit()
                #return False, faulty_files
            else:
                return True
        except AssertionError:
            faulty_files.append(file_item)



def _check_for_env_vars():
    """
    Boilerplate code to ensure validity of environment variables NHL_DATA_INPUT_PATH and NHL_DATA_OUTPUT_PATH if provided
    """
    DATA_INPUT_PATH = os.getenv('NHL_DATA_INPUT_PATH', None)
    DATA_OUTPUT_PATH = os.getenv('NHL_DATA_OUTPUT_PATH', None)
    for path in [real_path for real_path in [DATA_INPUT_PATH, DATA_OUTPUT_PATH] if real_path is not None]:
        assert os.path.isdir(path), f'Path {path} is not an existing directory'


def _check_for_cli_args():
    """
    Boilerplate code to ensure validity of passed args to NHLDataFetcher

    Returns:
        None if no arguments,
        NHLDataFetcher helper if REGEX pattern `-?-?he?.?` is matched against arguments,
        Arguments as if, if they are valid
    """
    passed_args = os.sys.argv[1:] if len(os.sys.argv) > 1 else None
    if passed_args is None:
        return None
    #Match for `help`
    if any([ match('-?-?he?.?', arg) for arg in passed_args ]):
        subprocess_popen_cmanager(['python', 'data/main.py', '--help'], timeout=10)
        print('Exiting..')
        os.sys.exit()
    p = subprocess_popen_cmanager(['python', 'data/main.py', '--parse-args', *passed_args], timeout=30)
    args = p.args
    if '--parse-args' in p.args:
        args.remove('--parse-args')
    return args


def main():
    #Gather input (unprocessed data) and output (processed csv of Pandas DataFrame) paths
    DATA_INPUT_PATH, DATA_OUTPUT_PATH, ROOT_DIR = _gather_and_check_paths(return_root_dir=True)
    #Move to ROOT_DIR for correct execution of subsequent scripts
    _CURRENT_DIR = os.getcwd()
    os.chdir(ROOT_DIR)
    #Assert that files in DATA_INPUT_PATH that have the pattern 'game*.json' are:
    # - of valid format
    # - contain necessary keys (see _assert_game_json() docstring)
    if _assert_game_json(DATA_INPUT_PATH.rglob("**/*game*.json")):
        #Assert validity of CLI arguments `--year`, `--type`, `--games` and `--output`
        passed_args = _check_for_cli_args()
        print("Attempting to download games")
        #Call CLI data acquisition NHLDataFetcher with passed_args if exists
        #Else download every regular season and playoffs games from season 2016-17 to 2023-24
        subprocess_popen_cmanager(passed_args or ['python', 'data/main.py', '-y', '2016-2023', '-t', '2,3'], timeout=1800)
        print("Filtering json, formatting to pandas DataFrame and saving to csv")
        #Call data formatting tidy pipeline on JSON in DATA_INPUT_PATH and save them to csv of pandas DataFrames
        features.data_formatting.process_and_save_json_file(DATA_INPUT_PATH, DATA_OUTPUT_PATH)
        #Move back to previous dir
        os.chdir(_CURRENT_DIR)

if __name__ == '__main__':
    main()
