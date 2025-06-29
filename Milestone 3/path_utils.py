#!/usr/bin/env python3

import pathlib
from data import format_string
from typing import Union


StringColor = format_string.StringColor()


def get_current_file_path(return_dir_path=False):
    """
    Get run file path from builtin `__file__` and returns it.
    Additionally returns file's directory path if param `return_dir_path` is set to True.
    If `__file__` is not defined (ex: interaction python shell), resorts to returning directory path from `pwd`

    Parameters:
      return_dir_path (bool, default=False): Additionally returns directory path, if file path exists

    Returns:
      current_file_path, [current_dir_path] (pathlib.Path): pathlib.Path objects of file and directory paths
    """
    try:
        current_file_path = pathlib.Path(__file__)
        if return_dir_path and current_file_path.is_file():
            current_dir_path = current_file_path.parent
            return current_file_path, current_dir_path
        return current_file_path
    except NameError:
        StringColor.warning('__file__ not defined, returning directory path from `pwd` only')
        current_dir_path = pwd
        current_dir_path = pathlib.Path(current_dir_path)
        return current_dir_path


def _is_git_folder(path: pathlib.Path):
    # while traversing entire directory path
    dirs_to_traverse = list(path.parents)
    while len(dirs_to_traverse) > 0:
        to_check = dirs_to_traverse.pop(0)
        if (to_check / '.git').is_dir():
            return to_check
    return None


def get_git_root_path(current_file_path: Union[pathlib.Path, str]):
    # Try to convert current_file_path to pathlib.Path
    if not type(current_file_path) == pathlib.Path:
        try:
            current_file_path = pathlib.Path(current_file_path).absolute().resolve()
        except TypeError as e:
            StringColor.error('Cannot convert current_file_path to pathlib.Path')
            print(f'current_file_path ({current_file_path}) seems to not be of type str')
            print(type(current_file_path))
            raise e
    # Search for .git folder and return parent folder if found
    res = _is_git_folder(current_file_path)
    if res is None:
        StringColor.error('Did not find root git directory')
        return None
    return res


def _get_base_root_path(current_file_path: Union[pathlib.Path, str]):
    # Try to convert current_file_path to pathlib.Path
    if not type(current_file_path) == pathlib.Path:
        try:
            current_path = pathlib.Path(current_file_path)
        except TypeError as e:
            StringColor.error('Cannot convert current_file_path to pathlib.Path')
            print(f'current_file_path ({current_file_path}) seems to not be of type str')
            print(type(current_file_path))
            raise e
    # Return if cannot find 'ift6758' (assumed to be root directory) in path hierarchy
    if 'ift6758' not in current_path.parts:
        StringColor.warning('Cannot find parent dir `ift6758`')
        return

    # Return truncated path to leftmost occurence of `ift6758`
    tmp_path_parts = current_path.parts
    base_root_index = 0
    for i in range(0, current_path.parts.count('ift6758')):
        base_root_index += tmp_path_parts.index('ift6758')
        base_root_path = pathlib.Path('/'.join(current_path.parts[:base_root_index+1]))
        #if (base_root_path / 'dataset').is_dir():
        #    break
        tmp_path_parts = tmp_path_parts[base_root_index+1:]
    return base_root_path
