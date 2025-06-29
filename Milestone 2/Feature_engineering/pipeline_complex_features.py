#!/usr/bin/env python3

import os
import pathlib
from complex_feature_formatting import augment_dataset
from typing import Union

os.sys.path.append(str(pathlib.Path(__file__).absolute().resolve().parents[1]))
from path_utils import get_current_file_path, get_git_root_path


def main():
    current_path = get_current_file_path()
    base_root_path = get_git_root_path(current_path)
    if base_root_path is None:
        raise Exception('Could not locate root git directory for processing dataset')
    data_input_path = (base_root_path / 'ift6758' / 'dataset' / 'unprocessed')
    data_output_path = (base_root_path / 'ift6758' / 'dataset' / 'complex_engineered')

    augment_dataset(data_input_path=data_input_path, data_output_path=data_output_path, years=None)


if __name__ == '__main__':
    main()
