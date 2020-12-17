from os import listdir
from pathlib import Path
import numpy as np
from shutil import copyfile
import pandas as pd

from src.scripts.config import data_path, results_path


def divide_data():
    # from a folder with several files, splits in several folders with less files
    n_folders = 10
    model = 'LorenzAttractor'

    files = sorted(listdir(Path.joinpath(data_path, model)))
    files_data = [f for f in files if 'solution' in f]
    params = [f for f in files if 'solution' not in f][0]
    n_files = len(files_data)
    by_folder = int(np.ceil(n_files / n_folders))

    for i in range(n_folders):
        new_folder = Path.joinpath(data_path, f'{model}_{i}')
        new_folder.mkdir(parents=True, exist_ok=True)
        copyfile(Path.joinpath(data_path, model, params), Path.joinpath(new_folder, params))
        for f in files_data[i * by_folder:(i * by_folder) + by_folder]:
            copyfile(Path.joinpath(data_path, model, f), Path.joinpath(new_folder, f))


def concat_times():
    n_folders = 10
    model = 'LorenzAttractor'

    times = []
    for i in range(n_folders):
        new_folder = Path.joinpath(results_path, f'{model}_{i}')
        files = listdir(new_folder)
        files_data = [f for f in files if 'solution' in f]
        files_time = [f for f in files if 'time' in f]
        for file_time in files_time:
            tt = pd.read_csv(Path.joinpath(new_folder, file_time), index_col=0).transpose()
            tt.columns = ['time']
            times.append(tt)
        for f in files_data:
            copyfile(Path.joinpath(new_folder, f), Path.joinpath(results_path, model, f))
    all_times = pd.concat(times)
    all_times['model'] = all_times.index
    all_times.to_csv(Path.joinpath(results_path, model, 'times.csv'))


if __name__== '__main__':
    # divide_data()
    concat_times()
    #TODO extract important variables and document the usage
