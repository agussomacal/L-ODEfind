import os
from pathlib import Path


def project_root():
    return Path(__file__).parent.parent.parent


data_path = Path.joinpath(project_root(), 'data')
results_path = Path.joinpath(project_root(), 'results')
plots_dir = str(results_path)

data_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)



def get_filename(filename, experiment, subfolders=[]):
    directory = plots_dir + experiment + '/'

    # check create directories
    for subfolder in subfolders:
        directory = directory + subfolder + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)

    return directory + filename

