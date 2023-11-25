import numpy as np


def load_npz(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data}
