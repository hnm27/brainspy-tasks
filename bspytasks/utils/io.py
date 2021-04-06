"""
Library that handles saving data
"""

import io
import os
import time
import pickle
import shutil
import torch
import yaml
import numpy as np
from brainspy.utils.io import save_configs


def save(
    mode: str, file_path: str, **kwargs: dict
):
    """
    This function formats data from a dictionary and saves it to the given file

    Parameters
    ----------

    mode : str
        file type as a python string  

    file_path : str
        file object or path to file

    kwargs : dict
        data that needs to be saved

    Example 
    --------
    configs = {"data" : "example"}
    path = "tests/unit/utils/testfiles/test.yaml"
    save("configs", path, data=configs)

    """
    if mode == "numpy":
        np.savez(file_path, **kwargs)
    elif not kwargs["data"]:
        raise ValueError(f"Value dictionary is missing in kwargs.")
    else:
        if mode == "configs":
            save_configs(kwargs["data"], file_path)
        elif mode == "pickle":
            save_pickle(kwargs["data"], file_path)
        else:
            raise NotImplementedError(
                f"Mode {mode} is not recognised. Please choose a value between 'numpy', 'torch', 'pickle' and 'configs'."
            )


def save_pickle(pickle_data: dict, file_path: str):
    """
    This function serializes data and saves it to the given file path
    The process to converts any kind of python objects (list, dict, etc.) into byte streams (0s and 1s) is called pickling or serialization.

    Parameters
    ---------

    pickle_data : dict
        list of data that needs to be saved

    file_path : str
        file object or path to file

    Example
    --------

    configs = {"data" : "example"}
    path = "tests/unit/utils/testfiles/test.pickle"
    save(configs,path)

    """
    with open(file_path, "wb") as f:
        pickle.dump(pickle_data, f)
        f.close()
