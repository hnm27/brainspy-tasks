import unittest
import HtmlTestRunner
import torch
import numpy as np
from numpy import load, asarray
from bspytasks.utils.io import save, save_pickle
import tracemalloc
import yaml
import shutil
import os
import pickle


class IOTest(unittest.TestCase):
    """
    Tests for the io.py class.
    """

    def __init__(self, test_name):
        super(IOTest, self).__init__()
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        configs["batch"] = None
        configs["data"] = {"data1": "New data"}
        self.configs = configs
        while "brainspy-tasks" not in os.getcwd():
            os.chdir("..")
            os.chdir("brainspy-tasks")
        self.path = os.path.join(os.getcwd(), "test/unit/utils/testfiles")

    def test_save(self):
        """
        Test for save() method with parameter - "configs"
        """
        path = self.path + "/testutil.yaml"
        save("configs", path, data=self.configs)
        with open(path) as f:
            x = yaml.load(f, yaml.FullLoader)
        self.assertEqual(x["data"], self.configs["data"])
        self.assertEqual(x["batch"], None)
        with self.assertRaises(
            KeyError
        ):  # Testing for data that does not exist in the file
            x["non_existant"]

    def test_savepickle(self):
        """
        Test for save_pickle() method which saves a python dictionary to yaml file
        """
        path = self.path + "/testutil.pickle"
        tracemalloc.start()
        save_pickle(self.configs, path)
        file = open(path, "rb")
        x = pickle.load(file)
        self.assertEqual(x["data"], self.configs["data"])
        with self.assertRaises(
            KeyError
        ):  # Testing for data that does not exist in the file
            x["non_existant"]
        file.close()

    def test_savenumpy(self):
        """
        Test for the save() method with parameter "numpy"
        Saving 2 numpy arrays to a .npz file
        """
        path = self.path + "/testutil.npz"
        x = np.arange(10)
        y = np.arange(11, 20)
        save("numpy", path, x=x, y=y)
        with np.load(path) as data:
            x2 = data["x"]
            y2 = data["y"]
            self.assertEqual(x[0], x2[0])
            self.assertEqual(y[7], y2[7])

    def runTest(self):
        self.test_save()
        self.test_savepickle()
        self.test_savenumpy()


if __name__ == "__main__":

    unittest.main()
