
import numpy as np

from torch.utils.data import Dataset

# ZERO = -0.5
# ONE = 0.5
# QUARTER =  (abs(ZERO) + abs(ONE)) / 4
# TODO: Include this is the configuration file
X = [-0.7, -0.7, 0.5, 0.5, -0.35, 0.25, 0.0, 0.0]
Y = [-0.7, 0.5, -0.7, 0.5, 0.0, 0.0, -0.35, 0.25]


class BooleanGateDataset(Dataset):

    def __init__(self, vc_dimension, target, transforms=None, verbose=True):
        self.transforms = transforms
        self.inputs = self.generate_inputs(vc_dimension)
        self.targets = target.T[:, np.newaxis]

    def __getitem__(self, index):
        #index = np.random.permutation(self.inputs.shape[0])
        inputs = self.inputs[index, :]
        targets = self.targets[index, :]

        sample = (inputs, targets)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.targets)

    def generate_inputs(self, vc_dimension):
        assert len(X) == len(Y), f"Number of data in both dimensions must be equal ({len(X)},{len(Y)})"
        assert vc_dimension <= len(X), 'VC Dimension exceeds the current number of points'
        return np.array([X[:vc_dimension], Y[:vc_dimension]]).T


def generate_targets(self, vc_dimension, verbose=True):
        # length of list, i.e. number of binary targets
    binary_target_no = 2**vc_dimension
    assignments = []
    list_buf = []

    # construct assignments per element i
    if verbose:
        print('===' * vc_dimension)
        print('ALL BINARY LABELS:')
    level = int((binary_target_no / 2))
    while level >= 1:
        list_buf = []
        buf0 = [0] * level
        buf1 = [1] * level
        while len(list_buf) < binary_target_no:
            list_buf += (buf0 + buf1)
        assignments.append(list_buf)
        level = int(level / 2)

    binary_targets = np.array(assignments).T
    if verbose:
        print(binary_targets)
        print('===' * vc_dimension)
    return binary_targets[1:-1]  # Remove [0,0,0,0] and [1,1,1,1] gates
