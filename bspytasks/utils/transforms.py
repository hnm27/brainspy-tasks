import numpy as np
from bspyproc.utils.pytorch import TorchUtils


class ToTensor(object):
    """Convert labelled data to pytorch tensor."""

    def __call__(self, sample):
        inputs, targets = sample[0], sample[1]
        inputs = TorchUtils.get_tensor_from_numpy(inputs)
        targets = TorchUtils.get_tensor_from_numpy(targets)
        return (inputs, targets)


class ToVoltageRange(object):
    def __init__(self, v_min, v_max):
        self.v_min = min
        self.v_max = max

    def __call__(self, inputs):
        scale, offset = self.get_map_to_voltage_vars(self.v_min, self.v_max, inputs)
        return self.map_to_voltage(inputs, v_min, v_max)

    def map_to_voltage(x, v_min, v_max):
        a = ((v_min - v_max) / (x.min() - x.max()))
        b = v_max - a * x.max()
        return (a * x) + b

    def get_map_to_voltage_vars(v_min, v_max, x=np.array([-1, 1])):
        scale = ((v_min - v_max) / (x.min() - x.max()))
        offset = v_max - scale * x.max()
        return scale, offset
