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
    def __init__(self, v_min, v_max, x_min=-1, x_max=1):
        self.scale, self.offset = self.get_map_to_voltage_vars(np.array(v_min), np.array(v_max), np.array(x_min), np.array(x_max))

    def __call__(self, inputs):
        return (inputs * self.scale) + self.offset

    # def map_to_voltage(self, v_min, v_max, x=np.array([-1, 1])):
    #     a = ((v_min - v_max) / (x.min() - x.max()))
    #     b = v_max - a * x.max()
    #     return (a * x) + b

    def get_map_to_voltage_vars(self, v_min, v_max, x_min, x_max):
        scale = ((v_min - v_max) / (x_min - x_max))
        offset = v_max - scale * x_max
        return scale, offset
