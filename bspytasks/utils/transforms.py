import numpy as np
import torch
from brainspy.utils.transforms import get_linear_transform_constants
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.pytorch import TorchUtils


class DataToVoltageRange:
    def __init__(self, v_min, v_max, x_min=-1, x_max=1):
        self.scale, self.offset = get_linear_transform_constants(
            np.array(v_min), np.array(v_max), np.array(x_min), np.array(x_max)
        )

    def __call__(self, data):
        inputs = data[0]
        inputs = (inputs * self.scale) + self.offset
        return (inputs, data[1])


class PointsToPlateaus:
    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, x):
        return self.mgr.points_to_plateaus(x)


class PlateausToPoints:
    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, x):
        return self.mgr.plateaus_to_points(x)


class DataPointsToPlateau:
    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, data):
        inputs, targets = data[0], data[1]

        inputs = self.mgr.points_to_plateaus(inputs)
        targets = self.mgr.points_to_plateaus(targets)

        return (inputs, targets)


class ToDevice:
    """Inputs and targets are transferred to GPU if necessary"""

    def __call__(self, data):
        inputs, targets = data[0], data[1]
        if inputs.device != TorchUtils.get_device():
            inputs = inputs.to(device=TorchUtils.get_device())
        if targets.device != TorchUtils.get_device():
            targets = targets.to(device=TorchUtils.get_device())
        return (inputs, targets)


class DataToTensor:
    """Convert labelled data to pytorch tensor."""

    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = TorchUtils.get_device()

    def __call__(self, data):
        inputs, targets = data[0], data[1]
        inputs = torch.tensor(
            inputs, device=self.device, dtype=torch.get_default_dtype()
        )
        targets = torch.tensor(
            targets, device=self.device, dtype=torch.get_default_dtype()
        )
        return (inputs, targets)
