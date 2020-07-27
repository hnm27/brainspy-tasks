from bspytasks.tasks.boolean.classifier import find_gate
from bspytasks.datasets.boolean import generate_targets
import torch
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.algorithms.gradient.gd import GD
from bspyalgo.algorithms.gradient.core.losses import corrsig
from bspyalgo.algorithms.gradient.core.logger import Logger
from bspyproc.processors.dnpu import DNPU
import numpy as np
import datetime as d
from torchvision import transforms
from bspytasks.utils.transforms import ToTensor, ToVoltageRange


def vc_dimension(current_dimension, custom_model, configs, criterion, custom_optimizer, threshold, transforms, logger):
    targets = generate_targets(current_dimension)
    results = []
    for target in targets:
        logger.gate = str(target)
        model = TorchUtils.format_tensor(custom_model(configs))
        optimizer = custom_optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0065)
        _, model, accuracy, veredict = find_gate(np.array(target), model, corrsig, optimizer, threshold, transforms=transforms, logger=logger)
        aux = {}
        aux['target'] = target
        aux['model'] = model
        aux['accuracy'] = accuracy
        aux['found'] = veredict
        results.append(aux)
    return results


def validate_vc_dimension_on_hardware(current_dimension):
    targets = generate_targets(current_dimension)
    results = []
    for target in targets:
        #_, model, accuracy, veredict = validate_gate_on_hardware(nmodel, corrsig, optimizer, threshold, transforms=transforms, logger=logger)
        aux = {}
        # aux['target'] = target
        # aux['model'] = model
        # aux['accuracy'] = accuracy
        # aux['found'] = veredict
        results.append(aux)
    return results


if __name__ == "__main__":

    configs = {}
    configs['platform'] = 'simulation'
    configs['torch_model_dict'] = '/home/unai/Documents/3-programming/brainspy-processors/tmp/input/models/test.pt'
    configs['input_indices'] = [0, 1]
    configs['input_electrode_no'] = 7
    # configs['waveform'] = {}
    # configs['waveform']['amplitude_lengths'] = 80
    # configs['waveform']['slope_lengths'] = 20
    V_MIN = [-1.2, -1.2]
    V_MAX = [0.7, 0.7]
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    transforms = transforms.Compose([
        ToTensor()
    ])

    #model = TorchUtils.format_tensor(DNPU(configs))

    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    results = vc_dimension(4, DNPU, configs, corrsig, torch.optim.Adam, 80.0, transforms=transforms, logger=logger)
