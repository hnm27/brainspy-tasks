# from bspyalgo.utils.io import load_configs
# from bspyalgo.utils.io import save
# from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager

# results_path = find_single_gate('configs/benchmark_tests/capacity/template_ga_simulation.json', '[1 0 0 1]')
# validate_single_gate('configs/benchmark_tests/capacity/template_ga_validation.json', results_path)
from bspyalgo.algorithms.gradient.gd import GD
from bspytasks.utils.transforms import ToTensor, ToVoltageRange

from bspyproc.utils.pytorch import TorchUtils
from bspytasks.datasets.boolean import BooleanGateDataset
from bspytasks.datasets.ring import RingDataGenerator

from torchvision import transforms
import torch
from bspyalgo.utils.performance import perceptron
from bspyalgo.algorithms.gradient.fitter import train, test
import numpy as np

import matplotlib.pyplot as plt
import torch

from bspyalgo.utils.performance import accuracy


def train_classifier(model, criterion, optimizer, target=np.array([0, 1, 1, 0]), threshold=0.8, transforms=None, logger=None):
    veredict = False
    dataset = BooleanGateDataset(target=target, transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=False)
    model, _, _ = train(model, (loader, None), 100, criterion, optimizer, logger=logger)
    return dataset, model


def postprocess(gate_name, dataset, model, logger, threshold):

    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)

    acc = accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)

    if (acc[0] >= threshold):
        veredict = True
        print('VEREDICT: PASS')
    else:
        veredict = False
        print('VEREDICT: FAIL')
    plot_gate(gate_name, veredict, predictions, targets, logger, show_plots=True)
    return veredict, acc


def plot_gate(gate_name, veredict, output, target, logger, show_plots=False):
    fig = plt.figure()
    plt.title(gate_name + ' Veredict:' + str(veredict))
    plt.plot(output.detach().cpu(), label='Prediction')
    plt.plot(target.detach().cpu(), label='Target')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')
    plt.legend()
    # if save_dir is not None:
    #     plt.savefig(save_dir)
    if logger is not None:
        logger.log.add_figure('Results/' + gate_name, fig)
    if show_plots:
        plt.show()
    plt.close()


def find_gate(gate, model, criterion, optimizer, threshold, transforms=None, logger=None):

    print('==========================================================================================')
    print(f"GATE: {gate} ")
    dataset, model = train_classifier(model, criterion, optimizer, target=gate, transforms=transforms, logger=logger)
    veredict, accuracy = postprocess(str(gate), dataset, model, logger, threshold)
    print('ACCURACY: ' + str(accuracy[0]) + '/' + str(threshold))
    print('==========================================================================================')
    return dataset, model, accuracy[0], veredict


def validate_on_hardware(model):
    pass


if __name__ == "__main__":
    from bspyalgo.algorithms.gradient.gd import GD
    from bspyalgo.algorithms.gradient.core.losses import corrsig
    from bspyalgo.algorithms.gradient.core.logger import Logger
    from bspyproc.processors.dnpu import DNPU
    import numpy as np
    import datetime as d

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

    model = TorchUtils.format_tensor(DNPU(configs))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0065)
    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    gate = np.array([0, 0, 0, 1])

    find_gate(gate, model, corrsig, optimizer, 80.0, transforms=transforms, logger=logger)
