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

from bspyalgo.utils.io import save

from bspyalgo.utils.performance import perceptron
from bspyalgo.algorithms.gradient.fitter import train
import numpy as np

import matplotlib.pyplot as plt
import torch

from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspyalgo.utils.performance import perceptron, corr_coeff_torch, plot_perceptron

import os


def train_classifier(model, criterion, optimizer, epochs, target=np.array([0, 1, 1, 0]), threshold=0.8, transforms=None, logger=None, save_dir='tmp/output/fitter'):
    veredict = False
    dataset = BooleanGateDataset(target=target, transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=False)
    model, train_performance, dev_performance = train(model, (loader, None), epochs, criterion, optimizer, logger=logger, save_dir=save_dir)
    return dataset, model, train_performance, dev_performance


def postprocess(gate_name, dataset, model, logger, threshold, save_dir=None):
    results = {}
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)

    results['inputs'] = inputs
    results['targets'] = targets
    results['predictions'] = predictions
    results['accuracy'] = perceptron(predictions, targets)  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results['correlation'] = corr_coeff_torch(predictions.T, targets.T)

    if (results['accuracy']['accuracy_value'] >= threshold):
        results['veredict'] = True
    else:
        results['veredict'] = False
    results['summary'] = 'VC Dimension: ' + str(len(dataset)) + ' Gate: ' + gate_name + ' Veredict: ' + str(results['veredict']) + ' ACCURACY: ' + str(results['accuracy']['accuracy_value'].item()) + '/' + str(threshold)
    results['results_fig'] = plot_results(results, save_dir)
    results['accuracy_fig'] = plot_perceptron(results['accuracy'], save_dir)
    if logger is not None:
        logger.log.add_figure('Results/VCDim' + str(len(dataset)) + '/' + gate_name, results['results_fig'])
        logger.log.add_figure('Accuracy/VCDim' + str(len(dataset)) + '/' + gate_name, results['accuracy_fig'])
    return results


def plot_results(results, save_dir=None, show_plots=False):
    fig = plt.figure()
    plt.title(results['summary'])
    plt.plot(results['predictions'].detach().cpu(), label='Prediction')
    plt.plot(results['targets'].detach().cpu(), label='Target')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'results.jpg'))
    if show_plots:
        plt.show()
    plt.close()
    return fig


def find_gate(gate, model, criterion, optimizer, epochs, threshold, transforms=None, logger=None, base_dir='tmp/output/boolean/gates', is_main=True):
    main_dir, reproducibility_dir = init_dirs(str(gate), base_dir, is_main)
    print('==========================================================================================')
    print("GATE: " + str(gate))
    dataset, model, performance, _ = train_classifier(model, criterion, optimizer, epochs, target=gate, transforms=transforms, logger=logger, save_dir=reproducibility_dir)
    results = postprocess(str(gate), dataset, model, logger, threshold, main_dir)
    results['performance'] = TorchUtils.format_tensor(torch.tensor(performance))
    print(results['summary'])
    if is_main:
        torch.save(results, os.path.join(reproducibility_dir, 'results.pickle'))
    print('==========================================================================================')
    return results


def validate_on_hardware(model):
    pass


def init_dirs(gate_name, base_dir, is_main):
    if is_main:
        base_dir = create_directory_timestamp(base_dir, gate_name)
        reproducibility_dir = os.path.join(base_dir, 'reproducibility')
        create_directory(reproducibility_dir)
    else:
        base_dir = os.path.join(base_dir, gate_name)
        reproducibility_dir = os.path.join(base_dir, 'reproducibility')
        create_directory(reproducibility_dir)
    return base_dir, reproducibility_dir


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

    find_gate(gate, model, corrsig, optimizer, epochs=80, threshold=87.5, transforms=transforms, logger=logger)
