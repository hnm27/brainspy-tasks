
# from bspyalgo.utils.io import save
# from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager

# results_path = find_single_gate('configs/benchmark_tests/capacity/template_ga_simulation.json', '[1 0 0 1]')
# validate_single_gate('configs/benchmark_tests/capacity/template_ga_validation.json', results_path)
from bspyalgo.algorithms.gradient.gd import GD


from bspyproc.utils.pytorch import TorchUtils
from bspytasks.tasks.boolean.data import BooleanGateDataset
from bspyalgo.manager import get_criterion, get_optimizer
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

from bspyalgo.utils.io import save


def train_classifier(model, criterion, optimizer, epochs, target=np.array([0, 1, 1, 0]), threshold=0.8, transforms=None, logger=None, save_dir='tmp/output/boolean'):
    dataset = BooleanGateDataset(target=target, transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=False)
    model, performance = train(model, (loader, None), epochs, criterion, optimizer, logger=logger, save_dir=save_dir)
    return dataset, model, performance


def postprocess(gate_name, results, model, threshold, logger=None, node=None, save_dir=None):
    results['accuracy'] = perceptron(results['predictions'], results['targets'], node)  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results['correlation'] = corr_coeff_torch(results['predictions'].T, results['targets'].T)

    if (results['accuracy']['accuracy_value'] >= threshold):
        results['veredict'] = True
    else:
        results['veredict'] = False
    results['threshold'] = threshold
    results['gate'] = gate_name
    results['summary'] = 'VC Dimension: ' + str(len(results['targets'])) + ' Gate: ' + gate_name + ' Veredict: ' + str(results['veredict']) + '\n Accuracy (Simulation): ' + str(results['accuracy']['accuracy_value'].item()) + '/' + str(threshold)
    results['results_fig'] = plot_results(results, save_dir)
    results['accuracy_fig'] = plot_perceptron(results['accuracy'], save_dir)
    if logger is not None:
        logger.log.add_figure('Results/VCDim' + str(len(results['targets'])) + '/' + gate_name, results['results_fig'])
        logger.log.add_figure('Accuracy/VCDim' + str(len(results['targets'])) + '/' + gate_name, results['accuracy_fig'])
    return results


def evaluate_model(model, dataset):
    results = {}
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)
    results['inputs'] = inputs
    results['targets'] = targets
    results['predictions'] = predictions
    return results


def validate_model(model, results):
    with torch.no_grad():
        model.eval()
        predictions = model(results['inputs'])
    results['predictions'] = predictions
    return results


def plot_results(results, save_dir=None, fig=None, show_plots=False):
    if fig is None:
        fig = plt.figure()
    plt.title(results['summary'])
    plt.plot(results['predictions'].detach().cpu(), label='Prediction (Simulation)')
    plt.plot(results['targets'].detach().cpu(), label='Target (Simulation)')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'results.jpg'))
    if show_plots:
        plt.show()
    plt.close()
    return fig


def find_gate(configs, gate, custom_model, threshold, transforms=None, logger=None, is_main=True):

    main_dir, reproducibility_dir = init_dirs(str(gate), configs['results_base_dir'], is_main)
    gate = np.array(gate)
    criterion = get_criterion(configs['algorithm'])
    dataset = BooleanGateDataset(target=gate, transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=configs['algorithm']['batch_size'], shuffle=True, pin_memory=False)
    print('==========================================================================================')
    print("GATE: " + str(gate))
    for i in range(configs['max_attempts'] + 1):
        print('ATTEMPT: ' + str(i))
        model = custom_model(configs['processor'])
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), configs['algorithm'])
        model, performance = train(model, (loader, None), configs['algorithm']['epochs'], criterion, optimizer, logger=logger, save_dir=reproducibility_dir)
        results = evaluate_model(model, dataset)
        results = postprocess(str(gate), results, model, threshold, logger=logger, save_dir=main_dir)
        results['performance_history'] = performance[0]  # Dev/Test performance is not relevant to the boolean gates task
        print(results['summary'])
        if results['veredict']:
            break
    torch.save(results, os.path.join(reproducibility_dir, 'results.pickle'))
    save('configs', os.path.join(reproducibility_dir, 'configs.yaml'), data=configs)
    print('==========================================================================================')
    return results


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
    from bspyalgo.utils.io import load_configs
    from bspyalgo.algorithms.gradient.gd import GD
    from bspyalgo.algorithms.gradient.core.losses import corrsig
    from bspyalgo.algorithms.gradient.core.logger import Logger
    from bspyproc.processors.dnpu import DNPU
    from bspytasks.utils.transforms import ToTensor, ToVoltageRange
    import numpy as np
    import datetime as d

    configs = load_configs('configs/boolean.yaml')

    # configs = {}
    # configs['platform'] = 'simulation'
    # configs['torch_model_dict'] = '/home/unai/Documents/3-programming/brainspy-processors/tmp/input/models/test.pt'
    # configs['input_indices'] = [0, 1]
    # configs['input_electrode_no'] = 7
    # configs['waveform'] = {}
    # configs['waveform']['amplitude_lengths'] = 80
    # configs['waveform']['slope_lengths'] = 20

    transforms = transforms.Compose([
        ToTensor()
    ])

    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    gate = [0, 0, 0, 1]

    find_gate(configs, gate, DNPU, threshold=0.8, transforms=transforms, logger=logger)
