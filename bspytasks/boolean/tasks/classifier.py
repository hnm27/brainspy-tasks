import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from bspytasks.boolean.data import BooleanGateDataset

from bspyalgo.utils.io import save
from bspyalgo.utils.performance import perceptron
from bspyalgo.algorithms.gradient.fitter import train
from bspyalgo.manager import get_criterion, get_optimizer
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspyalgo.utils.performance import perceptron, corr_coeff_torch, plot_perceptron


def boolean_task(configs, gate, custom_model, threshold, transforms=None, logger=None, is_main=True):
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


if __name__ == "__main__":
    import numpy as np
    import datetime as d
    from torchvision import transforms

    from bspytasks.boolean.logger import Logger
    from bspyalgo.utils.io import load_configs
    from bspyalgo.utils.transforms import ToTensor, ToVoltageRange
    from bspyproc.processors.dnpu import DNPU

    configs = load_configs('configs/boolean.yaml')

    transforms = transforms.Compose([
        ToTensor()
    ])

    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    gate = [0, 0, 0, 1]

    boolean_task(configs, gate, DNPU, threshold=0.8, transforms=transforms, logger=logger)
