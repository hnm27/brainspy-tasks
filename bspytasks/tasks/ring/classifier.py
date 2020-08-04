'''
This is a template for evolving the NN based on the boolean_logic experiment.
The only difference to the measurement scripts are on lines where the device is called.

'''
import torch
import numpy as np
import os
#from bspyalgo.algorithm_manager import get_algorithm
#from bspyproc.bspyproc import get_processor
from matplotlib import pyplot as plt
from bspyalgo.utils.performance import perceptron, corr_coeff_torch, plot_perceptron
from bspyalgo.utils.io import create_directory, create_directory_timestamp, save
from bspyproc.utils.pytorch import TorchUtils
import matplotlib

from bspytasks.tasks.ring.data import RingDataGenerator, RingDataLoader
from bspyalgo.algorithms.gradient.fitter import train, split
from bspyalgo.manager import get_criterion, get_optimizer
from bspyalgo.utils.io import save
from matplotlib import cm


def ring_task(dataloaders, custom_model, configs, transforms=None, logger=None, is_main=True):
    results = {}
    results['gap'] = str(dataloaders[0].dataset.gap)
    main_dir, results_dir, reproducibility_dir = init_dirs(str(dataloaders[0].dataset.gap), configs['results_base_dir'], is_main)
    criterion = get_criterion(configs['algorithm'])
    print('==========================================================================================')
    print("GAP: " + str(dataloaders[0].dataset.gap))
    model = custom_model(configs['processor'])
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), configs['algorithm'])

    model, performances = train(model, (dataloaders[0], dataloaders[1]), configs['algorithm']['epochs'], criterion, optimizer, logger=logger, save_dir=reproducibility_dir)

    if len(dataloaders[0]) > 0:
        results['train_results'] = postprocess(dataloaders[0].dataset[dataloaders[0].sampler.indices], model, criterion, logger, main_dir)
        results['train_results']['performance'] = performances[0]
    if len(dataloaders[1]) > 0:
        results['dev_results'] = postprocess(dataloaders[1].dataset[dataloaders[1].sampler.indices], model, criterion, logger, main_dir)
        results['dev_results']['performance'] = performances[1]
    if len(dataloaders[2]) > 0:
        results['test_results'] = postprocess(dataloaders[2].dataset[dataloaders[2].sampler.indices], model, criterion, logger, main_dir)
    plot_results(results, plots_dir=results_dir)
    torch.save(results, os.path.join(reproducibility_dir, 'results.pickle'))
    save('configs', os.path.join(reproducibility_dir, 'configs.yaml'), data=configs)
    print('==========================================================================================')
    return results


def get_ring_data(gap, configs, transforms, data_dir=None):
    # Returns dataloaders and split indices
    if configs['data']['load']:
        dataset = RingDataLoader(data_dir, transforms=transforms, save_dir=data_dir)
    else:
        dataset = RingDataGenerator(configs['data']['sample_no'], gap, transforms=transforms, save_dir=data_dir)
    dataloaders, split_indices = split(dataset, configs['algorithm']['batch_size'], num_workers=configs['algorithm']['worker_no'], split_percentages=configs['data']['split_percentages'])
    return dataset, dataloaders, split_indices


def postprocess(dataset, model, criterion, logger, save_dir=None):
    results = {}
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        indices = torch.argsort(targets[:, 0], dim=0)
        inputs, targets = inputs[indices], targets[indices]
        predictions = model(inputs)
        results['best_performance'] = criterion(predictions, targets)

    #results['gap'] = dataset.gap
    results['inputs'] = inputs
    results['targets'] = targets
    results['best_output'] = predictions
    results['accuracy'] = perceptron(predictions, targets)  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results['correlation'] = corr_coeff_torch(predictions.T, targets.T)

    # if (results['accuracy']['accuracy_value'] >= threshold):
    #     results['veredict'] = True
    # else:
    #     results['veredict'] = False
    # results['summary'] = 'VC Dimension: ' + str(len(dataset)) + ' Gate: ' + gate_name + ' Veredict: ' + str(results['veredict']) + ' ACCURACY: ' + str(results['accuracy']['accuracy_value'].item()) + '/' + str(threshold)
    # results['results_fig'] = plot_results(results, save_dir)
    # results['accuracy_fig'] = plot_perceptron(results['accuracy'], save_dir)
    # if logger is not None:
    #     logger.log.add_figure('Results/VCDim' + str(len(dataset)) + '/' + gate_name, results['results_fig'])
    #     logger.log.add_figure('Accuracy/VCDim' + str(len(dataset)) + '/' + gate_name, results['accuracy_fig'])
    return results


def init_dirs(gap, base_dir, is_main=False):
    base_dir = os.path.join(base_dir, 'gap_' + gap)
    main_dir = 'ring_classification'
    reproducibility_dir = 'reproducibility'
    results_dir = 'results'
    if is_main:
        base_dir = create_directory_timestamp(base_dir, main_dir)
    reproducibility_dir = os.path.join(base_dir, reproducibility_dir)
    create_directory(reproducibility_dir)
    results_dir = os.path.join(base_dir, results_dir)
    create_directory(results_dir)
    return main_dir, results_dir, reproducibility_dir


def plot_results(results, plots_dir=None, show_plots=False, extension='png'):
    plot_output(results['train_results'], 'Train', plots_dir=plots_dir, extension=extension)
    if 'dev_results' in results:
        plot_output(results['dev_results'], 'Dev', plots_dir=plots_dir, extension=extension)
    if 'test_results' in results:
        plot_output(results['test_results'], 'Test', plots_dir=plots_dir, extension=extension)
    plt.figure()
    plt.title(f'Learning profile', fontsize=12)
    plt.plot(results['train_results']['performance'], label='Train')
    if 'dev_results' in results:
        plt.plot(results['dev_results']['performance'], label='Dev')
    plt.legend()
    if plots_dir is not None:
        plt.savefig(os.path.join(plots_dir, f"training_profile." + extension))

    plt.figure()
    plt.title(f"Inputs (V) \n {results['gap']} gap (-1 to 1 scale)", fontsize=12)
    plot_inputs(results['train_results'], 'Train', ['blue', 'cornflowerblue'])
    if 'dev_results' in results:
        plot_inputs(results['dev_results'], 'Dev', ['orange', 'bisque'])
    if 'test_results' in results:
        plot_inputs(results['test_results'], 'Test', ['green', 'springgreen'])
    plt.legend()
    # if type(results['dev_inputs']) is torch.Tensor:
    if plots_dir is not None:
        plt.savefig(os.path.join(plots_dir, f"input." + extension))

    if show_plots:
        plt.show()
    plt.close('all')


def plot_output(results, label, plots_dir=None, extension='png'):
    plt.figure()
    plt.plot(results['best_output'].detach().cpu())
    plt.title(f"{label} Output (nA) \n Performance: {results['best_performance']} \n Accuracy: {results['accuracy']['accuracy_value']}", fontsize=12)
    if plots_dir is not None:
        plt.savefig(os.path.join(plots_dir, label + "_output." + extension))


def plot_inputs(results, label, colors=['b', 'r'], plots_dir=None, extension='png'):
    # if type(results['dev_inputs']) is torch.Tensor:
    inputs = results['inputs'].cpu().numpy()
    targets = results['targets'][:, 0].cpu().numpy()
    # else:
    #     inputs = results['dev_inputs']
    #     targets = results['dev_targets']
    plt.scatter(inputs[targets == 0][:, 0], inputs[targets == 0][:, 1], c=colors[0], label='Class 0 (' + label + ')', cmap=colors)
    plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1], c=colors[1], label='Class 1 (' + label + ')', cmap=colors)


def validate_on_hardware(model):
    pass


if __name__ == '__main__':
    from bspyalgo.algorithms.gradient.gd import GD
    from bspyalgo.algorithms.gradient.core.losses import fisher
    from bspyalgo.algorithms.gradient.core.logger import Logger
    from bspyproc.processors.dnpu import DNPU
    import numpy as np
    import datetime as d
    from bspytasks.utils.transforms import ToTensor, ToVoltageRange
    from torchvision import transforms
    from bspyalgo.utils.io import load_configs

    configs = load_configs('configs/ring.yaml')
    V_MIN = [-1.2, -1.2]
    V_MAX = [0.7, 0.7]
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    transforms = transforms.Compose([
        ToVoltageRange(V_MIN, V_MAX, -1, 1),
        ToTensor()
    ])
    gap = 0.4
    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))
    dataset, dataloaders, split_indices = get_ring_data(gap, configs, transforms)
    ring_task(dataloaders, DNPU, configs, logger=logger)
