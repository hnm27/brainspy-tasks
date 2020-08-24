from brainspy.utils.pytorch import TorchUtils
import os
import torch
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

from bspytasks.ring.data import RingDatasetGenerator, RingDatasetLoader, BalancedSubsetRandomSampler, balanced_permutation, split
from brainspy.utils.io import create_directory, create_directory_timestamp, save
from brainspy.utils.manager import get_criterion, get_optimizer, get_algorithm

from brainspy.algorithms.modules.performance.accuracy import get_accuracy, plot_perceptron
from brainspy.algorithms.modules.signal import corr_coeff


def ring_task(configs, dataloaders, custom_model, criterion, algorithm, waveform_transforms=None, logger=None, is_main=True):
    results = {}
    results['gap'] = str(configs['data']['gap'])
    print('==========================================================================================')
    print("GAP: " + str(results['gap']))

    results_dir, reproducibility_dir = init_dirs(str(results['gap']), configs['results_base_dir'], is_main)
    # criterion = get_criterion(configs['algorithm'])
    model = custom_model(configs['processor'])
    optimizer = get_optimizer(model, configs['algorithm'])
    # algorithm = get_algorithm(configs['algorithm'])

    model, train_data = algorithm(model, (dataloaders[0], dataloaders[1]), criterion, optimizer, configs['algorithm'], logger=logger, save_dir=reproducibility_dir, waveform_transforms=waveform_transforms)

    if len(dataloaders[0]) > 0:
        results['train_results'] = postprocess(configs['algorithm']['accuracy'], dataloaders[0].dataset[dataloaders[0].sampler.indices], model, criterion, logger, results_dir, name='train')
        results['train_results']['performance_history'] = train_data['performance_history'][0]
    if len(dataloaders[1]) > 0:
        results['dev_results'] = postprocess(configs['algorithm']['accuracy'], dataloaders[1].dataset[dataloaders[1].sampler.indices], model, criterion, logger, results_dir, name='dev')
        results['dev_results']['performance_history'] = train_data['performance_history'][1]
    if len(dataloaders[2]) > 0:
        results['test_results'] = postprocess(configs['algorithm']['accuracy'], dataloaders[2].dataset[dataloaders[2].sampler.indices], model, criterion, logger, results_dir, name='test')

    plot_results(results, plots_dir=results_dir)
    torch.save(model, os.path.join(reproducibility_dir, 'model.pt'))
    torch.save(results, os.path.join(reproducibility_dir, 'results.pickle'), pickle_protocol=p.HIGHEST_PROTOCOL)
    save('configs', os.path.join(reproducibility_dir, 'configs.yaml'), data=configs)
    print('==========================================================================================')
    return results


def get_ring_data(configs, transforms, data_dir=None):
    # Returns dataloaders and split indices
    if configs['data']['load']:
        dataset = RingDatasetLoader(data_dir, transforms=transforms, save_dir=data_dir)
    else:
        dataset = RingDatasetGenerator(configs['data']['sample_no'], configs['data']['gap'], transforms=transforms, save_dir=data_dir)
    dataloaders = split(dataset, configs['algorithm']['batch_size'], sampler=BalancedSubsetRandomSampler, num_workers=configs['algorithm']['worker_no'], split_percentages=configs['data']['split_percentages'])
    return dataloaders


def postprocess(configs, dataset, model, criterion, logger, save_dir=None, name='train'):
    results = {}
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        indices = torch.argsort(targets[:, 0], dim=0)
        inputs, targets = inputs[indices], targets[indices]
        predictions = model(inputs)
        results['performance'] = criterion(predictions, targets)

    # results['gap'] = dataset.gap
    results['inputs'] = inputs
    results['targets'] = targets
    results['best_output'] = predictions
    results['accuracy'] = get_accuracy(predictions, targets, configs)  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results['correlation'] = corr_coeff(predictions.T, targets.T)
    results['accuracy_fig'] = plot_perceptron(results['accuracy'], save_dir)

    return results


def init_dirs(gap, base_dir, is_main=False):
    main_dir = 'ring_classification_gap_' + gap
    reproducibility_dir = 'reproducibility'
    results_dir = 'results'
    if is_main:
        base_dir = create_directory_timestamp(base_dir, main_dir)
    reproducibility_dir = os.path.join(base_dir, reproducibility_dir)
    create_directory(reproducibility_dir)
    results_dir = os.path.join(base_dir, results_dir)
    create_directory(results_dir)
    return results_dir, reproducibility_dir


def plot_results(results, plots_dir=None, show_plots=False, extension='png'):
    plot_output(results['train_results'], 'Train', plots_dir=plots_dir, extension=extension)
    if 'dev_results' in results:
        plot_output(results['dev_results'], 'Dev', plots_dir=plots_dir, extension=extension)
    if 'test_results' in results:
        plot_output(results['test_results'], 'Test', plots_dir=plots_dir, extension=extension)
    plt.figure()
    plt.title(f'Learning profile', fontsize=12)
    plt.plot(TorchUtils.get_numpy_from_tensor(results['train_results']['performance_history']), label='Train')
    if 'dev_results' in results:
        plt.plot(TorchUtils.get_numpy_from_tensor(results['dev_results']['performance_history']), label='Dev')
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
    plt.title(f"{label} Output (nA) \n Performance: {results['performance']} \n Accuracy: {results['accuracy']['accuracy_value']}", fontsize=12)
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


if __name__ == '__main__':
    from torchvision import transforms

    from brainspy.utils import manager
    from brainspy.utils.io import load_configs
    from brainspy.utils.transforms import DataToTensor, DataToVoltageRange, DataPointsToPlateau

    from brainspy.processors.dnpu import DNPU

    V_MIN = [-1.2, -1.2]
    V_MAX = [0.7, 0.7]

    configs = load_configs('configs/ring.yaml')

    data_transforms = transforms.Compose([
        DataToVoltageRange(V_MIN, V_MAX, -1, 1),
        DataToTensor()
    ])

    # Add your custom transformations for the datapoints
    # waveform_transforms = transforms.Compose([
    #     DataPointsToPlateau(configs['processor']['waveform'])
    # ])

    criterion = manager.get_criterion(configs['algorithm'])
    algorithm = manager.get_algorithm(configs['algorithm'])

    dataloaders = get_ring_data(configs, data_transforms)

    ring_task(configs, dataloaders, DNPU, criterion, algorithm)  # , waveform_transforms=waveform_transforms)
