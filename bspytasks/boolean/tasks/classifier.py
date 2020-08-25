import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from bspytasks.boolean.data import BooleanGateDataset


from brainspy.utils.manager import get_optimizer
from brainspy.utils.io import save, create_directory, create_directory_timestamp
from brainspy.algorithms.modules.performance.accuracy import get_accuracy, plot_perceptron
from brainspy.algorithms.modules.signal import corr_coeff


def boolean_task(configs, custom_model, criterion, algorithm, data_transforms=None, waveform_transforms=None, logger=None, is_main=True):
    main_dir, reproducibility_dir = init_dirs(str(configs['gate']), configs['results_base_dir'], is_main)
    gate = np.array(configs['gate'])
    loader = get_data(gate, data_transforms, configs)
    print('==========================================================================================')
    print("GATE: " + str(gate))
    for i in range(configs['max_attempts'] + 1):
        print('ATTEMPT: ' + str(i))
        model = custom_model(configs['processor'])
        optimizer = get_optimizer(model, configs['algorithm'])
        model, training_data = algorithm(model, (loader, None), criterion, optimizer, configs['algorithm'], waveform_transforms=waveform_transforms, logger=logger, save_dir=reproducibility_dir)

        results = evaluate_model(model, loader.dataset, transforms=waveform_transforms)
        results['train_results'] = training_data
        results['threshold'] = configs['threshold']
        results['gate'] = str(gate)
        results = postprocess(results, model, configs['algorithm']['accuracy'], training_data, logger=logger, save_dir=main_dir)

        if results['veredict']:
            break
    torch.save(results, os.path.join(reproducibility_dir, 'results.pickle'))
    save('configs', os.path.join(reproducibility_dir, 'configs.yaml'), data=configs)
    print('==========================================================================================')
    return results


def get_data(gate, data_transforms, configs):
    dataset = BooleanGateDataset(target=gate, transforms=data_transforms)
    if 'batch_size' in configs['algorithm']:
        batch_size = configs['algorithm']['batch_size']
    else:
        batch_size = len(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)


def postprocess(results, model, configs, training_data, logger=None, node=None, save_dir=None):
    if torch.isnan(results['predictions']).all():
        print('Nan values detected in the predictions. It is likely that the gradients of the model exploded. Skipping..')
        results['veredict'] = False
        return results
    results['accuracy'] = get_accuracy(results['predictions'], results['targets'], configs, node)  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results['correlation'] = corr_coeff(results['predictions'].T, results['targets'].T)

    if (results['accuracy']['accuracy_value'] >= results['threshold']):
        results['veredict'] = True
    else:
        results['veredict'] = False
    results['summary'] = 'VC Dimension: ' + str(len(results['targets'])) + ' Gate: ' + results['gate'] + ' Veredict: ' + str(results['veredict']) + '\n Accuracy (Simulation): ' + str(results['accuracy']['accuracy_value'].item()) + '/' + str(results['threshold'])
    results['results_fig'] = plot_results(results, save_dir)
    results['accuracy_fig'] = plot_perceptron(results['accuracy'], save_dir)
    results['performance_fig'] = plot_performance(results['train_results']['performance_history'], save_dir=save_dir)
    results['training_data'] = training_data
    print(results['summary'])
    if logger is not None:
        logger.log.add_figure('Results/VCDim' + str(len(results['targets'])) + '/' + results['gate'], results['results_fig'])
        logger.log.add_figure('Accuracy/VCDim' + str(len(results['targets'])) + '/' + results['gate'], results['accuracy_fig'])
    return results


def evaluate_model(model, dataset, results={}, transforms=None):
    with torch.no_grad():
        model.eval()
        if transforms is None:
            inputs, targets = dataset[:]
        else:
            inputs, targets = transforms(dataset[:])

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


def plot_performance(performance, save_dir=None, fig=None, show_plots=False):
    if fig is None:
        plt.figure()
    plt.title(f'Learning profile', fontsize=12)
    plt.plot(performance)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"training_profile"))
    return fig


if __name__ == "__main__":
    import numpy as np
    import datetime as d
    from torchvision import transforms

    from brainspy.utils import manager
    from bspytasks.boolean.logger import Logger
    from brainspy.utils.io import load_configs
    from brainspy.utils.transforms import DataToTensor, DataToVoltageRange, DataPointsToPlateau
    from brainspy.processors.dnpu import DNPU

    configs = load_configs('configs/boolean.yaml')

    data_transforms = transforms.Compose([
        # DataPointToPlateau(configs['processor']['waveform']),
        DataToTensor()
    ])

    waveform_transforms = transforms.Compose([
        DataPointsToPlateau(configs['processor']['waveform'])
    ])

    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    configs['gate'] = [0, 0, 0, 1]
    configs['threshold'] = 0.8

    criterion = manager.get_criterion(configs['algorithm'])
    algorithm = manager.get_algorithm(configs['algorithm'])

    boolean_task(configs, DNPU, criterion, algorithm, data_transforms=data_transforms, waveform_transforms=waveform_transforms, logger=logger)
