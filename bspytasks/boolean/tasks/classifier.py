import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from bspytasks.boolean.data import BooleanGateDataset


from bspytasks.utils.manager import get_criterion, get_optimizer, get_algorithm
from bspytasks.utils.io import save, create_directory, create_directory_timestamp
from bspyalgo.algorithms.performance import perceptron, corr_coeff_torch, plot_perceptron


def boolean_task(configs, gate, custom_model, threshold, data_transforms=None, waveform_transforms=None, logger=None, is_main=True):
    main_dir, reproducibility_dir = init_dirs(str(gate), configs['results_base_dir'], is_main)
    gate = np.array(gate)
    criterion = get_criterion(configs['algorithm'])
    algorithm = get_algorithm(configs['algorithm'])
    loader = get_data(gate, data_transforms, configs)
    print('==========================================================================================')
    print("GATE: " + str(gate))
    for i in range(configs['max_attempts'] + 1):
        print('ATTEMPT: ' + str(i))
        model = custom_model(configs['processor'])
        optimizer = get_optimizer(model, configs['algorithm'])
        model, training_data = algorithm(model, (loader, None), criterion, optimizer, configs['algorithm'], waveform_transforms=waveform_transforms, logger=logger, save_dir=reproducibility_dir)
        results = evaluate_model(model, loader.dataset, transforms=waveform_transforms)
        results = postprocess(str(gate), results, model, threshold, logger=logger, save_dir=main_dir)
        results['training_data'] = training_data
        print(results['summary'])
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


def evaluate_model(model, dataset, transforms=None):
    results = {}
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


if __name__ == "__main__":
    import numpy as np
    import datetime as d
    from torchvision import transforms

    from bspytasks.boolean.logger import Logger
    from bspytasks.utils.io import load_configs
    from bspyalgo.algorithms.transforms import DataToTensor, DataToVoltageRange, DataPointsToPlateau
    from bspyproc.processors.dnpu import DNPU

    configs = load_configs('configs/boolean.yaml')

    data_transforms = transforms.Compose([
        # DataPointToPlateau(configs['processor']['waveform']),
        DataToTensor()
    ])

    waveform_transforms = transforms.Compose([
        DataPointsToPlateau(configs['processor']['waveform'])
    ])

    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    gate = [0, 0, 0, 1]

    boolean_task(configs, gate, DNPU, threshold=0.8, data_transforms=data_transforms, waveform_transforms=waveform_transforms, logger=logger)
