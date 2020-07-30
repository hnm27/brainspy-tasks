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


def get_ring_data(sample_no, gap, transforms, batch_size, num_workers, data_dir=None, save_dir=None):
    # Returns dataloaders and split indices
    if data_dir:
        dataset = RingDataLoader(data_dir, transforms=transforms, save_dir=save_dir)
    else:
        dataset = RingDataGenerator(sample_no, gap, transforms=transforms, save_dir=save_dir)
    dataloaders, split_indices = split(dataset, batch_size, num_workers=num_workers)
    return dataset, dataloaders, split_indices


def train_classifier(gap, model, criterion, optimizer, epochs, transforms=None, logger=None, data_dir=None, save_dir=None):
    veredict = False

    #loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=False)
    dataset, dataloaders, split_indices = get_ring_data(1000, gap, transforms=transforms, batch_size=512, num_workers=0, data_dir=data_dir, save_dir=save_dir)
    model, performances = train(model, (dataloaders[0], dataloaders[1]), epochs, criterion, optimizer, logger=logger, save_dir=save_dir)
    return dataset, model, performances, split_indices


def postprocess(dataset, model, logger, threshold, save_dir=None):
    results = {}
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)

    results['dev_inputs'] = inputs
    results['dev_targets'] = targets
    results['predictions'] = predictions
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


def classify_ring(gap, model, criterion, optimizer, epochs, transforms=None, logger=None, base_dir='tmp/output/ring/', is_main=True):

    main_dir, results_dir, reproducibility_dir = init_dirs(str(gap), base_dir, is_main)

    print('==========================================================================================')
    print("GAP: " + str(gap))
    dataset, model, performances, split_indices = train_classifier(gap, model, criterion, optimizer, epochs, transforms=transforms, logger=logger, save_dir=reproducibility_dir)
    results = postprocess(dataset[split_indices[1]], model, logger, main_dir)
    results['split_indices'] = split_indices
    results['train_performance'] = performances[0]
    results['dev_performance'] = performances[1]
    # print(results['summary'])
    if is_main:
        torch.save(results, os.path.join(reproducibility_dir, 'results.pickle'))
    print('==========================================================================================')
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
        ToVoltageRange(V_MIN, V_MAX, -1, 1),
        ToTensor()
    ])

    model = TorchUtils.format_tensor(DNPU(configs))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))

    classify_ring(0.004, model, fisher, optimizer, epochs=80, transforms=transforms, logger=logger)
