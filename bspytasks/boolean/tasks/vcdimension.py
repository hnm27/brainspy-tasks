
import os
import torch
import matplotlib.pyplot as plt

from bspytasks.boolean.tasks.classifier import boolean_task
from bspytasks.boolean.data import generate_targets
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspyproc.utils.pytorch import TorchUtils


def vc_dimension_task(current_dimension, custom_model, configs, transforms, logger, is_main=True):
    print('---------------------------------------------')
    print(f'\tVC DIMENSION {str(current_dimension)}\t')
    print('---------------------------------------------')
    # REMOVE THIS

    threshold = (1 - (configs['threshold_parameter'] / current_dimension)) * 100.0
    targets = generate_targets(current_dimension)
    accuracies = torch.zeros(len(targets))
    performances = torch.zeros(len(targets), configs['algorithm']['epochs'])
    veredicts = torch.zeros(len(targets))
    correlations = torch.zeros(len(targets))

    base_dir = init_dirs(current_dimension, configs['results_base_dir'], is_main=is_main)
    configs['results_base_dir'] = base_dir
    for i in range(len(targets)):
        logger.gate = str(targets[i])
        results = boolean_task(configs, targets[i], custom_model, threshold, transforms=transforms, logger=logger, is_main=False)
        accuracies[i] = results['accuracy']['accuracy_value']
        performances[i] = results['performance_history']
        veredicts[i] = results['veredict']
        correlations[i] = results['correlation']
        del results
    results = {'capacity': torch.mean(veredicts), 'threshold': threshold, 'targets': targets, 'accuracies': accuracies, 'performances': performances, 'veredicts': veredicts, 'correlations': correlations}
    plot_results(results, base_dir=base_dir)
    torch.save(results, os.path.join(base_dir, 'vcdim_' + str(current_dimension) + '.pickle'))
    return results


def init_dirs(dimension, base_dir, is_main):
    results_folder_name = 'vc_dimension_' + str(dimension)
    if is_main:
        base_dir = create_directory_timestamp(base_dir, results_folder_name)
        create_directory(base_dir)
    else:
        base_dir = os.path.join(base_dir, results_folder_name)
        create_directory(base_dir)
    return base_dir


def plot_results(results, base_dir=None, show_plots=False):
    fig = plt.figure()
    correlations = TorchUtils.get_numpy_from_tensor(torch.abs(results['correlations']))
    threshold = TorchUtils.get_numpy_from_tensor(results['threshold'] * torch.ones(correlations.shape))
    accuracies = TorchUtils.get_numpy_from_tensor(results['accuracies'])
    plt.plot(correlations, threshold, 'k')
    plt.scatter(correlations, accuracies)
    plt.xlabel('Fitness / Performance')
    plt.ylabel('Accuracy')

    # create_directory(path)
    plt.savefig(os.path.join(base_dir, 'fitness_vs_accuracy.png'))
    if show_plots:
        plt.show()
    plt.close()
    return fig


if __name__ == "__main__":
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

    results = vc_dimension_test(4, DNPU, configs, transforms=transforms, logger=logger)
