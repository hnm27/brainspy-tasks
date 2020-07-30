from bspytasks.tasks.boolean.vcdimension import vc_dimension_test
import torch
from bspyalgo.utils.io import create_directory_timestamp
import os
import matplotlib.pyplot as plt
import numpy as np
from bspyproc.utils.pytorch import TorchUtils
import pickle


def capacity_test(from_dimension, to_dimension, custom_model, configs, criterion, custom_optimizer, epochs, transforms, logger, base_dir='tmp/output/boolean/capacity'):
    print('*****************************************************************************************')
    print(f"CAPACITY TEST FROM VCDIM {from_dimension} TO VCDIM {to_dimension} ")
    print('*****************************************************************************************')
    base_dir = create_directory_timestamp(base_dir, 'capacity_test')
    # save(mode='configs', file_path=self.configs_dir, data=configs)
    summary_results = {'capacity_per_N': [],
                       'accuracy_distrib_per_N': [],
                       'performance_distrib_per_N': [],
                       'correlation_distrib_per_N': []}
    for i in range(from_dimension, to_dimension + 1):
        # capacity, accuracy_array, performance_array, correlation_array = vc_dimension_test(self.current_dimension, validate=validate)
        results = vc_dimension_test(i, custom_model, configs, criterion, custom_optimizer, epochs, transforms=transforms, logger=logger, base_dir=base_dir, is_main=False)
        summary_results['capacity_per_N'].append(TorchUtils.get_numpy_from_tensor(results['capacity']))
        summary_results['accuracy_distrib_per_N'].append(TorchUtils.get_numpy_from_tensor(results['accuracies']))
        summary_results['performance_distrib_per_N'].append(TorchUtils.get_numpy_from_tensor(results['performances'][:, -1]))
        summary_results['correlation_distrib_per_N'].append(TorchUtils.get_numpy_from_tensor(results['correlations']))
        del results
    # self.vcdimension_test.close_results_file()
    # self.plot_summary()
    # dict_loc = os.path.join(self.configs['vc_dimension_test']['results_base_dir'], 'summary_results.pkl')
    with open(os.path.join(base_dir, 'summary_results.pickle'), 'wb') as fp:
        pickle.dump(summary_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #torch.save(summary_results, os.path.join(base_dir, 'summary_results.pickle'))
    plot_summary(summary_results, from_dimension, to_dimension, base_dir)
    print('*****************************************************************************************')


def plot_summary(results, from_dimension, to_dimension, base_dir=None):
    dimensions = np.arange(from_dimension, to_dimension + 1)
    plt.figure()
    plt.plot(dimensions, results['capacity_per_N'])
    plt.title('Capacity over N points')
    plt.xlabel('Nr. of points N')
    plt.ylabel('Capacity')
    if base_dir:
        plt.savefig(os.path.join(base_dir, "Capacity_over_N"))

    plot_boxplot(dimensions, results, 'accuracy_distrib_per_N', title='Accuracy over N points', base_dir=base_dir)
    plot_boxplot(dimensions, results, 'performance_distrib_per_N', title='Performance over N points', base_dir=base_dir)
    plot_boxplot(dimensions, results, 'correlation_distrib_per_N', title='Correlation over N points', base_dir=base_dir)

    plt.show()


def plot_boxplot(pos, results, key, title='', base_dir=None):
    plt.figure()
    plt.title(title)
    plt.boxplot(results[key], positions=pos)
    plt.xlabel('Nr. of points N')
    plt.ylabel(key.split('_')[0])
    if base_dir:
        plt.savefig(os.path.join(base_dir, key))


if __name__ == "__main__":
    from bspyproc.processors.dnpu import DNPU
    from bspytasks.utils.transforms import ToTensor
    from bspyalgo.algorithms.gradient.core.logger import Logger
    from bspyalgo.algorithms.gradient.core.losses import corrsig
    from torchvision import transforms
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

    # model = TorchUtils.format_tensor(DNPU(configs))

    logger = Logger(f'tmp/output/logs/experiment' + str(d.datetime.now().timestamp()))
    capacity_test(4, 5, DNPU, configs, corrsig, torch.optim.Adam, 80, transforms=transforms, logger=logger)
