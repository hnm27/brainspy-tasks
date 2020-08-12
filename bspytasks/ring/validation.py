import os
import torch
import matplotlib.pyplot as plt
from bspyalgo.utils.performance import perceptron, corr_coeff_torch
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspyproc.utils.pytorch import TorchUtils


def load_reproducibility_results(base_dir, model_name='best_model.pt'):
    base_dir = os.path.join(base_dir, 'reproducibility')
    # configs = load_configs(os.path.join(gate_base_dir, 'configs.yaml'))
    model = torch.load(os.path.join(base_dir, model_name))
    results = torch.load(os.path.join(base_dir, 'results.pickle'))
    return model, results  # , configs


def validate(model, results, criterion, results_dir, plot=None, transforms=None, is_main=True):

    #results_dir = init_dirs(results_dir, is_main=is_main, gate=results['gap'])
    if 'train_results' in results:
        results['train_results'] = apply_transforms(results['train_results'], transforms=transforms)
        results['train_results_hw'] = _validate(model, results['train_results'], criterion)
    if 'dev_results' in results:
        results['dev_results'] = apply_transforms(results['dev_results'], transforms=transforms)
        results['dev_results_hw'] = _validate(model, results['dev_results'], criterion)
    if 'test_results' in results:
        results['test_results'] = apply_transforms(results['test_results'], transforms=transforms)
        results['test_results_hw'] = _validate(model, results['test_results'], criterion)

    plot_all(results, save_dir=results_dir)
    torch.save(results, os.path.join(
        results_dir, 'hw_validation_results.pickle'))


def plot_all(results, save_dir=None):
    if 'train_results' in results:
        plot_validation_results(TorchUtils.get_numpy_from_tensor(results['train_results']['best_output']), TorchUtils.get_numpy_from_tensor(results['train_results_hw']['best_output']), save_dir=save_dir)
    if 'dev_results' in results:
        plot_validation_results(TorchUtils.get_numpy_from_tensor(results['dev_results']['best_output']), TorchUtils.get_numpy_from_tensor(results['dev_results_hw']['best_output']), save_dir=save_dir)
    if 'test_results' in results:
        plot_validation_results(TorchUtils.get_numpy_from_tensor(results['test_results']['best_output']), TorchUtils.get_numpy_from_tensor(results['test_results_hw']['best_output']), save_dir=save_dir)


def plot_validation_results(model_output, real_output, save_dir=None, show_plot=False, name='validation_plot', extension='png'):

    error = ((model_output - real_output) ** 2).mean()
    print(f'Total Error: {error}')

    plt.figure()
    plt.title(
        f'Comparison between Hardware and Simulation \n (MSE: {error})', fontsize=12)
    plt.plot(model_output)
    plt.plot(real_output, '-.')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')

    plt.legend(['Simulation', 'Hardware'])
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + '.' + extension))
        # np.savez(os.path.join(self.main_dir, name + '_data'),
        #         model_output=model_output, real_output=real_output, mask=mask)
    if show_plot:
        plt.show()
        plt.close()


def apply_transforms(results, transforms):
    if transforms is not None:
        results['inputs'] = transforms(results['inputs'])
        results['targets'] = transforms(results['targets'])
        results['best_output'] = transforms(results['best_output'])
    return results


def _validate(model, results, criterion):
    with torch.no_grad():
        model.eval()
        predictions = model(results['inputs'])
        results['performance'] = criterion(predictions, results['targets'])

    # results['gap'] = dataset.gap
    results['best_output'] = predictions
    results['accuracy'] = perceptron(predictions, results['targets'])  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results['correlation'] = corr_coeff_torch(predictions.T, results['targets'].T)
    return results


def init_dirs(base_dir, is_main=True, gate=''):
    if is_main:
        base_dir = create_directory_timestamp(base_dir, gate)
    else:
        base_dir = os.path.join(base_dir, gate)
        create_directory(base_dir)
    return base_dir


if __name__ == "__main__":
    from torchvision import transforms

    from bspyalgo.utils.transforms import PointsToPlateau
    from bspyalgo.algorithms.gradient.core.losses import fisher
    from bspyalgo.utils.io import load_configs

    base_dir = 'tmp/TEST/output/ring/gap_0.4/ring_classification_2020_08_12_154212'
    model, results = load_reproducibility_results(base_dir)

    configs = load_configs('configs/ring.yaml')
    waveform_transforms = transforms.Compose([
        PointsToPlateau(configs['validation_processor']['waveform'])
    ])

    results_dir = init_dirs(os.path.join(base_dir, 'validation'))
    validate(model, results, fisher, results_dir, transforms=waveform_transforms)
