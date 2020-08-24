import os
import torch
import matplotlib.pyplot as plt

from brainspy.utils.io import load_configs
from bspytasks.boolean.tasks.classifier import postprocess
from bspytasks.boolean.tasks.classifier import plot_results

from brainspy.utils.io import create_directory, create_directory_timestamp

# TODO: Add possibility to validate multiple times


def validate_gate(gate_base_dir, results_dir=None, model_name='model.pt', is_main=True):
    # configs = load_configs(os.path.join(gate_base_dir, 'configs.yaml'))
    if results_dir is None:
        results_dir = os.path.join(os.path.split(gate_base_dir)[0], 'validation')
    model = torch.load(os.path.join(gate_base_dir, model_name))
    results = process_results(torch.load(os.path.join(gate_base_dir, 'results.pickle')))
    results_dir = init_dirs(results_dir, is_main=is_main, gate=results['gate'])
    with torch.no_grad():
        model.eval()
        # model.hw_eval()
        predictions = model(results['inputs'])

    results['hw_validation'] = postprocess(results['gate'], results['simulation'], model, results['simulation']['threshold'], node=results['simulation']['accuracy']['node'], save_dir=results_dir)
    results['hw_validation']['predictions'] = predictions
    results['simulation']['summary'] = results['simulation']['summary'] + '\n Accuracy (Hardware): ' + str(results['hw_validation']['accuracy']['accuracy_value'].item()) + '/' + str(results['hw_validation']['threshold'])
    plot_validation_results(results, save_dir=results_dir)
    torch.save(results, os.path.join(results_dir, 'hw_validation_results.pickle'))


def validate_vcdim(vcdim_base_dir, model_name='model.pt', is_main=True):
    base_dir = init_dirs(os.path.join(vcdim_base_dir, 'validation'))
    dirs = [os.path.join(vcdim_base_dir, o) for o in os.listdir(vcdim_base_dir) if os.path.isdir(os.path.join(vcdim_base_dir, o))]
    for d in dirs:
        if os.path.split(d)[1] != 'validation':
            validate_gate(os.path.join(d, 'reproducibility'), base_dir, is_main=False)


def process_results(results):
    new_results = {}
    new_results['inputs'] = results['inputs']
    new_results['targets'] = results['targets']
    new_results['gate'] = results['gate']
    # Add waveforms here in relevant places
    del results['inputs']
    # del results['targets']
    del results['gate']
    new_results['simulation'] = results
    return new_results


def plot_validation_results(results, save_dir):
    fig = plt.figure()
    plt.plot(results['hw_validation']['predictions'].detach().cpu(), label='Prediction (Hardware)')
    plt.plot(results['hw_validation']['targets'].detach().cpu(), label='Target (Hardware)')
    plot_results(results['simulation'], fig=fig, save_dir=save_dir)


def init_dirs(base_dir, is_main=True, gate=''):
    if is_main:
        base_dir = create_directory_timestamp(base_dir, gate)
    else:
        base_dir = os.path.join(base_dir, gate)
        create_directory(base_dir)
    return base_dir


if __name__ == "__main__":

    from brainspy.processors.dnpu import DNPU
    import numpy as np
    import datetime as d

    # transforms = transforms.Compose([
    #     ToTensor()
    # ])

    # find_gate(configs, gate, DNPU, threshold=0.8, transforms=transforms, logger=logger)
    validate_gate('tmp/TEST/output/boolean/[0, 0, 0, 1]_2020_08_17_131228/reproducibility')
    # validate_vcdim('tmp/TEST/output/boolean/vc_dimension_4_2020_08_06_133047/')
