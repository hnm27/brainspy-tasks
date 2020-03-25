from bspyproc.bspyproc import get_processor
from bspytasks.tasks.ring.classifier import RingClassificationTask
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspyproc.utils.waveform import generate_waveform, generate_mask
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.performance import perceptron
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspytasks.tasks.ring.debugger import ArchitectureDebugger
from bspyalgo.utils.performance import accuracy


import matplotlib.pyplot as plt
import numpy as np
import os


class RingClassifierValidator():

    def __init__(self, configs):
        self.configs = configs
        self.init_processors()
        self.init_data()
        self.debugger = ArchitectureDebugger(configs)
        self.init_dirs()

    def init_data(self):
        self.data_loader = RingDataLoader(configs)
        test_inputs, test_targets, _ = self.data_loader.generate_new_data(self.configs['algorithm_configs']['processor'], gap=self.configs['ring_data']['gap'])
        self.test_results = {}
        self.test_results['inputs'] = test_inputs
        self.test_results['targets'] = test_targets

    def init_processors(self):
        self.validation_processor = get_processor(self.configs['validation']['processor'])
        self.processor = get_processor(self.configs['algorithm_configs']['processor'])

    def init_dirs(self):
        self.main_dir = create_directory_timestamp(os.path.join(self.configs['results_base_dir'], 'validation'), 'validation')
        self.debugger.init_dirs(self.main_dir)
        if self.processor.configs['debug'] and self.processor.configs['architecture'] == 'device_architecture':
            self.processor.init_dirs(self.main_dir, is_main=False)
        if self.validation_processor.configs['debug'] and self.validation_processor.configs['architecture'] == 'device_architecture':
            self.validation_processor.init_dirs(self.main_dir, is_main=False)
        self.debug_plots = os.path.join(self.main_dir, 'debug', 'results')
        create_directory(self.debug_plots)

    def get_model_output(self, model, results):
        self.processor.load_state_dict(model.copy())
        self.processor.eval()
        if self.configs['algorithm_configs']['processor']['platform'] == 'simulation':
            inputs = TorchUtils.get_tensor_from_numpy(results['inputs'])
        else:
            inputs = results['inputs']
        model_output = self.processor.forward(inputs).detach().cpu().numpy()
        return generate_waveform(model_output[:, 0], self.configs['validation']['processor']['waveform']
                                 ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])

    def get_hardware_output(self, model, results):
        self.validation_processor.load_state_dict(model.copy())
        inputs, targets, mask = self.get_validation_inputs(results)

        return self.validation_processor.get_output_(inputs, mask)[:, 0], mask

    def get_validation_inputs(self, results):

        targets = generate_waveform(results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        mask = generate_mask(results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])

        inputs_1 = generate_waveform(results['inputs'][:, 0], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        inputs_2 = generate_waveform(results['inputs'][:, 1], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        inputs = np.asarray([inputs_1, inputs_2]).T

        return inputs, targets, mask

    def validate(self, results, model, debugger_mask=True, debugger_extension='png'):
        model_output = self.get_model_output(model, results)
        real_output, mask = self.get_hardware_output(model, results)
        self.plot_validation_results(model_output, real_output, mask, self.main_dir, self.configs['show_plots'])
        if debugger_mask:
            self.debugger.plot_data(mask=mask, extension=debugger_extension)
        else:
            self.debugger.plot_data(extension=debugger_extension)

    def test_accuracy(self, model, debugger_mask=True, debugger_extension='png'):
        targets = generate_waveform(self.test_results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        # mask = generate_mask(self.test_results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        model_output = self.get_model_output(model, self.test_results)
        real_output, mask = self.get_hardware_output(model, self.test_results)
        self.plot_validation_results(model_output, real_output, mask, self.main_dir, self.configs['show_plots'], name='test_accuracy')

        simulation_accuracy = accuracy(model_output[mask],
                                       targets[mask],
                                       plot=os.path.join(self.main_dir, f"simulation_perceptron." + debugger_extension))
        print(f"Simulation accuracy: {simulation_accuracy}")

        hardware_accuracy = accuracy(real_output[mask],
                                     targets[mask],
                                     plot=os.path.join(self.main_dir, f"hardware_perceptron." + debugger_extension))
        print(f"Hardware accuracy: {hardware_accuracy}")
        return simulation_accuracy, hardware_accuracy

    def plot_validation_results(self, model_output, real_output, mask, save_dir=None, show_plot=False, name='validation_plot', extension='png'):

        error = ((model_output[mask][:, 0] - real_output[mask]) ** 2).mean()
        print(f'Total Error: {error}')

        plt.figure()
        plt.title(f'Comparison between Hardware and Simulation \n (MSE: {error})', fontsize=12)
        plt.plot(model_output[mask])
        plt.plot(real_output[mask], '-.')
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')

        plt.legend(['Simulation', 'Hardware'])
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, name + extension))
            np.savez(os.path.join(self.main_dir, name + '_data'), model_output=model_output, real_output=real_output, mask=mask)
        if show_plot:
            plt.show()
            plt.close()


if __name__ == '__main__':
    import torch
    import os
    import pickle
    from bspyalgo.utils.io import load_configs
    from bspytasks.utils.datasets import load_data

    folder_name = 'searcher_0.2mV_2020_03_03_204859'
    base_dir = os.path.join('tmp', 'output', 'ring', folder_name)
    model, results, configs = load_data(base_dir)
    val = RingClassifierValidator(configs)
    val.validate(results, model)
    # val.test_accuracy(model)
