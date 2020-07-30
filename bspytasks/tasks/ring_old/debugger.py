import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from bspyproc.utils.waveform import generate_waveform
from bspyalgo.utils.io import create_directory


class ArchitectureDebugger():

    def __init__(self, configs):
        self.configs = configs
        self.plot_names = self.generate_plot_names(2)

    def init_dirs(self, base_dir):
        debug_path = os.path.join(base_dir, 'debug')
        self.debug_path_hardware = os.path.join(debug_path, 'hardware')
        self.debug_path_simulation = os.path.join(debug_path, 'simulation')
        self.results_path = os.path.join(debug_path, 'plots')
        create_directory(self.results_path)
        self.error_path = os.path.join(self.results_path, 'error')
        create_directory(self.error_path)

    def generate_plot_names(self, layer_no):
        result = []
        for i in range(layer_no):
            result.append('device_layer_' + str(i + 1) + '_output_0')
            result.append('bn_afterclip_' + str(i + 1) + '_0')
            result.append('bn_afterbatch_' + str(i + 1) + '_0')
            result.append('bn_aftercv_' + str(i + 1) + '_0')

            result.append('device_layer_' + str(i + 1) + '_output_1')
            result.append('bn_afterclip_' + str(i + 1) + '_1')
            result.append('bn_afterbatch_' + str(i + 1) + '_1')
            result.append('bn_aftercv_' + str(i + 1) + '_1')
        result.append('device_layer_3')
        result.append('bn_afterbatch_3')
        result.append('output')
        return result

    def plot_comparison(self, a, b, name, mse, show=False):
        plt.figure()
        plt.plot(a, label='device')
        plt.plot(b, label='model')
        plt.title(name + '\n' + f'MSE: {mse}')
        plt.legend()
        plt.savefig(os.path.join(self.results_path, name + '.' + self.extension))
        if show:
            plt.show()
        plt.close()

    def plot_error(self, x, name, show=False):
        plt.figure()
        plt.plot(x, label='error')
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(self.error_path, name + '_error.' + self.extension))
        if show:
            plt.show()
        plt.close()

    def read(self, name, mask=None):
        a = np.load(os.path.join(self.debug_path_hardware, name + '.npy'))
        b = torch.load(os.path.join(self.debug_path_simulation, name + '.pt')).detach().cpu().numpy()
        # b = generate_waveform(b, self.configs['validation']['processor']['waveform']
        #                       ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])
        if mask is not None:
            a = a[mask]
            b = b[mask]

        return a, b

    def print_error(self, a, b, name):
        print('Error ' + name)
        error = (a - b)
        mse = (error ** 2).mean()
        print(f'     max error: {np.max(error)}')
        print(f'     mean error: {mse}')
        print(f'     std error: {(error ** 2).std()}')
        return mse

    def default_plot(self, i, name, mask):
        a, b = self.read(name, mask)
        name = str(i) + '_' + name
        mse = self.print_error(a, b, name)
        self.plot_comparison(a, b, name, mse)
        self.plot_error(mse, name)

    def plot_raw_input(self, mask):
        name = 'raw_input'
        a, b = self.read(name, mask)
        input_indices = self.configs['validation']['processor']['input_indices']
        mse = self.print_error(a[:, input_indices[0]], b[:, 0], '0_' + name + '_0')
        self.plot_comparison(a[:, input_indices[0]], b[:, 0], '0_' + name + '_0', mse)
        mse = self.print_error(a[:, input_indices[1]], b[:, 1], '1_' + name + '_1')
        self.plot_comparison(a[:, input_indices[1]], b[:, 1], '1_' + name + '_1', mse)

    def plot_data(self, mask=None, extension='png', show_plots=False):
        self.extension = extension
        self.plot_raw_input(mask)
        i = 2
        for name in self.plot_names:
            self.default_plot(i, name, mask)
            i += 1
