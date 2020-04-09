import os
import numpy as np
from bspyproc.utils.waveform import generate_waveform, generate_mask
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.input import normalise, map_to_voltage
from bspytasks.utils.datasets import generate_data

# @todo: This data should come from the model
# MAX_INPUT_VOLT = np.asarray([0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3])
# MIN_INPUT_VOLT = np.asarray([-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7])


class RingDataLoader():

    def __init__(self, configs):
        self.configs = configs

    def get_data(self, processor_configs, gap, istest=False):
        if self.configs['ring_data']['generate_data']:
            return self.generate_new_data(processor_configs, gap)
        else:
            return self.read_data(processor_configs, gap, istest=istest)

    def generate_new_data(self, processor_configs, gap):
        inputs, targets = generate_data(processor_configs, self.configs['ring_data']['sample_no'], self.configs['ring_data']['gap'])
        inputs, targets, mask = self.process_data(inputs, targets, processor_configs)

        return inputs, targets, mask

    def get_data_filename(self, gap, istest=False):
        if istest:
            testname = 'testset'
        else:
            testname = ''
        return os.path.join(self.configs['ring_data']['data_dir'], f'class_data_{gap}' + testname + '.npz')

    def read_data(self, processor_configs, gap, istest=False):

        with np.load(self.get_data_filename(gap, istest=istest)) as data:
            # inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            inputs = data['inputs']
            print('Input shape: ', inputs.shape)
            targets = data['targets']
            # targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)

        return self.process_data(inputs, targets, processor_configs=processor_configs)

    def process_data(self, inputs, targets, processor_configs):
        inputs = self.process_inputs(inputs, processor_configs)
        mask = generate_mask(targets, processor_configs['waveform']['amplitude_lengths'], slope_lengths=processor_configs['waveform']['slope_lengths'])
        targets = self.process_targets(targets, processor_configs)

        return inputs, targets, mask

    def process_inputs(self, inputs, processor_configs):
        assert inputs.shape[1] == len(processor_configs['input_indices'])
        # for i in range(inputs.shape[1]):
        #     inputs[:, i] = normalise(inputs[:, i])
        #     inputs[:, i] = map_to_voltage(inputs[:, i],
        #                                   MIN_INPUT_VOLT[processor_configs['input_indices'][i]],
        #                                   MAX_INPUT_VOLT[processor_configs['input_indices'][i]])
        # inputs[:, i] = generate_waveform(inputs[:, i], processor_configs['waveform']['amplitude_lengths'], slope_lengths=processor_configs['waveform']['slope_lengths'])
        # inputs = self.generate_data_waveform(inputs, processor_configs['waveform']['amplitude_lengths'], processor_configs['waveform']['slope_lengths'])
        if processor_configs["processor_type"] == 'dnpu':
            return TorchUtils.get_tensor_from_numpy(inputs)
        return inputs

    def process_targets(self, targets, processor_configs):
        mask = (targets == 1)
        targets[targets == 0] = 1
        targets[mask] = 0
        # targets = np.asarray(generate_waveform(targets, processor_configs['waveform']['amplitude_lengths'], processor_configs['waveform']['slope_lengths'])).T
        if processor_configs["processor_type"] == 'dnpu':
            return TorchUtils.get_tensor_from_numpy(targets)
        return targets

    def generate_data_waveform(self, data, amplitude_lengths, slope_lengths):
        data_waveform = np.array([])
        for i in range(data.shape[1]):
            d_waveform = generate_waveform(data[:, i], amplitude_lengths, slope_lengths=slope_lengths)
            if data_waveform.shape == (0,):
                data_waveform = np.concatenate((data_waveform, d_waveform))
            else:
                data_waveform = np.vstack((data_waveform, d_waveform))

        return data_waveform.T
