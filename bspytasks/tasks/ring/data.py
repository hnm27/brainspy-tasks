
import os
import numpy as np

from torch.utils.data import Dataset


class RingDatasetLoader(Dataset):
    def __init__(self, file_path, transforms=None, save_dir=None, verbose=True):
        # self.scale, self.offset = get_voltage_conversion_vars(input_voltage_range)
        data = np.load(file_path)
        self.inputs, self.targets = data['inputs'], data['targets']
        self.gap = data['gap']
        self.transforms = transforms
        assert len(self.inputs) == len(self.targets), "Targets and inputs must have the same length"
        if verbose:
            print(f'There are a total of {len(self.inputs[self.targets == 0]) + len(self.inputs[self.targets == 1])} samples')
            print(f"The input ring dataset has a {self.gap} gap (In a range from -1 to 1).")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = (self.inputs[index, :], self.targets[index, :])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class RingDatasetGenerator(Dataset):

    def __init__(self, sample_no, gap, transforms=None, save_dir=None, verbose=True):
        # The gap is a value between 0 and 1
        # The sample_no is related to the data that is going to be generated but it actually gets reduced when filtering the circles
        # TODO: Make the dataset generate the exact number of samples as requested by the user
        self.transforms = transforms
        # self.scale, self.offset = get_voltage_conversion_vars(input_voltage_range)
        self.inputs, targets = self.generate_data(sample_no, gap, verbose=verbose)
        self.targets = targets[:, np.newaxis]
        self.gap = gap
        assert len(self.inputs) == len(self.targets), "Targets and inputs must have the same length"

        if save_dir is not None:
            np.savez(os.path.join(save_dir, 'input_data_gap_' + str(gap)), gap=gap, inputs=self.inputs, targets=self.targets)
            # TODO: Save a plot of the data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = (self.inputs[index, :], self.targets[index, :])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def generate_data(self, sample_no, gap, verbose=True):
        # Get information from the electrode ranges in order to calculate the linear transformation parameters for the inputs
        # min_voltage, max_voltage = load_voltage_ranges(configs)
        # i = configs['input_indices']
        # scale, offset = get_map_to_voltage_vars(min_voltage[i], max_voltage[i])
        # gap = self.transform_gap(gap, self.scale)
        length_factor = 20

        while True:
            data, labels = self.ring(sample_no=sample_no * length_factor, gap=gap)
            if (min(len(labels[labels == 0]), len(labels[labels == 1])) * 2) > sample_no:
                break
            length_factor = length_factor * 2

        data, labels = self.process_dataset(sample_no, data[labels == 0], data[labels == 1])

        # Transform dataset to control voltage range
        # samples = (data * self.scale) + self.offset
        # gap = gap * self.scale
        if verbose:
            print(f'There are a total of {len(data[labels == 0]) + len(data[labels == 1])} samples')
            print(f'The input ring dataset has a {gap} gap (In a range from -1 to 1).')
        return data, labels

    def ring(self, sample_no, inner_radius=0.25, gap=0.5, outer_radius=1):
        '''Generates labelled TorchUtilsdata of a ring with class 1 and the center with class 0'''
        assert outer_radius <= 1

        samples = (-1 * outer_radius) + (2 * outer_radius * np.random.rand(sample_no, 2))
        norm = np.sqrt(np.sum((samples)**2, axis=1))

        # Filter out samples outside the classes
        labels = np.empty(samples.shape[0])
        labels[norm < inner_radius] = 0
        labels[(norm < outer_radius) * (norm > inner_radius + gap)] = 1
        labels[norm > outer_radius] = np.nan
        labels[(norm > inner_radius) * (norm < inner_radius + gap)] = np.nan

        return samples, labels

    def subsample(self, class0, class1):
            # Subsample the largest class
        nr_samples = min(len(class0), len(class1))
        max_array = max(len(class0), len(class1))
        indices = np.random.permutation(max_array)[:nr_samples]
        if len(class0) == max_array:
            class0 = class0[indices]
        else:
            class1 = class1[indices]
        return class0, class1

    def sort(self, class0, class1):
        # Sort samples within each class wrt the values of input x (i.e. index 0)
        sorted_index0 = np.argsort(class0, axis=0)[:, 0]
        sorted_index1 = np.argsort(class1, axis=0)[:, 0]
        return class0[sorted_index0], class1[sorted_index1]

    def filter_and_reverse(self, class0, class1):
        # Filter by positive and negative values of y-axis
        class0_positive_y = class0[class0[:, 1] >= 0]
        class0_negative_y = class0[class0[:, 1] < 0][::-1]
        class1_positive_y = class1[class1[:, 1] >= 0]
        class1_negative_y = class1[class1[:, 1] < 0][::-1]

        # Define input variables and their target
        class0 = np.concatenate((class0_positive_y, class0_negative_y))
        class1 = np.concatenate((class1_positive_y, class1_negative_y))
        inputs = np.concatenate((class0, class1))
        targets = np.concatenate((np.zeros_like(class0[:, 0]), np.ones_like(class1[:, 0])))

        # Reverse negative 'y' inputs for negative cases
        return inputs, targets

    def process_dataset(self, sample_no, class0, class1):

        class0, class1 = self.subsample(class0, class1)
        class0, class1 = self.reduce_length(sample_no, class0, class1)
        class0, class1 = self.sort(class0, class1)

        return self.filter_and_reverse(class0, class1)

    def reduce_length(self, sample_no, class0, class1):
        max_index = int(sample_no / 2)
        return class0[:max_index], class1[:max_index]
        # The gap needs to be in a scale from -1 to 1. This function enables to transform the gap in volts to this scale.

    def transform_gap(self, gap_in_volts, scale):
        assert (len(scale[scale == scale.mean()]) == len(scale)), "The GAP information is going to be inaccurate because the selected input electrodes have a different voltage range. In order for this data to be accurate, please make sure that the input electrodes have the same voltage ranges."
        if len(scale) > 1:
            scale = scale[0]

        return (gap_in_volts / scale)


def get_voltage_conversion_vars(input_voltage_range):
    return get_map_to_voltage_vars(np.array(input_voltage_range[0]), np.array(input_voltage_range[1]), np.array([-1, -1]), np.array([1, 1]))


def get_map_to_voltage_vars(v_min, v_max, x_min, x_max):
    scale = ((v_min - v_max) / (x_min - x_max))
    offset = v_max - scale * x_max
    return scale, offset
