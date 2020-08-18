
import os
import torch
import numpy as np

from torch.utils.data import Dataset, Sampler
from torch.utils.data import random_split, SubsetRandomSampler


class RingDatasetGenerator(Dataset):

    def __init__(self, sample_no, gap, transforms=None, save_dir=None, verbose=True):
        # The gap needs to be in a scale from -1 to 1.
        # The sample_no is related to the data that is going to be generated but it actually gets reduced when filtering the circles
        # TODO: Make the dataset generate the exact number of samples as requested by the user
        self.transforms = transforms
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
        length_factor = 20

        while True:
            data, labels = self.ring(sample_no=sample_no * length_factor, gap=gap)
            if (min(len(labels[labels == 0]), len(labels[labels == 1])) * 2) > sample_no:
                break
            length_factor = length_factor * 2

        data, labels = self.process_dataset(sample_no, data[labels == 0], data[labels == 1])

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
        inputs, targets = self.filter_and_reverse(class0, class1)
        indices = self.get_balanced_distribution_indices(len(targets))

        return inputs[indices], targets[indices]

    def get_balanced_distribution_indices(self, data_length):
        permuted_indices = np.random.permutation(data_length)
        class0 = permuted_indices[permuted_indices < int(data_length / 2)]
        class1 = permuted_indices[permuted_indices >= int(data_length / 2)]
        assert len(class0) == len(class1), 'Sampler only supports datasets with an even class distribution'
        result = []
        for i in range(len(class0)):
            result.append(class0[i])
            result.append(class1[i])
        return np.array(result)

    def reduce_length(self, sample_no, class0, class1):
        max_index = int(sample_no / 2)
        return class0[:max_index], class1[:max_index]

    def transform_gap(self, gap_in_volts, scale):
        assert (len(scale[scale == scale.mean()]) == len(scale)), "The GAP information is going to be inaccurate because the selected input electrodes have a different voltage range. In order for this data to be accurate, please make sure that the input electrodes have the same voltage ranges."
        if len(scale) > 1:
            scale = scale[0]

        return (gap_in_volts / scale)


class RingDatasetLoader(Dataset):
    def __init__(self, file_path, transforms=None, save_dir=None, verbose=True):

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


class BalancedSubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, generator=None):
        super().__init__(indices)
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in balanced_permutation(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def split(dataset, batch_size, num_workers, sampler=SubsetRandomSampler, split_percentages=[0.8, 0.1, 0.1]):
    # Split percentages are expected to be in the following format: [80,10,10]
    percentages = np.array(split_percentages)
    assert np.sum(percentages) == 1, 'Split percentage does not sum up to 1'
    indices = list(range(len(dataset)))
    indices = balanced_permutation(len(dataset))
    max_train_index = int(np.floor(percentages[0] * len(dataset)))
    max_dev_index = int(np.floor((percentages[0] + percentages[1]) * len(dataset)))
    max_test_index = int(np.floor(np.sum(percentages) * len(dataset)))

    train_index = indices[:max_train_index]
    dev_index = indices[max_train_index:max_dev_index]
    test_index = indices[max_dev_index:max_test_index]

    train_sampler = sampler(train_index)
    dev_sampler = sampler(dev_index)
    test_sampler = sampler(test_index)
    if batch_size > 0:
        batch_size = [batch_size, batch_size, batch_size]
    else:

        batch_size = [get_batch_size(train_sampler), get_batch_size(dev_sampler), get_batch_size(test_sampler)]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size[0], sampler=train_sampler, num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size[1], sampler=dev_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size[2], sampler=test_sampler, num_workers=num_workers)

    return [train_loader, dev_loader, test_loader]  # , [train_index, dev_index, test_loader]
    # lengths = [int(len(dataset) * split_percentages[0]), int(len(dataset) * split_percentages[1]), int(len(dataset) * split_percentages[2])]
    # datasets = random_split(dataset, lengths)
    # samplers = [sampler(ds.indices) for ds in datasets]

    # return [torch.utils.data.DataLoader(datasets[i], sampler=samplers[i], batch_size=batch_size, num_workers=num_workers) for i in range(len(datasets))]


def get_batch_size(sampler):
    if len(sampler) > 0:
        return len(sampler)
    else:
        return 1


def balanced_permutation(len_indices):
    permuted_indices = torch.randperm(len_indices)
    class0 = permuted_indices[permuted_indices % 2 == 0]
    class1 = permuted_indices[permuted_indices % 2 == 1]
    assert len(class0) == len(class1), 'Sampler only supports datasets with an even class distribution'
    result = []
    for i in range(len(class0)):
        result.append(class0[i])
        result.append(class1[i])
    return torch.tensor(result, dtype=torch.int64)
