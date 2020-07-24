#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz and ualegre
"""
import os
import pickle
from bspytasks.benchmarks.vcdim.vc_dimension_test import VCDimensionTest
from bspyalgo.utils.io import create_directory_timestamp, save
import numpy as np
import matplotlib.pyplot as plt


class CapacityTest():

    def __init__(self, configs):
        self.current_dimension = configs['from_dimension']
        configs = self.init_dirs(configs)
        self.configs = configs
        self.vcdimension_test = VCDimensionTest(configs['vc_dimension_test'], is_main=False)

    def init_dirs(self, configs):
        base_dir = create_directory_timestamp(configs['results_base_dir'], 'capacity_test')
        configs['vc_dimension_test']['results_base_dir'] = base_dir
        self.configs_dir = os.path.join(base_dir, 'capacity_configs.json')
        return configs

    def run_test(self, validate=False):
        print('*****************************************************************************************')
        print(f"CAPACITY TEST FROM VCDIM {self.configs['from_dimension']} TO VCDIM {self.configs['to_dimension']} ")
        print('*****************************************************************************************')
        save(mode='configs', file_path=self.configs_dir, data=configs)
        self.summary_results = {'capacity_per_N': [],
                                'accuracy_distib_per_N': [],
                                'performance_distrib_per_N': [],
                                'correlation_distrib_per_N': []}
        while True:
            capacity, accuracy_array, performance_array, correlation_array = self.vcdimension_test.run_test(self.current_dimension, validate=validate)
            self.summary_results['capacity_per_N'].append(capacity)
            self.summary_results['accuracy_distib_per_N'].append(accuracy_array[1:-1])
            self.summary_results['performance_distrib_per_N'].append(performance_array[1:-1])
            self.summary_results['correlation_distrib_per_N'].append(correlation_array[1:-1])
            if not self.next_vcdimension():
                break

        self.vcdimension_test.close_results_file()
        self.plot_summary()
        dict_loc = os.path.join(self.configs['vc_dimension_test']['results_base_dir'], 'summary_results.pkl')
        with open(dict_loc, 'wb') as fp:
            pickle.dump(self.summary_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('*****************************************************************************************')

    def next_vcdimension(self):
        if self.current_dimension + 1 > self.configs['to_dimension']:
            return False
        else:
            self.current_dimension += 1
            return True

    def plot_summary(self):
        dimensions = np.arange(self.configs['from_dimension'], self.configs['to_dimension'] + 1)
        plt.figure()
        plt.plot(dimensions, self.summary_results['capacity_per_N'])
        plt.title('Capacity over N points')
        plt.xlabel('Nr. of points N')
        plt.ylabel('Capacity')
        file_path = os.path.join(self.configs['vc_dimension_test']['results_base_dir'], "Capacity_over_N")
        plt.savefig(file_path)

        self.plot_boxplot(dimensions, 'accuracy_distib_per_N', title='Accuracy over N points')
        self.plot_boxplot(dimensions, 'performance_distrib_per_N', title='Performance over N points')
        self.plot_boxplot(dimensions, 'correlation_distrib_per_N', title='Correlation over N points')

        plt.show()

    def plot_boxplot(self, pos, key, title=''):
        plt.figure()
        plt.title(title)
        plt.boxplot(self.summary_results[key], positions=pos)
        plt.xlabel('Nr. of points N')
        plt.ylabel(key.split('_')[0])
        file_path = os.path.join(self.configs['vc_dimension_test']['results_base_dir'], key)
        plt.savefig(file_path)


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('configs/benchmark_tests/capacity/template_ga_simulation.json')

    test = CapacityTest(configs['capacity_test'])
    test.run_test(validate=False)
