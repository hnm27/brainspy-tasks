import os
import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.utils.performance import perceptron, corr_coeff
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.algorithm_manager import get_algorithm
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspyproc.bspyproc import get_processor


MAX_CLIPPING_VALUE = np.array([1.5])
MIN_CLIPPING_VALUE = np.array([-1.5])


class BooleanGateTask():

    def __init__(self, configs, is_main=True):
        # configs = self.load_directory_configs(configs)
        self.is_main = is_main
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.load_methods(configs['algorithm_configs'])
        self.load_task_configs(configs)

        if 'validation' in configs:
            self.validation_processor_configs = configs['validation']['processor']
            self.validation_processor = get_processor(configs['validation']['processor'])
        else:
            self.validation_processor_configs = None

    def load_task_configs(self, configs):
        self.show_plots = configs['show_plots']
        self.base_dir = configs['results_base_dir']
        self.max_attempts = configs['max_attempts']

    def init_dirs(self, gate):
        if self.is_main:
            base_dir = create_directory_timestamp(self.base_dir, gate)
        else:
            base_dir = os.path.join(self.base_dir, gate)
            create_directory(base_dir)
        return base_dir

    def load_methods(self, configs):
        if configs['algorithm'] == 'gradient_descent' and configs['processor']['platform'] == 'simulation':
            self.find_gate_core = self.find_gate_with_torch
            self.ignore_gate = self.ignore_gate_with_torch
        else:
            self.find_gate_core = self.find_gate_with_numpy
            self.ignore_gate = self.ignore_gate_with_numpy

    def find_gate(self, encoded_inputs, gate, encoded_gate, mask, threshold):
        #min_threshold = (1 - 1 / len(gate)) * 100.0
        #assert threshold >= min_threshold, f"Threshold cannot be less or equal than {min_threshold}; it is now {threshold}"
        if len(np.unique(gate)) == 1:
            print('Label ', gate, ' ignored')
            excel_results = self.ignore_gate(encoded_gate)
        else:
            attempt = 1
            print('==========================================================================================')
            print(f"Gate {gate}: ")
            base_dir = self.init_dirs(str(gate))
            self.gate_base_dir = base_dir
            self.algorithm.init_dirs(base_dir)
            while True:
                excel_results = self.find_gate_core(encoded_inputs, encoded_gate, mask)
                excel_results['found'] = excel_results['accuracy'] >= threshold

                print(f"Attempt {str(attempt)}: " + self.is_found(excel_results['found']).upper() + ", Accuracy: " + str(excel_results['accuracy']) + ", Best performance: " + str(excel_results['best_performance']))

                if excel_results['found'] or attempt >= self.max_attempts:
                    if excel_results['found']:
                        print(f'VEREDICT: PASS - Gate was found successfully in {str(attempt)} attempt(s)')
                    else:
                        print(f'VEREDICT: FAILED - Gate was NOT found in {str(attempt)} attempt(s)')
                    print('==========================================================================================')
                    #  in get_plot_dir
                    self.plot_gate(excel_results, mask, str(gate), show_plots=self.show_plots, save_dir=self.get_plot_dir(gate, base_dir))
                    break
                else:
                    attempt += 1

        excel_results['gate'] = gate

        return excel_results

    def get_plot_dir(self, gate, path, extension='png'):
        #path = os.path.join(self.base_dir, found)
        create_directory(path, overwrite=False)
        return os.path.join(path, str(gate) + '.' + extension)

    def find_gate_with_numpy(self, encoded_inputs, encoded_gate, mask):
        excel_results = self.optimize(encoded_inputs, encoded_gate, mask)
        excel_results['accuracy'], _, _, _ = perceptron(excel_results['best_output'][excel_results['mask']], encoded_gate[excel_results['mask']])
        excel_results['encoded_gate'] = encoded_gate
        return excel_results

    def find_gate_with_torch(self, encoded_inputs, encoded_gate, mask):
        encoded_gate = TorchUtils.format_tensor(encoded_gate)
        excel_results = self.optimize(encoded_inputs, encoded_gate, mask)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], TorchUtils.get_numpy_from_tensor(encoded_gate[excel_results['mask']]))
        excel_results['encoded_gate'] = encoded_gate.cpu()
        # excel_results['targets'] = excel_results
        excel_results['correlation'] = corr_coeff(excel_results['best_output'][excel_results['mask']].T, excel_results['targets'].cpu()[excel_results['mask']].T)
        return excel_results

    def ignore_gate_with_torch(self, encoded_gate):
        excel_results = self.ignore_gate_core()
        excel_results['encoded_gate'] = encoded_gate.cpu()
        return excel_results

    def ignore_gate_with_numpy(self, encoded_gate):
        excel_results = self.ignore_gate_core()
        excel_results['encoded_gate'] = encoded_gate
        return excel_results

    def ignore_gate_core(self):
        excel_results = {}
        excel_results['control_voltages'] = np.nan
        excel_results['best_output'] = np.nan
        excel_results['best_performance'] = np.nan
        excel_results['accuracy'] = np.nan
        excel_results['correlation'] = np.nan
        excel_results['found'] = True
        return excel_results

    def optimize(self, encoded_inputs, encoded_gate, mask):
        algorithm_data = self.algorithm.optimize(encoded_inputs, encoded_gate, mask=mask)
        algorithm_data.judge()
        excel_results = algorithm_data.results

        return excel_results

    def plot_gate(self, row, mask, gate, show_plots, save_dir=None):
        plt.figure()
        plt.title(gate + ' ' + self.is_found(row['found']))
        plt.plot(row['best_output'][mask])
        plt.plot(row['encoded_gate'][mask])
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')
        if save_dir is not None:
            plt.savefig(save_dir)
        if show_plots:
            plt.show()
        plt.close()

    def clip(self, x, max_value, min_value):
        x[x > max_value] = max_value
        x[x < min_value] = min_value
        return x

    def is_found(self, found):
        if found:
            return 'found'
        else:
            return 'not_found'

    def validate_gate(self, gate, transformed_inputs, control_voltages, y_predicted, mask, base_dir):
        print('==========================================================================================')
        print(f"Gate {gate} validation: ")
        transformed_inputs = self.clip(transformed_inputs, max_value=MAX_CLIPPING_VALUE, min_value=MIN_CLIPPING_VALUE)
        control_voltages = self.clip(control_voltages, max_value=MAX_CLIPPING_VALUE, min_value=MIN_CLIPPING_VALUE)
        target = self.validation_processor.get_output_(transformed_inputs, control_voltages)
        error = ((target[mask] - y_predicted[mask]) ** 2).mean()
        print(f'MSE: {str(error)}')
        var_target = np.var(target[mask], ddof=1)
        print(f'(var) NMSE: {100*error/var_target} %')
        plot_gate_validation(target[mask], y_predicted[mask], str(gate), show_plots=self.show_plots, save_dir=self.get_plot_dir(gate, base_dir))
        print('==========================================================================================')
        return error


def single_gate(configs, gate, threshold, verbose=False, validate=False, control_voltages=None, best_output=None, vcdim=4):
    data_manager = VCDimDataManager(configs)
    configs['boolean_gate_test']['algorithm_configs']['processor']['shape'] = data_manager.get_shape(vcdim, validation=False)
    configs['boolean_gate_test']['validation']['processor']['shape'] = data_manager.get_shape(vcdim, validation=True)

    test = BooleanGateTask(configs['boolean_gate_test'])
    excel_results = None

    if control_voltages is None and best_output is None:
        _, transformed_inputs, readable_targets, transformed_targets, _, mask = data_manager.get_data(vcdim, verbose, validation=False)
        excel_results = test.find_gate(transformed_inputs, readable_targets[gate], transformed_targets[gate], mask, threshold)
        control_voltages = excel_results['control_voltages']
        best_output = excel_results['best_output']
    if validate:
        _, transformed_inputs, readable_targets, transformed_targets, _, mask = data_manager.get_data(vcdim, verbose, validation=True)
        validation_slopped_plato = data_manager.generate_slopped_plato(vcdim)
        control_voltages = validation_slopped_plato * control_voltages[:, np.newaxis]

        test.validate_gate(gate, transformed_inputs, control_voltages.T, best_output, mask, test.gate_base_dir)
    return excel_results


def plot_gate_validation(output, prediction, gate, show_plots, save_dir=None):
    plt.figure()
    plt.title(gate)
    plt.plot(output)
    plt.plot(prediction)
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')
    plt.legend(['Device output', 'NN prediction'])
    if save_dir is not None:
        plt.savefig(save_dir)
    if show_plots:
        plt.show()
    plt.close()


def find_single_gate(configs_path, gate):
    configs = load_configs(configs_path)
    configs = configs['capacity_test']['vc_dimension_test']

    # gate = '[0 1 1 0]'
    threshold = configs['boolean_gate_test']['algorithm_configs']['hyperparameters']['stop_threshold']  # 0.95

    result = single_gate(configs, gate, threshold, validate=False)

    # save('numpy', configs['boolean_gate_test']['results_base_dir'], 'control_voltages', overwrite=False, data=result['control_voltages'])
    # save('numpy', results_path, 'best_output', overwrite=False, data=result['best_output'], timestamp=False)

    # print(f"Control voltages: {result['control_voltages']}")
    return result


def validate_single_gate(configs_path, results_path):
    import os
    configs = load_configs(configs_path)
    configs = configs['capacity_test']['vc_dimension_test']

    cv_path = os.path.join(results_path, 'control_voltages.npz')
    bo_path = os.path.join(results_path, 'best_output.npz')
    cv = np.load(cv_path)['data']
    bo = np.load(bo_path)['data']

    # cv = np.array([-0.064, -0.858, -0.24, -0.41, 0.058])
    single_gate(configs, None, None, validate=True, control_voltages=cv, best_output=bo)


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    from bspyalgo.utils.io import save
    from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager

    results_path = find_single_gate('configs/benchmark_tests/capacity/template_ga_simulation.json', '[1 0 0 1]')
    # validate_single_gate('configs/benchmark_tests/capacity/template_ga_validation.json', results_path)
