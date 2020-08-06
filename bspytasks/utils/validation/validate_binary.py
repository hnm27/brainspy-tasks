from bspytasks.validation.validator import Hardware_Validator
from bspyalgo.utils.io import load_configs
from bspyalgo.utils.performance import perceptron, corr_coeff
import numpy as np
import os
import copy
import pickle
import matplotlib.pyplot as plt


def plot_boxplot(data, pos, save_dir, title=''):
    plt.figure()
    plt.title(title)
    plt.boxplot(data, positions=pos)
    plt.xlabel('Nr. of points N')
    plt.ylabel(title.split(' ')[0])
    file_path = os.path.join(save_dir, '_'.join(title.split(' ')))
    plt.savefig(file_path)


def validate_outputs(configs):
    '''Validates several optupts given a single input data stream.
    This is useful for example to validate a single VC-dim experiment.
    '''

    validator = Hardware_Validator(configs)
    data_file = os.path.join(validator.validation_dir, configs['npz_file'] + '.npz')
    with np.load(data_file) as data:
        predictions = data['output_array']
        mask = data['mask']
        targets_array = data['targets_array'][:, mask, np.newaxis]
        inputs = data['inputs']
        control_voltages_array = data['control_voltages_per_gate']
        gate_array = data['gate_array']

    N = len(gate_array[0])
    acceptance_threshold = (1 - 0.5 / N) * 100
    correlation_array = np.zeros(len(gate_array))
    accuracy_array = np.zeros_like(correlation_array)
    threshold_array = np.zeros_like(correlation_array)
    found_array = np.zeros_like(correlation_array)
    mserror_array = np.zeros_like(correlation_array)
    measurement_array = np.zeros((len(gate_array), len(inputs[mask, 0])))
    labels_array = np.zeros_like(measurement_array)

    for n, cv in enumerate(control_voltages_array):
        name = ''
        for s in str(gate_array[n]).split('.'):
            name += s
        if len(np.unique(gate_array[n])) == 1:
            print(f"Ignore validation for {name}")
            found_array[n] = True
            accuracy_array[n] = control_voltages_array[n, 0]
            threshold_array[n] = control_voltages_array[n, 0]
            labels_array[n] = control_voltages_array[n, 0] * inputs[mask, 0]
            measurement_array[n] = control_voltages_array[n, 0] * inputs[mask, 0]
            continue
        else:
            mserror, _, measurement = validator.validate_prediction(name, inputs, cv, predictions[n], mask)
            plt_name = os.path.join(validator.validation_dir, name)
            accuracy, predicted_labels, threshold = perceptron(measurement, targets_array[n], plot=plt_name)
            correlation_array[n] = corr_coeff(measurement.T, targets_array[n].T)
            mserror_array[n] = mserror
            print(f"{name} has accuracy {accuracy:.2f} % and correlation {correlation_array[n]:.2f}")
            found_array[n] = accuracy > acceptance_threshold
            accuracy_array[n] = accuracy
            threshold_array[n] = threshold
            labels_array[n] = predicted_labels[:, 0]
            measurement_array[n] = measurement[:, 0]

    capacity = np.mean(found_array)
    print(f"Capacity for N={N}: {capacity}")

    file_name = os.path.join(validator.validation_dir, 'validated_data')
    np.savez(file_name,
             mserror_array=mserror_array,
             correlation_array=correlation_array,
             accuracy_array=accuracy_array,
             threshold_array=threshold_array,
             measurement_array=measurement_array,
             labels_array=labels_array,
             found_array=found_array, capacity=capacity)

    return capacity, correlation_array, accuracy_array, mserror_array


def validate_capacity(configs):
    '''Validate the capacity of a device found via surogate model.
    Assumes experiment directory is given and there is a directory in each VC-dim called validation with the data.
    '''

    validation_summary = {'capacity_per_N': [],
                          'accuracy_distib_per_N': [],
                          'mserror_distrib_per_N': [],
                          'validation_prediction_corr_per_N': []}
    configs_vc = copy.deepcopy(configs)
    for vc in configs["from_to_vcdim"]:
        vc_dir = f"vc_dimension_{vc}"
        print("=" * 30 + f"  Validating {vc_dir}  " + "=" * 30)
        configs_vc["data_dir"] = os.path.join(configs["data_dir"], vc_dir, "validation")
        capacity, correlation_array, accuracy_array, mserror_array = validate_outputs(configs_vc)
        validation_summary['capacity_per_N'].append(capacity)
        validation_summary['accuracy_distib_per_N'].append(accuracy_array[1:-1])
        validation_summary['mserror_distrib_per_N'].append(mserror_array[1:-1])
        validation_summary['validation_prediction_corr_per_N'].append(correlation_array[1:-1])

    dict_loc = os.path.join(configs['data_dir'], 'validation_summary.pkl')
    with open(dict_loc, 'wb') as fp:
        pickle.dump(validation_summary, fp, protocol=pickle.HIGHEST_PROTOCOL)

    dimensions = np.arange(configs["from_to_vcdim"][0], configs["from_to_vcdim"][-1] + 1)

    plt.figure()
    plt.plot(dimensions, validation_summary['capacity_per_N'])
    plt.title('Validated capacity over N points')
    plt.xlabel('Nr. of points N')
    plt.ylabel('Capacity')
    file_path = os.path.join(configs['data_dir'], "Capacity_Validated")
    plt.savefig(file_path)

    plot_boxplot(validation_summary['accuracy_distib_per_N'], dimensions, configs['data_dir'], title='Validated accuracy')
    plot_boxplot(validation_summary['mserror_distrib_per_N'], dimensions, configs['data_dir'], title='MSE over N points')
    plot_boxplot(validation_summary['validation_prediction_corr_per_N'], dimensions, configs['data_dir'], title='Correlation prediction and measurement')

    plt.show()


if __name__ == "__main__":

    # configs = load_configs('configs/benchmark_tests/validation/validate_vcdim.json')
    # capacity, correlation_array, accuracy_array, mserror_array = validate_outputs(configs)

    configs = load_configs('configs/benchmark_tests/validation/validate_capacity.json')
    validate_capacity(configs)
