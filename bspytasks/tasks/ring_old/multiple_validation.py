import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from bspyproc.utils.waveform import generate_waveform
from bspyproc.processors.processor_mgr import get_processor


def get_final_result(main_dir, save=None, accuracy=False, architecture=True):
    mask = None
    dirs = list([name for name in os.listdir(main_dir) if os.path.isdir(
        os.path.join(main_dir, name)) and not name.startswith('.')])

    shape_main = 20020
    dirnames = generate_plot_names(2)
    if architecture:
        internal_models = np.zeros([len(dirnames) + 1, len(dirs), shape_main])
        internal_real = np.zeros([len(dirnames) + 1, len(dirs), shape_main])
    if accuracy:
        model_accuracies = np.zeros([len(dirs)])
        real_accuracies = np.zeros([len(dirs)])
    model_outputs = np.zeros([len(dirs), shape_main])
    real_outputs = np.zeros([len(dirs), shape_main])

    for i in range(len(dirs)):
        if accuracy:
            accuracy_data = np.load(os.path.join(main_dir, dirs[i], 'accuracy_data.npz'))
            model_accuracies[i] = accuracy_data['model_accuracy'].item()
            real_accuracies[i] = accuracy_data['hardware_accuracy'].item()
        if architecture:
            for j in range(len(dirnames)):
                # internal_models[j][i] = generate_waveform(torch.load(os.path.join(main_dir,
                #                                                                   dirs[i], 'debug', 'simulation', dirnames[j]+'.pt')).cpu().detach().numpy(), 80, 20)
                internal_models[j][i] = torch.load(os.path.join(main_dir,
                                                                dirs[i], 'debug', 'simulation', dirnames[j] + '.pt')).cpu().detach().numpy()
                internal_real[j][i] = np.load(os.path.join(main_dir,
                                                           dirs[i], 'debug', 'hardware', dirnames[j] + '.npy'))

        data = np.load(os.path.join(
            main_dir, dirs[i], 'validation_plot_data.npz'))
        # The last [:,0] of the next line can be removed
        if architecture:
            internal_models[len(dirnames)][i] = data['model_output'][:, 0]
            #######################
            internal_real[len(dirnames)][i] = data['real_output']
        model_outputs[i] = data['model_output'][:, 0]
        real_outputs[i] = data['real_output'][:, 0]

        mask = data['mask']

    if accuracy:
        print('Mean Model Accuracy: ' + str(model_accuracies.mean()))
        print('Mean Hardware Accuracy: ' + str(real_accuracies.mean()))
        np.savez(os.path.join(main_dir, 'accuracies'), model_accuracies=model_accuracies, real_accuracies=real_accuracies)

    np.savez(os.path.join(main_dir, 'outputs'), model_outputs=model_outputs.T, hardware_outputs=real_outputs.T, mask=mask)

    if architecture:
        dirnames.append('output')
        for j in range(len(dirnames)):
            aux1 = internal_models[j].T[mask]
            aux2 = internal_real[j].T[mask]
            plot(aux1.T, aux2.T,
                 str(j) + '_' + dirnames[j], main_dir)
            if dirnames[j] == 'output':
                all_outputs_real = internal_real[j].T[mask]

    # plot_mean_std(internal_models[len(dirnames)]
    #               [0], internal_real[len(dirnames)], 'output')

    print('test')

    # if save:
    #     np.savez(main_dir, performance=performance_results,
    #              accuracy=accuracy_results, gap=gap_results)
    # return performance_results, accuracy_results, gap_results


# def plot_mean_std(model, data, name):
#     data_mean = np.mean(data, axis=0)
#     data_std = np.std(data, axis=0)
#     plt.title(name)
#     plt.plot(model, label='model')
#     plt.plot(data_mean, 'k', color='b',  label='mean')
#     plt.plot(data_mean + data_std, ':k', label='std + ')
#     plt.plot(data_mean - data_std, ':k', label='std -')
#     plt.legend()
#     plt.show()

def plot_single(data, color, name):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    # plt.title(name)
    plt.plot(data_mean, color=color[0], label='mean_' + name)
    plt.plot(data_mean + data_std, ':', color=color[1], label='std_' + name)
    plt.plot(data_mean - data_std, ':', color=color[1])
    # plt.legend()
    # plt.show()


def plot(model_data, real_data, name, save=None):
    my_dpi = 300
    extension = 'png'
    plt.figure(figsize=(19.2, 10.8))

    errors = calculate_errors(model_data, real_data)
    plt.title(name + f'\n MSE: {errors}')
    plot_single(model_data, ['b', 'c'], 'model')
    plot_single(real_data, ['g', 'y'], 'hardware')
    plt.legend(loc='upper right')
    if save:
        plt.savefig(os.path.join(save, name + '.' + extension),
                    bbox_inches='tight', format=extension, dpi=my_dpi)
    # plt.show()


def calculate_errors(model_data, real_data):
    model_mean = np.mean(model_data, axis=0)
    real_mean = np.mean(real_data, axis=0)
    return ((model_mean - real_mean) ** 2).mean()


def generate_plot_names(layer_no):
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

    return result


if __name__ == "__main__":
    from bspyalgo.utils.io import load_configs
    import matplotlib
    matplotlib.use('TkAgg')
    main_dir = 'tmp/output/ring_nips/searcher_0.00625mV_2020_04_16_183324_single_newtrial/validation.old'

    get_final_result(main_dir, accuracy=True, architecture=False)
    # list = []
    # j = 1
    # for i in range(18020):

    #     if j == 41:
    #         list.append(True)
    #     else:
    #         list.append(False)
    #     if j == 100:
    #         j = 0
    #     j += 1
    # a = np.array(list)
    # np.save('mask', np.array(list))
