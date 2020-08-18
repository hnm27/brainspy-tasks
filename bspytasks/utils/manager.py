import torch
import bspyalgo.algorithms.loss as loss
import bspyalgo.algorithms.optim as bspyoptim
from bspyalgo.algorithms.ga import train as train_ga
from bspyalgo.algorithms.gd import train as train_gd
import torch_optimizer as torchoptim

from bspyproc.utils.pytorch import TorchUtils


def get_criterion(criterion_name):
    '''Gets the fitness function used in GA from the module FitnessFunctions
    The fitness functions must take two arguments, the outputs of the black-box and the target
    and must return a numpy array of scores of size len(outputs).
    '''
    if criterion_name == 'corr_fit':
        return loss.corr_fit
    elif criterion_name == 'accuracy_fit':
        return loss.accuracy_fit
    elif criterion_name == 'corrsig_fit':
        return loss.corrsig_fit
    elif criterion_name == 'fisher':
        return loss.fisher
    elif criterion_name == 'corrsig':
        return loss.corrsig
    elif criterion_name == 'sqrt_corrsig':
        return loss.sqrt_corrsig
    elif criterion_name == 'fisher_added_corr':
        return loss.fisher_added_corr
    elif criterion_name == 'fisher_multipled_corr':
        return loss.fisher_multipled_corr
    elif criterion_name == 'bce':
        bce = torch.nn.BCELoss()
        bce.cuda(TorchUtils.get_accelerator_type()).to(TorchUtils.data_type)
        return bce

    else:
        raise NotImplementedError(f"Criterion {criterion_name} is not recognised.")


def get_optimizer(model, configs):
    if configs['optimizer'] == 'genetic':
        # TODO: get gene ranges from model
        return bspyoptim.GeneticOptimizer(model.get_control_ranges(), configs['partition'], configs['mutation_rate'], configs['epochs'])
    elif configs['optimizer'] == 'elm':
        print('ELM optimizer not implemented yet')
        # return get_adam(parameters, configs)
    elif configs['optimizer'] == 'yogi':
        return get_yogi(model, configs)
    elif configs['optimizer'] == 'adam':
        return get_adam(model, configs)
    else:
        assert False, "Optimiser name {configs['optimizer']} not recognised. Please try"


def get_yogi(model, configs):
    print('Prediction using YOGI optimizer')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if "betas" in configs.keys():
        print("Set betas to values from the config file: ")
        print(*configs["betas"], sep=", ")
        return torchoptim.Yogi(parameters,
                               lr=configs['learning_rate'],
                               betas=configs["betas"]
                               )

    else:
        return torchoptim.Yogi(parameters,
                               lr=configs['learning_rate'])


def get_adam(model, configs):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Prediction using ADAM optimizer')
    if "betas" in configs.keys():
        print("Set betas to values from the config file: ")
        print(*configs["betas"], sep=", ")
        return torch.optim.Adam(parameters,
                                lr=configs['learning_rate'],
                                betas=configs["betas"]
                                )

    else:
        return torch.optim.Adam(parameters,
                                lr=configs['learning_rate'])


def get_algorithm(configs):
    if configs['type'] == 'gradient':
        return train_gd
    elif configs['type'] == 'genetic':
        return train_ga
    else:
        assert False, 'Unrecognised algorithm field in configs. It must have the value gradient or the value genetic.'
