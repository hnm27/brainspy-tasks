import torch
from torch import nn
from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU
from brainspy.processors.modules.bn import DNPUBatchNorm
from brainspy.utils.pytorch import TorchUtils
#from brainspy.utils.transforms import CurrentToVoltage


class Architecture21(nn.Module):
    def __init__(self, configs, info=None, state_dict=None):
        super().__init__()
        self.alpha = 1  # configs['regul_factor']
        if info is None:
            training_data = torch.load(
                'C:/Users/CasGr/Documents/Data/training_data_quick.pt',
                map_location=torch.device('cpu'))
            self.processor = Processor(
                configs,
                info=training_data['info'],
                model_state_dict=training_data['model_state_dict'])

        self.l1_nodes = 2
        self.l2_nodes = 1
        self.l1_input_list = [[0, 4]] * self.l1_nodes
        self.l2_input_list = [[0, 4]] * self.l2_nodes

        self.dnpu_l1 = DNPUBatchNorm(self.processor,
                                     data_input_indices=self.l1_input_list)
        self.dnpu_l1.add_input_transform(input_range=[-1, 1], strict=False)

        self.dnpu_out = DNPU(self.processor,
                             data_input_indices=self.l2_input_list)
        self.dnpu_out.add_input_transform(input_range=[0, 1])
        #self.dnpu_out.init_batch_norm(track_running_stats=False)

    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = self.dnpu_l1(x)
        # output = self.dnpu_l1.get_logged_variables()
        x = torch.sigmoid(x)
        x = self.dnpu_out(x)
        return x

    def format_targets(self, x):
        return self.dnpu_l1.format_targets(x)

    def hw_eval(self, configs, info=None):
        self.eval()
        self.processor.load_processor(configs, info)

    def set_running_mean(self, running_mean):
        self.dnpu_l1.bn.running_mean = running_mean

    def set_running_var(self, running_var):
        self.dnpu_l1.bn.running_var = running_var

    def get_running_mean(self):
        return self.dnpu_l1.bn.running_mean

    def get_input_ranges(self):
        # Necessary to implement for the automatic data input to voltage conversion
        pass

    def get_logged_variables(self):
        log = {}
        dnpu_l1_logs = self.dnpu_l1.get_logged_variables()
        for key in dnpu_l1_logs.keys():
            log['l1_' + key] = dnpu_l1_logs[key]

        # dnpu_l2_logs = self.dnpu_l2.get_logged_variables()
        # for key in dnpu_l2_logs.keys():
        #     log['l2_' + key] = dnpu_l2_logs[key]
        log['l3_output'] = self.output
        log['a'] = self.a
        return log

    def get_control_ranges(self):
        # Necessary to use the GA data input to voltage conversion
        control_ranges = self.dnpu_l1.get_control_ranges()
        control_ranges = torch.cat(
            (control_ranges, self.dnpu_out.get_control_ranges()))
        return control_ranges

    def get_control_voltages(self):
        control_voltages = self.dnpu_l1.get_control_voltages()
        control_voltages = torch.cat(
            (control_voltages, self.dnpu_out.get_control_voltages()))
        return control_voltages

    def set_control_voltages(self, control_voltages):
        control_voltages = control_voltages.view(3, 5)
        # Necessary to use the GA data input to voltage conversion
        self.dnpu_l1.set_control_voltages(control_voltages[0:2])
        self.dnpu_out.set_control_voltages(control_voltages[2].view(1, 5))

    def get_clipping_value(self):
        return self.processor.get_clipping_value()
        # return clipping_value

    def is_hardware(self):
        return self.processor.is_hardware

    def close(self):
        self.processor.close()

    def regularizer(self):
        return self.alpha * (self.dnpu_l1.regularizer() +
                             self.dnpu_out.regularizer())

    def constraint_weights(self):
        self.dnpu_l1.constraint_control_voltages()
        self.dnpu_out.constraint_control_voltages()


if __name__ == "__init__":

    import os
    import torch
    from brainspy.utils.io import load_configs
    from bspytasks.models.default_ring import DefaultCustomModel
    #from bspytasks.models.Architecture21 import Architecture21
    from brainspy.utils.performance.accuracy import get_accuracy
    from brainspy.utils.manager import get_criterion
    from bspytasks.ring.tasks.classifier import plot_perceptron
    import matplotlib.pyplot as plt
    import yaml
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    print('hello')
    # path to reproducibility fileC:\Users\CasGr\Documents\github\brainspy-tasks\tmp\ring\searcher_0.1gap_2022_02_23_150203_21\reproducibility
    path = r'/home/unai/Documents/3-Programming/bspy/tasks/tmp/ring/ring_classification_gap_0.5_2022_03_17_144531/reproducibility'

    # path to file where plots should be saved
    #plot_dir = 'C:/Users/CasGr/Documents/github/plots'

    # loading in necessary files
    loss_fn = get_criterion("fisher")
    configs = load_configs(os.path.join(path, "configs.yaml"))
    results = torch.load(os.path.join(path, "results.pickle"))

    # model_state_dict = torch.load(os.path.join(path, "model.pt"))
    training_data = torch.load(os.path.join(path, "training_data.pickle"))

    new_model_instance = Architecture21(configs['processor'])
    model = 'Arch21'

    new_model_instance.load_state_dict(training_data['model_state_dict'])