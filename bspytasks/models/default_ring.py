import torch

from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils


class DefaultCustomModel(torch.nn.Module):
    def __init__(self, configs):
        super(DefaultCustomModel, self).__init__()
        self.alpha = 1
        self.node_no = 1
        model_data = torch.load(configs['model_dir'],
                                map_location=TorchUtils.get_device())
        processor = Processor(configs, model_data['info'],
                              model_data['model_state_dict'])
        self.dnpu = DNPU(processor=processor,
                         data_input_indices=[configs['input_indices']] *
                         self.node_no,
                         forward_pass_type='vec')
        self.linear = torch.nn.Linear(3, 1)
        self.dnpu.add_input_transform([-1, 1])

        # self.dnpu2 = DNPU(processor=processor,
        #                   data_input_indices=[configs['input_indices']] *
        #                   self.node_no)
        # self.dnpu2.add_input_transform([-110, 100])

    def forward(self, x):
        #x = torch.cat((x, x, x), dim=1)
        x = self.dnpu(x)
        # x = self.linear(x)
        #x = self.dnpu2(x)

        return x

    def hw_eval(self, configs, info=None):
        self.eval()
        self.dnpu.hw_eval(configs, info)
        #self.dnpu2.hw_eval(configs, info)

    def get_input_ranges(self):
        return self.dnpu.get_input_ranges()

    def get_control_ranges(self):
        return self.dnpu.get_control_ranges()

    def get_control_voltages(self):
        return self.dnpu.get_control_voltages()

    def set_control_voltages(self, control_voltages):
        self.dnpu.set_control_voltages(control_voltages)

    def get_clipping_value(self):
        return self.dnpu.get_clipping_value()
        # return clipping_value

    def is_hardware(self):
        return self.dnpu.processor.is_hardware

    def close(self):
        self.dnpu.close()

    def regularizer(self):
        return self.alpha * (self.dnpu.regularizer())  # +
        #self.dnpu2.regularizer())

    def constraint_control_voltages(self):
        pass
        # self.dnpu.constraint_control_voltages()

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnpu.format_targets(x)
