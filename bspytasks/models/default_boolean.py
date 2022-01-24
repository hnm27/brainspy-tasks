from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor

class DefaultCustomModel(DNPU):
    def __init__(self, configs):
        super(DefaultCustomModel, self).__init__(Processor(configs), [configs['input_indices']])
        self.add_input_transform([-1, 1])
