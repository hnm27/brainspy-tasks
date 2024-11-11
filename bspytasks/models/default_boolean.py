import torch

from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils 

class DefaultCustomSimulationModel(DNPU):
    
    def __init__(self, configs):
        # For this simple example, we just need a simple instance of a DNPU, but where input indices are defined 
        # already in the configs. The input indices are the electrodes that will be receiving the two dimensional
        # data for the boolean gates task. 
        
        # In order to load a surrogate model, the data can be extracted from the training_data.pt
        # generated during the training with the smg. 
        
        model_data = torch.load(configs['model_dir'],
                                map_location=TorchUtils.get_device())
        
        # This data contains the info dictionary, required to know, among other things,
        # the structure used in the neural network for training the device 
        # (In this example 5 layers of 90 nodes each, with ReLU as activation function).
        # Additionally, this file contains the model_state_dict, which
        # contains the weight values for the trained neural network simulating the DNPU.
        
        # The following line, is very similar to that used for initialising the hardware in notebook
        # number 2. But it now contains the info dictionary and the model_state_dict keys.
        super(DefaultCustomSimulationModel, self).__init__(Processor(configs, model_data['info'], model_data['model_state_dict']), [configs['input_indices']])
        
        # Additonally, we know that the data that we will be receiving for our example will be in a range from -1 to 1.
        # brains-py supports automatic transformation of the inputs, to the voltage ranges of the selected input indices.
        # This is done with the following line:
        self.add_input_transform([-1, 1])

class DefaultCustomHardwareModel(DNPU):
    
    def __init__(self, configs):
        # For this simple example, we just need a simple instance of a DNPU, but where input indices are defined 
        # already in the configs. The input indices are the electrodes that will be receiving the two dimensional
        # data for the boolean gates task. 
        
        # Since the current model will just need a hardware driver, there is no need to
        # load any model and model info.    
        
        super(DefaultCustomHardwareModel, self).__init__(Processor(configs), data_input_indices=[configs['input_indices']])
        
        # Additonally, we know that the data that we will be receiving for our example will be in a range from -1 to 1.
        # brains-py supports automatic transformation of the inputs, to the voltage ranges of the selected input indices.
        # This is done with the following line:
        self.add_input_transform([-1, 1])