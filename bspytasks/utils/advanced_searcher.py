# TODO: Implement with the new libraries
# import torch
# import matplotlib.pyplot as plt

# from brainspy.utils.manager import get_processor
# from bspytasks.tasks.ring.classifier import RingClassificationTask as Task
# from brainspy.utils.pytorch import TorchUtils


# class AdvancedRingSearcher():
#     def __init__(self, configs):
#         self.configs = configs
#         self.processor = get_processor(configs['algorithm_configs']['processor'])
#         self.task = Task(configs)

#     def improve_solution(self, results, model):
#         self.processor.load_state_dict(model.copy())
#         inputs = TorchUtils.format(results['inputs'])
#         targets = TorchUtils.format(results['targets'])
#         TorchUtils.init_seed(results['seed'], deterministic=True)
#         new_results = self.task.run_task(inputs, targets, results['mask'], save_data=True)

#         plt.figure()
#         plt.plot(results['best_output'], label='old_output')
#         plt.plot(new_results['best_output'], label='new_output')
#         plt.legend()
#         plt.show()


# if __name__ == '__main__':
#     import pickle
#     import torch
#     import matplotlib.pyplot as plt
#     from brainspy.utils.io import load_configs
#     from bspytasks.utils.datasets import load_data

#     folder_name = 'searcher_0.2mV_2020_02_26_231845'
#     base_dir = 'tmp/output/ring/' + folder_name
#     model, results, configs = load_data(base_dir)
#     searcher = AdvancedRingSearcher(configs)
#     searcher.improve_solution(results, model)
