from bspytasks.utils.io import create_directory, create_directory_timestamp


# TODO: Finish implementing the capacity test with the new libraries
# class RingClassifierCapacity():

#     def __init__(self, configs):
#         self.configs = configs
#         self.base_dir = configs['results_base_dir']
#         #self.searcher = RingSearcher(configs, is_main=False)

#     def init_dirs(self):
#         main_dir = f'capacity'
#         base_dir = create_directory_timestamp(self.base_dir, main_dir)
#         return base_dir

#     def run_task(self):
#         self.searcher.base_dir = self.init_dirs()
#         gap = self.configs['start_gap']
#         while gap >= self.configs['stop_gap']:
#             print(f'********* GAP {gap} **********')
#             self.searcher.search_solution(gap)
#             gap = gap / 2
#             print(f'*****************************')


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from brainspy.utils.io import load_configs
#     from bspytasks.tasks.ring.capacity import RingClassifierCapacity

#     configs = load_configs('configs/tasks/ring/template_gd_architecture_3.json')
#     task = RingClassifierCapacity(configs)
#     task.run_task()
