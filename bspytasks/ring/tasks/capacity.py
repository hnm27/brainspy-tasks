from bspytasks.ring.tasks.searcher import search_solution
from brainspy.utils.io import create_directory_timestamp


def capacity_test(configs,
                  custom_model,
                  criterion,
                  algorithm,
                  transforms=None,
                  custom_logger=None):
    base_dir = create_directory_timestamp(configs["results_dir"], "capacity")
    gap = configs["start_gap"]
    while gap >= configs["stop_gap"]:
        print(f"********* GAP {gap} **********")
        configs["data"]["gap"] = gap
        configs["results_dir"] = base_dir
        search_solution(configs,
                        custom_model,
                        criterion,
                        algorithm,
                        transforms=transforms,
                        is_main=False,
                        custom_logger=custom_logger)
        gap = gap / 2
        print(f"*****************************")


if __name__ == "__main__":
    import datetime as d

    from brainspy.utils import manager
    from brainspy.utils.io import load_configs
    from bspytasks.ring.logger import Logger
    from bspytasks.models.default_ring import DefaultCustomModel

    configs = load_configs("configs/ring.yaml")

    # logger = Logger(f"tmp/output/logs/experiment" +
    #                 str(d.datetime.now().timestamp()))

    criterion = manager.get_criterion(configs["algorithm"]['criterion'])
    algorithm = manager.get_algorithm(configs["algorithm"]['type'])

    capacity_test(configs,
                  DefaultCustomModel,
                  criterion,
                  algorithm,
                  custom_logger=Logger)
