from bspytasks.ring.tasks.searcher import search_solution
from brainspy.utils.io import create_directory_timestamp


def capacity_test(configs,
                  custom_model,
                  criterion,
                  algorithm,
                  transforms=None,
                  custom_logger=None,
                  average_plateaus=True,
                  dnpu_layer_no=1):
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
                        custom_logger=custom_logger,
                        average_plateaus=average_plateaus,
                        dnpu_layer_no=dnpu_layer_no)
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

    capacity_test(
        configs,
        DefaultCustomModel,
        criterion,
        algorithm,
        custom_logger=Logger,
        average_plateaus=
        False,  # Whether if to average plateaus or not. Set average_plateaus according to how you have declared in the constructor of the processor of your model.
        dnpu_layer_no=1
    )  # Specifies the number of DNPU layers that you are using. It is used to calculate the plateau length of the output. It is only used  in case you are not averaging plateaus.
