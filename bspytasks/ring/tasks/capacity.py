from bspytasks.ring.tasks.searcher import search_solution
from brainspy.utils.io import create_directory_timestamp


def capacity_test(configs, custom_model, criterion, algorithm, transforms=None):
    base_dir = create_directory_timestamp(configs["results_base_dir"], "capacity")
    gap = configs["start_gap"]
    while gap >= configs["stop_gap"]:
        print(f"********* GAP {gap} **********")
        configs["data"]["gap"] = gap
        configs["results_base_dir"] = base_dir
        search_solution(
            configs, DNPU, criterion, algorithm, transforms=transforms, is_main=False
        )
        gap = gap / 2
        print(f"*****************************")


if __name__ == "__main__":

    from torchvision import transforms

    from brainspy.utils import manager
    from brainspy.utils.io import load_configs
    from brainspy.utils.transforms import DataToTensor, DataToVoltageRange
    from brainspy.processors.dnpu import DNPU

    V_MIN = [-1.2, -1.2]
    V_MAX = [0.7, 0.7]

    transforms = transforms.Compose(
        [DataToVoltageRange(V_MIN, V_MAX, -1, 1), DataToTensor()]
    )

    configs = load_configs("configs/ring.yaml")

    criterion = manager.get_criterion(configs["algorithm"])
    algorithm = manager.get_algorithm(configs["algorithm"])

    capacity_test(configs, DNPU, criterion, algorithm, transforms=transforms)
