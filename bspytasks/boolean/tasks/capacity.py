import os
import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt

from bspytasks.boolean.tasks.vcdimension import vc_dimension_test
from brainspy.utils.io import create_directory_timestamp


from brainspy.utils.pytorch import TorchUtils


def capacity_test(
    configs,
    custom_model,
    criterion,
    algorithm,
    data_transforms=None,
    logger=None,
):
    print(
        "*****************************************************************************************"
    )
    print(
        f"CAPACITY TEST FROM VCDIM {configs['from_dimension']} TO VCDIM {configs['to_dimension']} "
    )
    print(
        "*****************************************************************************************"
    )
    base_dir = create_directory_timestamp(configs["results_base_dir"], "capacity_test")

    # save(mode='configs', file_path=self.configs_dir, data=configs)
    summary_results = {
        "capacity_per_N": [],
        "accuracy_distrib_per_N": [],
        "performance_distrib_per_N": [],
        "correlation_distrib_per_N": [],
    }
    for i in range(configs["from_dimension"], configs["to_dimension"] + 1):
        # capacity, accuracy_array, performance_array, correlation_array = vc_dimension_test(self.current_dimension, validate=validate)
        configs["results_base_dir"] = base_dir
        configs["current_dimension"] = i
        results = vc_dimension_test(
            configs,
            custom_model,
            criterion,
            algorithm,
            data_transforms=data_transforms,
            logger=logger,
            is_main=False,
        )
        summary_results["capacity_per_N"].append(
            TorchUtils.to_numpy(results["capacity"])
        )
        summary_results["accuracy_distrib_per_N"].append(
            TorchUtils.to_numpy(results["accuracies"])
        )
        summary_results["performance_distrib_per_N"].append(
            TorchUtils.to_numpy(results["performances"][:, -1])
        )
        summary_results["correlation_distrib_per_N"].append(
            TorchUtils.to_numpy(results["correlations"])
        )
        del results
    # self.vcdimension_test.close_results_file()
    # self.plot_summary()
    # dict_loc = os.path.join(self.configs['vc_dimension_test']['results_base_dir'], 'summary_results.pkl')
    with open(os.path.join(base_dir, "summary_results.pickle"), "wb") as fp:
        pickle.dump(summary_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # torch.save(summary_results, os.path.join(base_dir, 'summary_results.pickle'))
    plot_summary(
        summary_results, configs["from_dimension"], configs["to_dimension"], base_dir
    )
    print(
        "*****************************************************************************************"
    )


def plot_summary(results, from_dimension, to_dimension, base_dir=None):
    dimensions = np.arange(from_dimension, to_dimension + 1)
    plt.figure()
    plt.plot(dimensions, results["capacity_per_N"])
    plt.title("Capacity over N points")
    plt.xlabel("Nr. of points N")
    plt.ylabel("Capacity")
    if base_dir:
        plt.savefig(os.path.join(base_dir, "Capacity_over_N"))

    plot_boxplot(
        dimensions,
        results,
        "accuracy_distrib_per_N",
        title="Accuracy over N points",
        base_dir=base_dir,
    )
    plot_boxplot(
        dimensions,
        results,
        "performance_distrib_per_N",
        title="Performance over N points",
        base_dir=base_dir,
    )
    plot_boxplot(
        dimensions,
        results,
        "correlation_distrib_per_N",
        title="Correlation over N points",
        base_dir=base_dir,
    )

    plt.show()


def plot_boxplot(pos, results, key, title="", base_dir=None):
    plt.figure()
    plt.title(title)
    plt.boxplot(results[key], positions=pos)
    plt.xlabel("Nr. of points N")
    plt.ylabel(key.split("_")[0])
    if base_dir:
        plt.savefig(os.path.join(base_dir, key))


if __name__ == "__main__":
    import datetime as d
    from torchvision import transforms
    from brainspy.utils import manager

    from bspytasks.boolean.logger import Logger

    from brainspy.utils.io import load_configs
    from bspytasks.utils.transforms import DataToVoltageRange, DataToTensor
    from bspytasks.models.default import DefaultCustomModel

    configs = load_configs("configs/boolean.yaml")
    data_transforms = transforms.Compose(
        [DataToTensor(device=torch.device('cpu'))]
    )

    criterion = manager.get_criterion(configs["algorithm"])
    algorithm = manager.get_algorithm(configs["algorithm"])

    logger = Logger(f"tmp/output/logs/experiment" + str(d.datetime.now().timestamp()))
    capacity_test(
        configs,
        DefaultCustomModel,
        criterion,
        algorithm,
        data_transforms=data_transforms,
        logger=logger,
    )
