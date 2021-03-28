import os
import torch

import numpy as np
import pickle as p
import matplotlib.pyplot as plt

from bspytasks.boolean.data import BooleanGateDataset


from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.manager import get_optimizer
from brainspy.utils.io import create_directory, create_directory_timestamp
from bspytasks.utils.io import save
from brainspy.algorithms.modules.performance.accuracy import (
    get_accuracy,
    plot_perceptron,
)
from brainspy.algorithms.modules.signal import pearsons_correlation


def boolean_task(
    configs,
    custom_model,
    criterion,
    algorithm,
    data_transforms=None,
    waveform_transforms=None,
    logger=None,
    is_main=True,
):
    main_dir, reproducibility_dir = init_dirs(
        str(configs["gate"]), configs["results_base_dir"], is_main
    )
    gate = np.array(configs["gate"])
    loader = get_data(gate, data_transforms, configs)
    if "track_running_stats" in configs["algorithm"]:
        configs["processor"]["track_running_stats"] = configs["algorithm"][
            "track_running_stats"
        ]
    print(
        "=========================================================================================="
    )
    print("GATE: " + str(gate))
    for i in range(1, configs["max_attempts"] + 1):
        print("ATTEMPT: " + str(i))

        model = custom_model(configs["processor"])
        optimizer = get_optimizer(model, configs["algorithm"])
        model, training_data = algorithm(
            model,
            (loader, None),
            criterion,
            optimizer,
            configs["algorithm"],
            waveform_transforms=waveform_transforms,
            logger=logger,
            save_dir=reproducibility_dir,
        )

        results = evaluate_model(
            model, loader.dataset, criterion, transforms=waveform_transforms
        )
        results["training_data"] = training_data
        results["threshold"] = configs["threshold"]
        results["gate"] = str(gate)
        results = postprocess(
            results,
            model,
            configs["accuracy"],
            logger=logger,
            save_dir=main_dir,
        )
        close(model, results, configs, reproducibility_dir)
        if results["veredict"]:
            break

    print(
        "=========================================================================================="
    )
    return results


def close(model, results, configs, save_dir):
    torch.save(results, os.path.join(save_dir, "results.pickle"))
    save("configs", os.path.join(save_dir, "configs.yaml"), data=configs)
    # Save the latest model
    if model.is_hardware():
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    else:
        model = torch.load(os.path.join(save_dir, "model.pt"))
    torch.save(
        results,
        os.path.join(save_dir, "results.pickle"),
        pickle_protocol=p.HIGHEST_PROTOCOL,
    )
    # Close the model adequately if it is on hardware
    if model.is_hardware() and "close" in dir(model):
        model.close()


def get_data(gate, data_transforms, configs):
    dataset = BooleanGateDataset(target=gate, transforms=data_transforms)
    if "batch_size" in configs["data"]:
        batch_size = configs["data"]["batch_size"]
    else:
        batch_size = len(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=configs["data"]["pin_memory"],
    )


def postprocess(results, model, node_configs, logger=None, node=None, save_dir=None):
    if (
        torch.isnan(results["predictions"]).any()
        or torch.isinf(results["predictions"]).any()
    ):
        print(
            "Nan values detected in the predictions. It is likely that the gradients of the model exploded. Skipping.."
        )
        results["veredict"] = False
        return results
    results["accuracy"] = get_accuracy(
        results["predictions"], results["targets"], node_configs, node
    )  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    results["correlation"] = pearsons_correlation(
        results["predictions"], results["targets"]
    )

    if (results["accuracy"]["accuracy_value"] / 100) >= results["threshold"]:
        results["veredict"] = True
    else:
        results["veredict"] = False
    results["summary"] = (
        "VC Dimension: "
        + str(len(results["targets"]))
        + " Gate: "
        + results["gate"]
        + " Veredict: "
        + str(results["veredict"])
        + "\n Accuracy (Simulation): "
        + str(results["accuracy"]["accuracy_value"])
        + "/"
        + str(results["threshold"])
    )

    # results["results_fig"] =
    plot_results(results, save_dir)
    # results["performance_fig"] =
    plot_performance(results, save_dir=save_dir)
    # results["accuracy_fig"] =
    plot_perceptron(results["accuracy"], save_dir)
    print(results["summary"])
    # if logger is not None:
    #     logger.log.add_figure(
    #         "Results/VCDim" + str(len(results["targets"])) + "/" + results["gate"],
    #         results["results_fig"],
    #     )
    #     logger.log.add_figure(
    #         "Accuracy/VCDim" + str(len(results["targets"])) + "/" + results["gate"],
    #         results["accuracy_fig"],
    #     )
    return results


def evaluate_model(model, dataset, criterion, results={}, transforms=None):
    with torch.no_grad():
        model.eval()
        if transforms is None:
            inputs, targets = dataset[:]
        else:
            inputs, targets = transforms(dataset[:])
        inputs = inputs.to(device=TorchUtils.get_device())
        targets = targets.to(device=TorchUtils.get_device())

        predictions = model(inputs)

    results["inputs"] = inputs
    results["targets"] = targets
    results["predictions"] = predictions
    results["performance"] = criterion(predictions, targets)
    return results


def init_dirs(gate_name, base_dir, is_main):
    if is_main:
        base_dir = create_directory_timestamp(base_dir, gate_name)
        reproducibility_dir = os.path.join(base_dir, "reproducibility")
        create_directory(reproducibility_dir)
    else:
        base_dir = os.path.join(base_dir, gate_name)
        reproducibility_dir = os.path.join(base_dir, "reproducibility")
        create_directory(reproducibility_dir)
    return base_dir, reproducibility_dir


def plot_results(results, save_dir=None, fig=None, show_plots=False, line="-"):
    if fig is None:
        fig = plt.figure()
    plt.title(results["summary"])
    plt.plot(
        results["predictions"].detach().cpu(), line, label="Prediction (Simulation)"
    )
    plt.plot(results["targets"].detach().cpu(), line, label="Target (Simulation)")
    plt.ylabel("Current (nA)")
    plt.xlabel("Time")
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "results.jpg"))
    if show_plots:
        plt.show()
    plt.close()
    return fig


def plot_performance(results, save_dir=None, fig=None, show_plots=False):
    if fig is None:
        plt.figure()
    plt.title(f"Learning profile", fontsize=12)
    for i in range(len(results["training_data"]["performance_history"])):
        plt.plot(
            TorchUtils.to_numpy(
                results["training_data"]["performance_history"][i]
            )
        )
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"training_profile"))
    plt.close()
    return fig


if __name__ == "__main__":
    import numpy as np
    import datetime as d
    from torchvision import transforms

    from brainspy.utils import manager
    from bspytasks.boolean.logger import Logger
    from brainspy.utils.io import load_configs
    from brainspy.utils.transforms import (
        DataToTensor,
        DataToVoltageRange,
        DataPointsToPlateau,
    )
    from brainspy.processors.dnpu import DNPU

    V_MIN = [-1.2, -1.2]
    V_MAX = [0.6, 0.6]

    configs = load_configs("configs/boolean.yaml")

    data_transforms = transforms.Compose(
        [
            DataToVoltageRange(V_MIN, V_MAX, -1, 1),
            DataToTensor(device=torch.device("cpu")),
        ]
    )

    waveform_transforms = transforms.Compose(
        [DataPointsToPlateau(configs["processor"]["data"]["waveform"])]
    )

    logger = Logger(f"tmp/output/logs/experiment" + str(d.datetime.now().timestamp()))

    configs["gate"] = [0, 0, 0, 1]
    configs["threshold"] = 0.8

    criterion = manager.get_criterion(configs["algorithm"])
    algorithm = manager.get_algorithm(configs["algorithm"])

    boolean_task(
        configs,
        DNPU,
        criterion,
        algorithm,
        data_transforms=data_transforms,
        waveform_transforms=waveform_transforms,
        logger=logger,
    )
