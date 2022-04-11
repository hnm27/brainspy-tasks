import os
import torch
import matplotlib.pyplot as plt

from brainspy.utils.io import create_directory, create_directory_timestamp
from brainspy.algorithms.modules.performance.accuracy import get_accuracy
from brainspy.algorithms.modules.signal import pearsons_correlation
from brainspy.utils.pytorch import TorchUtils


def load_reproducibility_results(base_dir, model_name="model.pt"):
    base_dir = os.path.join(base_dir, "reproducibility")
    # configs = load_configs(os.path.join(gate_base_dir, 'configs.yaml'))
    model = torch.load(os.path.join(base_dir, model_name),
                       map_location=TorchUtils.get_device())
    results = torch.load(os.path.join(base_dir, "results.pickle"),
                         map_location=TorchUtils.get_device())
    return model, results  # , configs


def validate(
    model,
    results,
    configs,
    criterion,
    results_dir,
    transforms=None,
    show_plots=False,
    is_main=True,
):

    # results_dir = init_dirs(results_dir, is_main=is_main, gate=results['gap'])
    if "train_results" in results:
        results["train_results"] = apply_transforms(results["train_results"],
                                                    transforms=transforms)
        results["train_results_hw"] = _validate(
            model, results["train_results"].copy(), criterion, configs)
    if "dev_results" in results:
        results["dev_results"] = apply_transforms(results["dev_results"],
                                                  transforms=transforms)
        results["dev_results_hw"] = _validate(model,
                                              results["dev_results"].copy(),
                                              criterion, configs)
    if "test_results" in results:
        results["test_results"] = apply_transforms(results["test_results"],
                                                   transforms=transforms)
        results["test_results_hw"] = _validate(model,
                                               results["test_results"].copy(),
                                               criterion, configs)
    if "train_results" in results:
        results["train_results"][
            "best_output_formatted"] = model.format_targets(
                results["train_results"]["best_output"])
    if "dev_results" in results:
        results["dev_results"]["best_output_formatted"] = model.format_targets(
            results["dev_results"]["best_output"])
    if "test_results" in results:
        results["test_results"][
            "best_output_formatted"] = model.format_targets(
                results["test_results"]["best_output"])

    plot_all(results, save_dir=results_dir, show_plots=show_plots)
    torch.save(results,
               os.path.join(results_dir, "hw_validation_results.pickle"))
    return results


def plot_all(results, save_dir=None, show_plots=False):
    if "train_results" in results:
        plot_validation_results(
            TorchUtils.to_numpy(
                results["train_results"]["best_output_formatted"]),
            TorchUtils.to_numpy(results["train_results_hw"]["best_output"]),
            name="train_plot",
            save_dir=save_dir,
            show_plot=show_plots,
        )
    if "dev_results" in results:
        plot_validation_results(
            TorchUtils.to_numpy(
                results["dev_results"]["best_output_formatted"]),
            TorchUtils.to_numpy(results["dev_results_hw"]["best_output"]),
            name="validation_plot",
            save_dir=save_dir,
            show_plot=show_plots,
        )
    if "test_results" in results:
        plot_validation_results(
            TorchUtils.to_numpy(
                results["test_results"]["best_output_formatted"]),
            TorchUtils.to_numpy(results["test_results_hw"]["best_output"]),
            name="test_plot",
            save_dir=save_dir,
            show_plot=show_plots,
        )


def plot_validation_results(
    model_output,
    real_output,
    save_dir=None,
    show_plot=False,
    name="validation_plot",
    extension="png",
):

    error = ((model_output - real_output)**2).mean()
    print(f"Total Error: {error}")

    plt.figure()
    plt.title(
        f"{name.capitalize()}\nComparison between Hardware and Simulation \n (MSE: {error})",
        fontsize=10)
    plt.plot(model_output)
    plt.plot(real_output, "-.")
    plt.ylabel("Current (nA)")
    plt.xlabel("Time")

    plt.legend(["Simulation", "Hardware"])
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + "." + extension))
        # np.savez(os.path.join(self.main_dir, name + '_data'),
        #         model_output=model_output, real_output=real_output, mask=mask)
    if show_plot:
        plt.show()
        plt.close()


def apply_transforms(results, transforms):
    if transforms is not None:
        results["inputs"] = transforms(results["inputs"])
        results["targets"] = transforms(results["targets"])
        results["best_output"] = transforms(results["best_output"])
    return results


def _validate(model, results, criterion, hw_processor_configs):
    with torch.no_grad():
        model.hw_eval(hw_processor_configs)
        predictions = model(results["inputs"])
        targets = model.format_targets(results['targets'])
        model.close()
        results["performance"] = criterion(predictions, targets)

    # results['gap'] = dataset.gap
    results["best_output"] = predictions
    print(
        f"Simulation accuracy {results['accuracy']['accuracy_value'].item()}: "
    )

    results['accuracy'] = get_accuracy(
        predictions,
        targets,
        configs=results['accuracy']['configs'],
        node=results['accuracy']['node']
    )  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    print(
        f"Hardware accuracy: {results['accuracy']['accuracy_value'].item()} \n"
    )
    results["correlation"] = pearsons_correlation(predictions, targets)
    return results


def init_dirs(base_dir, is_main=True, gate=""):
    if is_main:
        base_dir = create_directory_timestamp(base_dir, gate)
    else:
        base_dir = os.path.join(base_dir, gate)
        create_directory(base_dir)
    return base_dir


if __name__ == "__main__":
    from torchvision import transforms

    from brainspy.utils.io import load_configs
    from bspytasks.utils.transforms import PointsToPlateaus
    from brainspy.algorithms.modules.signal import fisher

    base_dir = "tmp/TEST/output/ring/ring_classification_gap_0.00625_2020_09_23_140014"
    model, results = load_reproducibility_results(base_dir)

    configs = load_configs("configs/ring.yaml")
    hw_processor_configs = load_configs("configs/defaults/processors/hw.yaml")
    waveform_transforms = transforms.Compose(
        [PointsToPlateaus(hw_processor_configs["data"]["waveform"])])

    results_dir = init_dirs(os.path.join(base_dir, "validation"))

    validate(
        model,
        results,
        hw_processor_configs,
        fisher,
        results_dir,
        transforms=waveform_transforms,
    )
