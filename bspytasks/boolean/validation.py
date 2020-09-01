import os
import torch
import matplotlib.pyplot as plt

from brainspy.utils.io import load_configs
from bspytasks.boolean.tasks.classifier import postprocess
from bspytasks.boolean.tasks.classifier import plot_results
from brainspy.algorithms.modules.performance.accuracy import plot_perceptron
from brainspy.utils.io import create_directory, create_directory_timestamp

# TODO: Add possibility to validate multiple times


def validate_gate(model, results, configs, criterion, results_dir=None, transforms=None, show_plots=False, is_main=True):
    results = process_results(results, transforms=transforms)
    with torch.no_grad():
        model.hw_eval(configs)
        predictions = model(results["inputs"])

    results["hw_validation"] = postprocess(results, model, results['accuracy']['configs'], node=results['accuracy']['node'],
                                           save_dir=None
                                           )
    results["hw_validation"]["predictions"] = predictions
    results['hw_validation']['performance'] = criterion(predictions, results['targets'])
    results['hw_validation']["accuracy_fig"] = plot_perceptron(results["accuracy"], results_dir, name='hardware')
    results["summary"] = (
        results["summary"]
        + "\n Accuracy (Hardware): "
        + str(results["hw_validation"]["accuracy"]["accuracy_value"].item())
        + "/"
        + str(results["hw_validation"]["threshold"])
    )
    plot_validation_results(results, save_dir=results_dir)
    torch.save(results, os.path.join(results_dir, "hw_validation_results.pickle"))


def validate_vcdim(vcdim_base_dir, model_name="model.pt", is_main=True):
    base_dir = init_dirs(os.path.join(vcdim_base_dir, "validation"))
    dirs = [
        os.path.join(vcdim_base_dir, o)
        for o in os.listdir(vcdim_base_dir)
        if os.path.isdir(os.path.join(vcdim_base_dir, o))
    ]
    for d in dirs:
        if os.path.split(d)[1] != "validation":
            model = torch.load(os.path.join(d, 'reproducibility', 'model.pt'), map_location=torch.device(TorchUtils.get_accelerator_type()))
            results = torch.load(os.path.join(d, 'reproducibility', "results.pickle"), map_location=torch.device(TorchUtils.get_accelerator_type()))

            results_dir = init_dirs(d, is_main=True)

            criterion = manager.get_criterion(configs["algorithm"])

            # validate_gate(os.path.join(d, "reproducibility"), base_dir, is_main=False)
            validate_gate(
                model, results, configs['validation_processor'], criterion, results_dir=results_dir, transforms=waveform_transforms
            )


def process_results(results, transforms=None):
    if transforms is not None:
        results["inputs"] = transforms(results["inputs"])
        results["targets"] = transforms(results["targets"])
        results['predictions'] = transforms(results['predictions'])
    return results


def plot_validation_results(results, save_dir):
    fig = plt.figure()
    error = ((results['predictions'] - results["hw_validation"]["predictions"]) ** 2).mean()
    print(f"\n MSE: {error}")
    results['summary'] = results['summary'] + f"\n MSE: {error}"
    plt.plot(
        results["hw_validation"]["predictions"].detach().cpu(),
        label="Prediction (Hardware)",
    )
    plt.plot(
        results["hw_validation"]["targets"].detach().cpu(), label="Target (Hardware)"
    )
    plot_results(results, fig=fig, save_dir=save_dir)


def init_dirs(base_dir, is_main=True):
    name = 'validation'
    base_dir = os.path.join(base_dir, 'validation')
    if is_main:
        base_dir = create_directory_timestamp(base_dir, name)
    else:
        base_dir = os.path.join(base_dir, name)
        create_directory(base_dir)
    return base_dir


if __name__ == "__main__":
    from torchvision import transforms

    from brainspy.utils.io import load_configs
    from brainspy.utils.transforms import PointsToPlateaus, PlateausToPoints
    from brainspy.utils import manager
    from brainspy.utils.pytorch import TorchUtils

    configs = load_configs("configs/ring.yaml")

    base_dir = "tmp/TEST/output/boolean/[0, 0, 0, 1]_2020_09_01_115645"

    model = torch.load(os.path.join(base_dir, 'reproducibility', 'model.pt'), map_location=torch.device(TorchUtils.get_accelerator_type()))
    results = torch.load(os.path.join(base_dir, 'reproducibility', "results.pickle"), map_location=torch.device(TorchUtils.get_accelerator_type()))
    experiment_configs = load_configs(os.path.join(base_dir, 'reproducibility', 'configs.yaml'))

    waveform_transforms = transforms.Compose(
        [PlateausToPoints(experiment_configs['processor']['waveform']),  # Required to remove plateaus from training because the perceptron cannot accept less than 10 values for each gate
         PointsToPlateaus(configs["validation_processor"]["waveform"])]
    )

    results_dir = init_dirs(base_dir, is_main=True)

    criterion = manager.get_criterion(configs["algorithm"])

    # validate_gate(
    #     model, results, configs['validation_processor'], criterion, results_dir=results_dir, transforms=waveform_transforms
    # )
    validate_vcdim('tmp/TEST/output/boolean/vc_dimension_4_2020_09_01_145233')
