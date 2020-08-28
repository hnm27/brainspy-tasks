import pickle as p
import numpy as np
from brainspy.utils.input import map_to_voltage

# @todo: This data should come from the model
MAX_INPUT_VOLT = np.asarray([0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3])
MIN_INPUT_VOLT = np.asarray([-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7])


def adjust_results(new_input, results_name, output_dir):
    a = np.load(new_input)
    f = open(results_name, "rb")
    res = p.load(f)
    f.close()
    if a["inputs"].max() > MAX_INPUT_VOLT[0] or a["inputs"].min() < MIN_INPUT_VOLT[0]:
        for i in range(inputs.shape[1]):
            inputs[:, i] = map_to_voltage(
                inputs[:, i],
                MIN_INPUT_VOLT[processor_configs["input_indices"][i]],
                MAX_INPUT_VOLT[processor_configs["input_indices"][i]],
            )
    else:
        inputs = a["inputs"]
    res["inputs"] = inputs
    res["targets"] = a["targets"]

    f2 = open("processed.pickle", "wb")
    p.dump(res, f2)
    f2.close()
