from csv import writer
import os.path as osp
import numpy as np

from models.segmenter import set_args, train
import random as r


def evaluate_segmentation_parameter(parameters, parameter_name, iterations=3, epochs=3):
    """
    Calls segmentation training a predetermined amount of times to evaluate a certain parameter
    """
    for parameter in parameters:
        all_scores = []
        args = set_args()
        setattr(args, 'epochs', epochs)
        setattr(args, parameter_name, parameter)
        for iteration in range(iterations):
            _, score = train(args)
            all_scores.append(score)

        mean_score = np.mean(all_scores)

        random_identifier = r.randint(10000, 99999)

        with open('tuning_results_identifiers', 'a+', newline='') as f:
            append_writer = writer(f)
            append_writer.writerow([mean_score, random_identifier])

        with open(osp.join('tuning_results', str(random_identifier)), 'x', newline='') as f:
            append_writer = writer(f)
            append_writer.writerow([f"{getattr(args, arg)}" for arg in vars(args)])


def evaluate_corrupted_transforms():
    parameters = ["nothing", "replace_corrupted_0s", "replace_corrupted_noise", "add_band_corrupted_arrays"]
    evaluate_segmentation_parameter(parameters=parameters,
                                    parameter_name="transform_method",
                                    iterations=3,
                                    epochs=20)


if __name__ == '__main__':
    evaluate_corrupted_transforms()
