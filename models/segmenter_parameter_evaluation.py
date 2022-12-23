from csv import writer
import os.path as osp
import numpy as np

from models.segmenter import set_args, train
import random as r

from models.utils import loss_functions


def evaluate_segmentation_parameter(parameters, iterations=3, epochs=3):
    """
    Calls segmentation training a predetermined amount of times to evaluate a certain parameter
    """
    for parameter in parameters:
        all_scores = []
        args = set_args()
        setattr(args, 'epochs', epochs)

        for name, val in parameter.items():
            setattr(args, name, val)

        for iteration in range(iterations):
            _, score = ("hi", 25)  # train(args)
            all_scores.append(score)

        mean_score = np.mean(all_scores)

        random_identifier = r.randint(10000, 99999)

        with open('tuning_results_identifiers', 'a+', newline='') as f:
            append_writer = writer(f)
            append_writer.writerow([mean_score, random_identifier])

        with open(osp.join('tuning_results', str(random_identifier)), 'x', newline='') as f:
            append_writer = writer(f)
            append_writer.writerow([f"{getattr(args, arg)}" for arg in vars(args)])


def evaluate_corrupted_transforms(iterations, epochs):
    parameters = [{"transform_method": "nothing"},
                  {"transform_method": "replace_corrupted_0s"},
                  {"transform_method": "replace_corrupted_noise"},
                  {"transform_method": "add_band_corrupted_arrays"}]
    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)


def evaluate_loss_functions(iterations, epochs):
    parameters = [{"loss_function": loss_functions.mse_loss},
                  {"loss_function": loss_functions.rmse_loss},
                  {"loss_function": loss_functions.logit_binary_cross_entropy_loss},
                  {"loss_function": loss_functions.dice_loss}]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)


def evaluate_encoder_model_and_weights(iterations, epochs):
    parameters = [{"loss_function": "efficientnet-b0", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b1", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b2", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b3", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b4", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b5", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b6", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b7", "encoder_weights": None},
                  {"encoder_name": "efficientnet-b0", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b1", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b2", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b3", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b4", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b5", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b6", "encoder_weights": "imagenet"},
                  {"encoder_name": "efficientnet-b7", "encoder_weights": "imagenet"}
                  ]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)

if __name__ == '__main__':
    iterations = 3
    epochs = 20
    evaluate_corrupted_transforms(iterations=iterations, epochs=epochs)
    evaluate_loss_functions(iterations=iterations, epochs=epochs)
    evaluate_encoder_model_and_weights(iterations=iterations, epochs=epochs)