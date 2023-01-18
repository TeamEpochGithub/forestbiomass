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
            _, score = train(args)
            all_scores.append(float(score))
        print(all_scores)
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
    parameters = [{"train_loss_function": loss_functions.mse_loss},
                  {"train_loss_function": loss_functions.rmse_loss},
                  {"train_loss_function": loss_functions.logit_binary_cross_entropy_loss},
                  {"train_loss_function": loss_functions.dice_loss}]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)


def evaluate_encoder_model_and_weights(iterations, epochs):
    parameters = [{"encoder_name": "efficientnet-b0", "encoder_weights": None},
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


def evaluate_model_architectures(iterations, epochs):
    parameters = [{"band_segmenter_name": "Unet", "month_segmenter_name": "Unet"},
                  {"band_segmenter_name": "Unet++", "month_segmenter_name": "Unet++"},
                  {"band_segmenter_name": "MAnet", "month_segmenter_name": "MAnet"},
                  {"band_segmenter_name": "Linknet", "month_segmenter_name": "Linknet"},
                  {"band_segmenter_name": "FPN", "month_segmenter_name": "FPN"},
                  {"band_segmenter_name": "PSPNet", "month_segmenter_name": "PSPNet"},
                  {"band_segmenter_name": "PAN", "month_segmenter_name": "PAN"},
                  {"band_segmenter_name": "DeepLabV3", "month_segmenter_name": "DeepLabV3"},
                  {"band_segmenter_name": "DeepLabV3+", "month_segmenter_name": "DeepLabV3+"},
                  ]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)


def evaluate_band_selection(iterations, epochs):
    parameters = [
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
        {"band_selection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
        ]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)


if __name__ == '__main__':
    iterations = 3
    epochs = 20

    # evaluate_corrupted_transforms(iterations=iterations, epochs=epochs)
    evaluate_loss_functions(iterations=iterations, epochs=epochs)
    # evaluate_encoder_model_and_weights(iterations=iterations, epochs=epochs)
