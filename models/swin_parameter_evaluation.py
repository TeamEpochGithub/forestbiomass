from csv import writer
import os.path as osp
import numpy as np

from models.segmenter_efficient_swin_transformation import set_args, train
import random as r

from models.utils import loss_functions
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


def evaluate_different_bands(iterations, epochs):
    parameters = [{"bands_to_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
                  {"bands_to_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]}]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)

def evaluate_different_augmentations(iterations, epochs):
    aug_set0 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])

    aug_set1 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomGridShuffle(),
        ToTensorV2()
    ])

    aug_set2 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomGridShuffle(),
        A.Cutout(num_holes=12, max_h_size=12, max_w_size=12, p=0.5),
        ToTensorV2()
    ])
    parameters = [{"augmentation_set": aug_set0},
                  {"augmentation_set": aug_set1},
                  {"augmentation_set": aug_set2}]

    evaluate_segmentation_parameter(parameters=parameters,
                                    iterations=iterations,
                                    epochs=epochs)



if __name__ == '__main__':
    iterations = 3
    epochs = 20

    # evaluate_corrupted_transforms(iterations=iterations, epochs=epochs)
    evaluate_loss_functions(iterations=iterations, epochs=epochs)
    # evaluate_encoder_model_and_weights(iterations=iterations, epochs=epochs)