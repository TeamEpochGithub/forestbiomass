from models.segmenter import set_args, train


def evaluate_segmentation_parameter():
    iterations = 2
    epochs = 1

    parameters, parameter_name = corrupted_transforms()  # CHANGE THIS BASED ON TRIAL

    for parameter in parameters:
        for iteration in range(iterations):
            args = set_args()
            setattr(args, 'epochs', epochs)
            setattr(args, parameter_name, parameter)
            # args['epochs'] = epochs
            # args[parameter_name] = parameter
            train(args)


def corrupted_transforms():
    return ["nothing", "replace_corrupted_0s", "add_band_corrupted_arrays"], "transform_method"


if __name__ == '__main__':
    evaluate_segmentation_parameter()
