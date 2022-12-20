from csv import writer

from models.segmenter import set_args, train


def evaluate_segmentation_parameter(parameters, parameter_name, results_filename, iterations=3, epochs=3):
    """
    Calls segmentation training a predetermined amount of times to evaluate a certain parameter
    """
    for parameter in parameters:
        for iteration in range(iterations):
            args = set_args()
            setattr(args, 'epochs', epochs)
            setattr(args, parameter_name, parameter)
            _, score = train(args)

            with open(results_filename, 'a+', newline='') as f:
                append_writer = writer(f)
                append_writer.writerow([str(parameter), score])


def evaluate_corrupted_transforms():
    parameters = ["nothing", "replace_corrupted_0s", "add_band_corrupted_arrays"]
    evaluate_segmentation_parameter(parameters=parameters,
                                    parameter_name="transform_method",
                                    results_filename="train_results.csv",
                                    iterations=2,
                                    epochs=2)



if __name__ == '__main__':
    evaluate_corrupted_transforms()
