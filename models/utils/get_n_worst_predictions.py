import numpy as np
from models.utils.root_mean_squared_error import root_mean_squared_error


def get_n_worst(predictions, labels, selected_patch_names, n=10) -> list:
    """
    Calculate the worst predictions by comparing them to labels based on RMSE, returns prediction, label, and score
    """
    pred_label_score = []
    for ind, pred in enumerate(predictions):
        label = labels[ind]
        pred_label_score.append((pred, label, root_mean_squared_error(y_true=pred, y_pred=label), selected_patch_names[ind]))
    worst = sorted(pred_label_score, key=lambda x: x[2], reverse=True)
    return worst[0:n]


def get_worst_predictions_from_model(model_prediction_function, X_all, y_all, selected_patch_names, n=10) -> list:
    """
    Calculate the worst predictions from a corrupted_model based on RMSE, returns prediction, label, and score
    """
    pred_label_score = []
    for ind, patch in enumerate(X_all):
        label = y_all[ind]
        pred = model_prediction_function(patch)[0]  # np.array([patch])
        pred_label_score.append((pred, label, root_mean_squared_error(y_true=pred, y_pred=label), selected_patch_names[ind]))

    worst = sorted(pred_label_score, key=lambda x: x[2], reverse=True)

    return worst[0:n]
