from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate RMSE based on label and prediction, can be used in models too
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
