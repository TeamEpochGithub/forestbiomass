import torch.nn.functional as F
import torch


def mse_loss(prediction, target):
    return F.mse_loss(prediction, target)


def rmse_loss(prediction, target):
    return torch.sqrt(F.mse_loss(prediction, target))


def huber_loss(y_true, y_pred, delta=40.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.where(abs_error < delta, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta))
    return torch.mean(quadratic)


def rmsle_loss(y_true, y_pred):
    terms_to_sum = (torch.log(y_pred + 1) - torch.log(y_true + 1)) ** 2
    return torch.sqrt(torch.mean(terms_to_sum))


# Always seems to have a loss of 0.
def cross_entropy_loss(prediction, target):
    return F.cross_entropy(prediction, target)


# Supposedly doesn't work if the label contains negative values.
# Gives a Cuda error if run on my (Lars) laptop.
def binary_cross_entropy_loss(prediction, target):
    return F.binary_cross_entropy(prediction, target)


def logit_binary_cross_entropy_loss(prediction, target):
    return F.binary_cross_entropy_with_logits(prediction, target)


# Source: https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183?permalink_comment_id=3025867#gistcomment-3025867
def dice_loss(prediction, target):
    numerator = 2 * torch.sum(prediction * target)
    denominator = torch.sum(prediction + target)
    return 1 - (numerator + 1) / (denominator + 1)
