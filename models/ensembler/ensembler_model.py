# ensemble_model =
# 	- 180 datapoints in model
# 	- 3 weights out of model (x1, x2, x3)
#
# prediction =
# 	- x1 * (inference swin_transformer) + x2 * (inference segmenter) + x3 *  (inference pixel_wise)
# 	- then calculate and backpropagate model over ensemble_model
import torch
from torch import nn
import matplotlib.pyplot as plt

from models.utils.loss_functions import rmse_loss

metadata_dim = 180
n_hidden = 180
weights_dim = 3
batch_size = 32
learning_rate = 1e-4
image_size = 256

input = torch.randn(batch_size, metadata_dim)
label = (torch.rand(size=(image_size, image_size))).float()

metadata_model = nn.Sequential(nn.Linear(metadata_dim, 2 * n_hidden),
                               nn.ReLU(),
                               nn.Linear(2 * n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, weights_dim),
                               nn.Softmax(0))

loss_function = rmse_loss
optimizer = torch.optim.SGD(metadata_model.parameters(), lr=learning_rate)
epochs = 100

### Here, trained models should be loaded. Model-specific data preparation also needs to be done.
placeholder = (lambda x: torch.rand((batch_size, image_size, image_size)))

segmentation_model = placeholder
swin_model = placeholder
pixelwise_model = placeholder
###

if __name__ == '__main__':

    losses = []
    for epoch in range(epochs):
        w1, w2, w3 = torch.tensor_split(metadata_model(input), weights_dim, dim=1)

        w1, w2, w3 = w1.view(-1, 1, 1), w2.view(-1, 1, 1), w3.view(-1, 1, 1)
        pred_y = (w1 * segmentation_model(input)) + (w2 * swin_model(input)) + (w3 * pixelwise_model(input))

        loss = loss_function(pred_y, label)
        losses.append(loss.item())

        metadata_model.zero_grad()
        loss.backward()

        optimizer.step()

    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f" % learning_rate)
    plt.show()
