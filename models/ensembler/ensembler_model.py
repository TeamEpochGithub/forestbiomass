# ensemble_model =
# 	- 180 datapoints in model
# 	- 3 weights out of model (x1, x2, x3)
#
# prediction =
# 	- x1 * (inference swin_transformer) + x2 * (inference segmenter) + x3 *  (inference pixel_wise)
# 	- then calculate and backpropagate model over ensemble_model
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ensembler.extract_metadata import extract_metadata_batch
from models.ensembler.universal_dataloader import UniversalDatasetTrain
from models.utils.loss_functions import rmse_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bar(curr_step, total_steps, bar_length=30):
    """
    Printing progress bar.
    """
    filled = int(bar_length * curr_step / total_steps)
    return f"[{'#' * filled}{'.' * (bar_length - filled)}]"


def train_and_eval(model, train_loader, valid_loader, optimizer, criterion, epochs=10, log_freq=100):
    """
    Train and evaluate the model.

    @param model: (torch.nn.Module) model to train
    @param train_loader: (torch.utils.data.DataLoader) train dataloader
    @param valid_loader: (torch.utils.data.DataLoader) validation dataloader
    @param optimizer: (torch.optim) optimizer
    @param criterion: (torch.nn) loss function
    @param epochs: (int) number of epochs
    @return: (list, list) train loss and validation loss
    """
    global_train_loss, global_valid_loss = [], []
    best_train_loss, best_valid_loss = 100, 100

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = []
        for step, (features, agbm) in enumerate(train_loader):
            # ======== TRAINING ========
            metadata = extract_metadata_batch(features)

            # 1. Move the tensors to the configured device.
            # metadata, agbm, features = metadata.to(DEVICE), agbm.to(DEVICE), features.to(DEVICE)
            # 2. Forward pass by passing the images through the model.
            w1, w2, w3 = torch.tensor_split(model(metadata), 3, dim=1)
            w1, w2, w3 = w1.view(-1, 1, 1), w2.view(-1, 1, 1), w3.view(-1, 1, 1)
            pred = (w1 * segmentation_model(features)) + (w2 * swin_model(features)) + (w3 * pixelwise_model(features))

            # 3. Zero the gradients of all model parameters.
            optimizer.zero_grad()
            # 4. Compute the loss.
            loss = criterion(pred, agbm)
            # 5. Backward pass to compute the gradients of the loss w.r.t. the model parameters.
            loss.backward()
            # 6. Step the optimizer to update the model parameters.
            optimizer.step()
            # =========================

            # Logging metrics
            train_loss.append(loss.item())

            if step % log_freq == 0 or step == len(train_loader) - 1:
                print(
                    f"Epoch {epoch + 1:2d}/{epochs:2d} {bar(step + 1, len(train_loader))} Train-Loss: {np.mean(train_loss):.4f} \r",
                    end="")
        print()

        # Evaluation loop
        model.eval()
        valid_loss = []
        with torch.no_grad():
            for step, (features, agbm) in enumerate(valid_loader):
                metadata = extract_metadata_batch(features)

                # ======== VALIDATION ========
                # 1. Move the tensors to the configured device.
                # metadata, agbm, features = metadata.to(DEVICE), agbm.to(DEVICE), features.to(DEVICE)
                # 2. Forward pass by passing the images through the model.
                w1, w2, w3 = torch.tensor_split(model(metadata), 3, dim=1)
                w1, w2, w3 = w1.view(-1, 1, 1), w2.view(-1, 1, 1), w3.view(-1, 1, 1)
                pred = (w1 * segmentation_model(features)) + (w2 * swin_model(features)) + (w3 * pixelwise_model(features))

                # 3. Compute the loss.
                loss = criterion(pred, agbm)
                # Note we do not backpropagate the gradients in the validation loop.
                # ==========================

                # Logging metrics
                valid_loss.append(loss.item())

                if step % log_freq == 0 or step == len(valid_loader) - 1:
                    print(
                        f"Epoch {epoch + 1:2d}/{epochs:2d} {bar(step + 1, len(valid_loader))} Valid-Loss: {np.mean(valid_loss):.4f} \r",
                        end="")
            print()

        global_train_loss.extend(train_loss)
        global_valid_loss.extend(valid_loss)
        best_train_loss = min(best_train_loss, np.mean(train_loss))
        best_valid_loss = min(best_valid_loss, np.mean(valid_loss))
        print(f"Best Train Loss: {best_train_loss} Best Valid Loss: {best_valid_loss}")

    return global_train_loss, global_valid_loss


def create_metadata_model(metadata_dim, n_hidden, weights_dim):
    return nn.Sequential(nn.Linear(metadata_dim, 2 * n_hidden),
                         nn.ReLU(),
                         nn.Linear(2 * n_hidden, n_hidden),
                         nn.ReLU(),
                         nn.Linear(n_hidden, weights_dim),
                         nn.Softmax(0))


def create_train_val_dataloaders(num_workers):
    dataset = UniversalDatasetTrain()
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, valid_size])

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader_train, dataloader_val


if __name__ == '__main__':
    metadata_dim = 180
    n_hidden = 180
    weights_dim = 3
    learning_rate = 1e-4
    metadata_model = create_metadata_model(metadata_dim=metadata_dim, n_hidden=n_hidden, weights_dim=weights_dim)
    optimizer = torch.optim.SGD(metadata_model.parameters(), lr=learning_rate)
    loss_function = rmse_loss
    epochs = 100

    batch_size = 16
    image_size = 256
    placeholder = (lambda x: torch.rand((x.shape[0], image_size, image_size)))
    # TODO: Placeholders should be replaced by a method that takes a batch and returns a batch of predictions.
    #  To make the predictions a pretrained model should be loaded.
    segmentation_model = placeholder
    swin_model = placeholder
    pixelwise_model = placeholder

    train_loader, valid_loader = create_train_val_dataloaders(num_workers=20)

    train_and_eval(model=metadata_model,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   optimizer=optimizer,
                   criterion=loss_function,
                   epochs=epochs, log_freq=100)
