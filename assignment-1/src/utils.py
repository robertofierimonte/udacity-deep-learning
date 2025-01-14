import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms.functional as tF
from torch import Tensor


def show_grid(imgs: list[Tensor] | Tensor) -> None:
    """Show a grid of images using matplotlib.

    Args:
        imgs (list[Tensor] | Tensor): Image tensor with shape N x C x H x W, or list
            of len N of tensors with shape C x H x W.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def _matplotlib_imshow(img: Tensor) -> None:
    """Plot an image tensor using matplotlib.

    Args:
        img (Tensor): Image tensor with shape C x H x W.
    """
    npimg = img.numpy()
    plt.imshow(npimg, cmap="Greys")


def _images_to_probs(net: nn.Module, images: Tensor) -> tuple[np.ndarray, list[float]]:
    """Generates predictions and probabilities from a trained network and a list of images.

    Args:
        net (nn.Module): The model used to generate the predictions.
        images (Tensor): Image tensor with shape N x C x H x W.

    Returns:
        np.ndarray: Array (N,) of model predictions.
        list[float]: List of len N of probabilities for the predicted class.
    """
    output = net(images)

    # Convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [nnF.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(
    net: nn.Module, images: Tensor, labels: Tensor, classes: list
) -> matplotlib.figure.Figure:
    """Generate a plot of images together with real and predicted classes.

    Generates a matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, colouring this
    information based on whether the prediction was correct or not.

    Args:
        net (nn.Module): Model used to generate the predictions.
        images (Tensor): Image tensor with shape N x H x W.
        labels (Tensor): Targer tensor with shape N
        classes (list): List of len T of all unique image labels.

    Returns:
        matplotlib.figure.Figure: Plot of images with corresponding ground truth and
            predicted labels.
    """
    # Shuffle the data before plotting
    perm = torch.randperm(labels.shape[0])
    images = images[perm, :, :]
    labels = labels[perm]

    preds, probs = _images_to_probs(net, images[:, None, :, :])

    # Plot the images in the batch, along with predicted and true labels
    # Plot at most 100 images
    ncols = 5
    nplots = min(images.shape[0], 100)
    nrows = int(np.ceil(nplots / ncols))

    fig = plt.figure(figsize=(12, 60))
    for idx in np.arange(nplots):
        ax = fig.add_subplot(nrows, ncols, idx + 1, xticks=[], yticks=[])
        _matplotlib_imshow(images[idx])
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig
