import atexit
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import click
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import datasets, transforms, utils

from udacity_deep_learning.models import DenseNet, Lenet5
from udacity_deep_learning.utils import plot_classes_preds, plot_grid


@click.command()
@click.option("--model", type=click.Choice(["densenet", "lenet5"]), default="lenet5")
def main(model: Literal["densenet", "lenet5"]) -> None:

    file_path = Path(sys.argv[0])
    project_path = file_path.parents[1]
    data_path = project_path / "data"

    # Check if GPU is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using {device} device.")

    # Load the raw training data
    raw_data = datasets.MNIST(data_path, download=True, train=True, transform=transforms.ToTensor())
    raw_loader = DataLoader(raw_data, batch_size=16)

    # Initialize MLflow tracking
    mlflow.set_experiment("handwritten-digits-classifier")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run = mlflow.start_run(run_name=run_name, log_system_metrics=True)
    run_path = run.info.artifact_uri

    # Initialise TensorBoard writer
    writer = SummaryWriter(run_path)

    # Register closing the MLflow run and the TensorBoard writer to be called when the program exits
    atexit.register(mlflow.end_run)
    atexit.register(writer.close)
    atexit.register(writer.flush)

    # Explore the raw data
    logger.info(f"Number of samples: {len(raw_data)}.")
    logger.info(f"Number of classes: {len(raw_data.classes)}.")
    logger.info(f"Classes: {raw_data.classes}.")
    mlflow.log_text(str(raw_data.classes), "classes.txt")

    # Log a sample of the raw data to TensorBoard
    raw_batch = next(iter(raw_loader))[0]
    logger.info(f"Shape of batch: {tuple(raw_batch.shape)}.")
    logger.info(
        f"Min pixel value: {raw_batch.min().item()}, max pixel value: {raw_batch.max().item()}."
    )
    raw_sample = next(iter(raw_loader))[0]
    raw_grid = plot_grid(raw_sample)
    mlflow.log_figure(raw_grid, "raw_data_sample.png")

    # Set experiment configuration
    batch_size = 32  # Batch size
    n_epochs = 10  # Number of training epochs]

    # Log experiment configuration to MLflow
    mlflow.log_param("model", model)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("n_epochs", n_epochs)

    # Define the data transformations and load the training, validation and test data
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomInvert(p=0.2),
            transforms.RandomRotation(degrees=(-10, 10), expand=False),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        ]
    )
    train_data = datasets.MNIST(data_path, download=True, train=True, transform=train_transform)
    # For the test and validation data we don't need to apply data augmentation
    test_val_data = datasets.MNIST(
        data_path, download=True, train=False, transform=transforms.ToTensor()
    )
    test_val_generator = torch.Generator().manual_seed(42)
    test_data, val_data = random_split(
        test_val_data, lengths=[0.5, 0.5], generator=test_val_generator
    )

    # Initialize the data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    logger.info(f"Number of training samples: {len(train_data)}.")
    logger.info(f"Number of validation samples: {len(val_data)}.")
    logger.info(f"Number of test samples: {len(test_data)}.")

    # Log a sample of the training and test data to TensorBoard
    train_sample, test_sample = next(iter(train_loader))[0], next(iter(test_loader))[0]
    train_grid = utils.make_grid(train_sample)
    test_grid = utils.make_grid(test_sample)
    # mlflow.log_image(train_grid.numpy(), "train_data_sample.png")
    # mlflow.log_image(test_grid.numpy(), "test_data_sample.png")

    # Initialize the model and move it to the appropriate device
    if model == "densenet":
        net = DenseNet()
    else:
        net = Lenet5()
    net.to(device)

    # Log the model summary to MLflow
    model_summary = summary(net, input_size=[batch_size, 1, 28, 28])
    mlflow.log_text(str(model_summary), "model_summary.txt")

    # Define the loss function (cross-entropy loss) and the optimizer
    if model == "densenet":
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the Neural Network
    train_loss_history = list()
    val_loss_history = list()

    # Log the experiment configuration to TensorBoard
    writer.add_text("Model type", model)
    writer.add_scalar("Batch size", batch_size)
    writer.add_scalar("Number of epochs", n_epochs)
    writer.add_image("Train images", train_grid)
    writer.add_image("Test images", test_grid)

    for epoch in range(n_epochs):
        net.train()
        train_loss = 0.0
        train_correct = 0
        for i, data in enumerate(train_loader):
            # Data is a list of [inputs, labels]
            inputs, labels = data

            # Log the model during the first iteration
            if epoch == 0:
                writer.add_graph(net, inputs)

            # Pass to GPU if available.
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            train_correct += (preds == labels).float().mean().item()
            train_loss += loss.item()

        # Log the training stats
        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch + 1)
        writer.add_scalar("Accuracy/train", train_correct / len(train_loader), epoch + 1)
        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch + 1)
        mlflow.log_metric("train_accuracy", train_correct / len(train_loader), step=epoch + 1)
        logger.info(
            f"Epoch {epoch + 1} training accuracy: {train_correct / len(train_loader):.2%} "
            f"training loss: {train_loss / len(train_loader):.5f}."
        )
        train_loss_history.append(train_loss / len(train_loader))

        val_loss = 0.0
        val_correct = 0.0
        net.eval()
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            val_correct += (preds == labels).float().mean().item()
            val_loss += loss.item()

        # Log the validation stats
        writer.add_scalar("Loss/valid", val_loss / len(val_loader), epoch + 1)
        writer.add_scalar("Accuracy/valid", val_correct / len(val_loader), epoch + 1)
        mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch + 1)
        mlflow.log_metric("val_accuracy", val_correct / len(val_loader), step=epoch + 1)
        logger.info(
            f"Epoch {epoch + 1} validation accuracy: {val_correct / len(val_loader):.2%} "
            f"validation loss: {val_loss / len(val_loader):.5f}."
        )
        val_loss_history.append(val_loss / len(val_loader))

    # Log the model
    mlflow.pytorch.log_model(
        pytorch_model=net,
        name="model",
        registered_model_name="handwritten-digits-classifier",
        export_model=True,
        input_example=test_sample.numpy(),
        signature=infer_signature(
            test_sample.numpy(), net(test_sample.to(device)).cpu().detach().numpy()
        ),
    )

    # Evaluate the model on the test set
    test_loss = 0.0
    test_correct = 0.0
    net.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        test_correct += (preds == labels).float().mean().item()
        test_loss += loss.item()

    # Log the test stats
    writer.add_scalar("Loss/test", test_loss / len(test_loader), n_epochs)
    writer.add_scalar("Accuracy/test", test_correct / len(test_loader), n_epochs)
    mlflow.log_metric("test_loss", test_loss / len(test_loader), step=n_epochs)
    mlflow.log_metric("test_accuracy", test_correct / len(test_loader), step=n_epochs)
    logger.info(
        f"Test accuracy: {test_correct / len(test_loader):.2%} "
        f"test loss: {test_loss / len(test_loader):.5f}."
    )

    # Log the predictions vs actuals plot to TensorBoard and MLflow
    test_actuals_preds = plot_classes_preds(
        net=net,
        images=test_data.dataset.data.float(),
        labels=test_data.dataset.targets,
        classes=test_data.dataset.classes,
    )
    writer.add_figure("Test/predictions vs actuals", test_actuals_preds, n_epochs)
    mlflow.log_figure(test_actuals_preds, "test_predictions_vs_actuals.png")

    logger.success("Script completed.")


if __name__ == "__main__":
    main()
