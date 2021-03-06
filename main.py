from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import datasets, transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm

from mlp import MLP


def load_data(data_dir="./data"):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.KMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.KMNIST(root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(config, checkpoint_dir=None, data_dir=None, num_epochs=10):
    net = MLP(config["l1"], config["l2"], config["dr"])
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True)

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_epoch_loss = 0
        train_epoch_acc = 0

        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            train_epoch_acc += torch.mean(equals.type(torch.FloatTensor))

        # Validation loss
        val_epoch_loss = 0
        val_epoch_acc = 0

        net.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)
                    val_epoch_acc += torch.mean(equals.type(torch.FloatTensor))

                    loss = criterion(outputs, labels)
                    val_epoch_loss += loss.item()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        loss_stats['train'].append(train_epoch_loss / len(trainloader))
        loss_stats['val'].append(val_epoch_loss / len(valloader))
        accuracy_stats['train'].append(train_epoch_acc.item() / len(trainloader))
        accuracy_stats['val'].append(val_epoch_acc.item() / len(valloader))

        tune.report(train_loss=train_epoch_loss / len(trainloader), loss=val_epoch_loss / len(valloader),
                    train_accuracy=train_epoch_acc.item() / len(trainloader),
                    accuracy=val_epoch_acc.item() / len(valloader))
    print("Finished Training")
    return accuracy_stats, loss_stats


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False)

    val_epoch_acc = 0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            val_epoch_acc += torch.mean(equals.type(torch.FloatTensor))

    return val_epoch_acc.item() / len(testloader)


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.grid_search([2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
        "l2": tune.grid_search([2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
        "lr": tune.grid_search([0.0005, 0.001, 0.0007]),  # Learning Rate
        "batch_size": tune.grid_search([64, 128, 256]),  # Batch Size
        "dr": tune.grid_search([0.3, 0.5, 0.85]),  # Dropout
        # "momentum": tune.uniform(0.1, 0.9)
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["train_loss", "loss", "train_accuracy", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train, data_dir=data_dir, num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./results",
        name="test_experiment")

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    # dfs = result.trial_dataframes
    # # Plot by epoch
    # ax = None  # This plots everything on the same plot
    # for d in dfs.values():
    #     ax = d.accuracy.plot(ax=ax, legend=False)
    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("Accuracy")
    # plt.savefig("./mlp-accuracy.png")
    # plt.show()

    best_trained_model = MLP(best_trial.config['l1'], best_trial.config['l2'], best_trial.config['dr'])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=1, max_num_epochs=200, gpus_per_trial=2)
