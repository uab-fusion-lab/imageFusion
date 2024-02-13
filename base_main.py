import copy

import torch
import torch.nn as nn

from MNmodel import OneLayerCNN, LinearClassifier, MLP, CNNClassifier, CNN
import naiveModel
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from data import noniid_partition_loader

fbsz = 64

model = CNN()
model_path = './model/naiveMLP_NMIST.pt'

model_clone_test = CNN()

criterion_clone_test = nn.CrossEntropyLoss()
optimizer_clone_test = optim.SGD(model_clone_test.parameters(), lr=0.001)


def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=fbsz, shuffle=True)

    noniid_client_train_loader = noniid_partition_loader(train_dataset, bsz=fbsz)
    dataset_len = len(noniid_client_train_loader)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=fbsz, shuffle=False)

    # Create a dictionary to hold datasets for each label
    label_datasets = {}

    # Loop through the dataset and separate by labels
    for data, label in train_dataset:
        if label not in label_datasets:
            label_datasets[label] = []
        label_datasets[label].append((data, label))

    # Create data loaders for each label dataset
    label_data_loaders = {}
    for label, data_samples in label_datasets.items():
        label_data_loaders[label] = torch.utils.data.DataLoader(data_samples, batch_size=fbsz, shuffle=True)

    label_datasets_test = {}

    # Loop through the dataset and separate by labels
    for data, label in test_dataset:
        if label not in label_datasets_test:
            label_datasets_test[label] = []
        label_datasets_test[label].append((data, label))

    label_data_loaders_test = {}
    for label, data_samples in label_datasets.items():
        label_data_loaders_test[label] = torch.utils.data.DataLoader(data_samples, batch_size=fbsz, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model_clone_test.train()

    #
    # optimizer_test = optim.Adam(model_clone_test.parameters(), lr=0.001)
    # Training loop
    num_epochs = 100

    grads_diff = []
    for name, W in model.named_parameters():
        z = torch.zeros_like(W)
        grads_diff.append(z)
    # test_acc(model_clone_test, test_loader)
    for epoch in range(num_epochs):
        print(f"epoch: {epoch:.2f}")

        # model.train()
        # model_clone_test.train()
        for d in range(dataset_len):
            for i, (images, labels) in enumerate(noniid_client_train_loader[d]):

                optimizer.zero_grad()
                outputs = model(images)
                loss_clone = criterion(outputs, labels)
                loss_clone.backward()
                optimizer.step()

        test_acc(model, test_loader)


def test_acc(model_clone_test, test_loader):
    model_clone_test.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_clone_test(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")



def show_image(x):
    import matplotlib.pyplot as plt
    # Plot using matplotlib
    plt.imshow(x.squeeze(0).T)
    plt.show()
    x = torch.clamp(x, 0, 1)

if __name__ == '__main__':
    # pre_train()

    # load_model()
    # print(model)
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Size: {param.size()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params}")

    # conv_params = list(model.linear.parameters())

    # Calculate the total number of parameters in the convolutional layer
    # total_conv_params = sum(p.numel() for p in conv_params)
    # print(f"Total parameters in the convolutional layer: {total_conv_params}")
    #
    # fc1_params = list(model.fc1.parameters())
    #
    # # Calculate the total number of parameters in the convolutional layer
    # total_fc1_params = sum(p.numel() for p in fc1_params)
    # print(f"Total parameters in the fully connected layer: {total_fc1_params}")

    train()
