import copy

import torch
import torch.nn as nn

from MNmodel import OneLayerCNN, LinearClassifier, SMLP, CNNClassifier, CNN, SLinearClassifier
import naiveModel
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from fetch_data import fetch_scaler_data, fetch_mnist_data

from data import noniid_partition_loader

fbsz = 2

model = SMLP()
model_path = './model/naiveMLP_NMIST.pt'

model_clone_test = SMLP()

criterion_clone_test = nn.CrossEntropyLoss()
optimizer_clone_test = optim.SGD(model_clone_test.parameters(), lr=0.001)


def train():
    # train_dataset, test_dataset = fetch_mnist_data()
    train_dataset, test_dataset = fetch_scaler_data()

    noniid_client_train_loader = noniid_partition_loader(train_dataset, bsz=fbsz)
    dataset_len = len(noniid_client_train_loader)

    # Download and load the test data

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=fbsz, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

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
