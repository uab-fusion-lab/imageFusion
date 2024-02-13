import copy

import torch
import torch.nn as nn

from MNmodel import OneLayerCNN, LinearClassifier, MLP, CNNClassifier, CNN, ResNet18
import naiveModel
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from data import noniid_partition_loader

fbsz = 64

model = ResNet18()
model_path = './model/naiveMLP_NMIST.pt'

model_clone_test = ResNet18()

criterion_clone_test = nn.CrossEntropyLoss()
optimizer_clone_test = optim.SGD(model_clone_test.parameters(), lr=0.001)

def pre_train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=fbsz, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=fbsz, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%")

    print("Training finished!")

    # Model evaluation on the test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")


    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to '{model_path}'")

def load_model():
    model.load_state_dict(torch.load(model_path))
    print("Trained model loaded")

def fusion_train():
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
                # optimizer_clone_test.zero_grad()
                # outputs_clone_test = model_clone_test(images)
                # loss_clone_test = criterion(outputs_clone_test, labels)
                # loss_clone_test.backward()
                # optimizer_clone_test.step()

                fmodel = fusion_model(model)
                # fmodel.basic_model.load_state_dict(copy.deepcopy(model.state_dict()))
                model_clone = ResNet18()
                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
                optimizer_clone = optim.SGD(model_clone.parameters(), lr=0.001)
                optimizer_clone.zero_grad()
                outputs = model_clone(images)
                loss_clone = criterion(outputs, labels)
                loss_clone.backward()
                optimizer_clone.step()

                avg_images = torch.mean(images,dim=0)

                grads = []
                for name, W in model_clone.named_parameters():
                    grad = W.grad.detach()
                    grads.append(grad)

                grads = [g1 + g2 for g1, g2 in zip(grads, grads_diff)]

                final_image = fmodel(avg_images.unsqueeze(0), grads, torch.tensor([labels[0]]))

                optimizer.zero_grad()
                result = model(final_image)
                # index = torch.argmax(result, dim=1)
                loss = criterion(result, torch.tensor([labels[0]]))
                loss.backward()
                optimizer.step()

                grads_new = []
                for name, W in model.named_parameters():
                    grad = W.grad.detach()
                    grads_new.append(grad)

                grads_diff = [g1 - g2 for g1, g2 in zip(grads, grads_new)]

                # print("p......")
            # test_acc(model_clone_test, test_loader)
            # test_acc(model, test_loader)
                # test_acc(model_clone_test, test_loader)
                # if index == labels[0]:
                #     print("result are correct")
                # else:
                #     print("error")
            # test_acc(model, test_loader)
        # test_acc(model_clone_test, test_loader)
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


class fusion_model(nn.Module):

    def __init__(self, basic_model):
        super(fusion_model, self).__init__()
        self.basic_model = basic_model
        self.rand = True
        self.step_size = 8.0 / 255
        self.epsilon = 8.0 / 255
        self.num_steps = 30


    def forward(self, input, grad_target, target):  # do forward in the module.py
        # if not args.attack :
        #    return self.basic_model(input), input

        x = input.detach()
        citerien = nn.L1Loss(reduction='sum')

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False)
                # print(loss)
                grad_w = torch.autograd.grad(loss, self.basic_model.parameters(),create_graph=True)
                # grad_w_reshape = self.change_dim(grad_w)
                # grad_target_reshape = self.change_dim(grad_target)
                # loss_w = F.mse_loss(grad_w_reshape, grad_target_reshape)
                # loss_0 = citerien(grad_w[0], grad_target[0])
                # loss_1 = citerien(grad_w[1], grad_target[1])
                # loss_2 = citerien(grad_w[2], grad_target[2])
                # loss_3 = citerien(grad_w[3], grad_target[3])
                # loss_4 = citerien(grad_w[4], grad_target[4])
                # loss_5 = citerien(grad_w[5], grad_target[5])
                # loss_6 = citerien(grad_w[6], grad_target[6])
                # loss_7 = citerien(grad_w[7], grad_target[7])

                loss_w = -1
                for i in range(len(grad_w)):
                    if loss_w == -1:
                        loss_w = citerien(grad_w[i], grad_target[i])
                    else:
                        loss_w += citerien(grad_w[i], grad_target[i])


                # loss_w = loss_0 + loss_1 + 1 * loss_2 + loss_3 # + loss_4 + loss_5 + loss_6 + loss_7
                # print(loss_w)
            grad_x = torch.autograd.grad(loss_w, [x])[0]
            x = x.detach() - self.step_size * torch.sign(grad_x.detach())
            # x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)

        # show_image(x)


        return x
    def change_dim(self, list):
        for index in range(len(list)):
            ele = list[index]
            if len(ele.shape) == 1:
                list[index] = ele.unsqueeze(1)
        return torch.cat(list, dim=1)

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

    fusion_train()
