import copy

import torch
import torch.nn as nn

from fetch_data import fetch_scaler_data, fetch_mnist_data
from MNmodel import OneLayerCNN, LinearClassifier, MLP, CNNClassifier, CNN, ResNet18
import naiveModel
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from data import noniid_partition_loader

fbsz = 64

model = MLP()
model_path = './model/naiveMLP_NMIST.pt'

model_clone_test = MLP()

criterion_clone_test = nn.CrossEntropyLoss()
optimizer_clone_test = optim.SGD(model_clone_test.parameters(), lr=0.001)

def fusion_train():
    train_dataset, test_dataset = fetch_mnist_data()
    # fetch_scaler_data()

    noniid_client_train_loader = noniid_partition_loader(train_dataset, bsz=fbsz)
    dataset_len = len(noniid_client_train_loader)

    # Download and load the test data

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=fbsz, shuffle=False)

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

        model.train()
        for d in range(dataset_len):
            for i, (images, labels) in enumerate(noniid_client_train_loader[d]):

                fmodel = fusion_model(model)
                # fmodel.basic_model.load_state_dict(copy.deepcopy(model.state_dict()))
                model_clone = MLP()
                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
                optimizer_clone = optim.SGD(model_clone.parameters(), lr=0.001)
                optimizer_clone.zero_grad()
                model_clone.train()
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

        x = input.detach()
        citerien = nn.L1Loss(reduction='sum')

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False)
                grad_w = torch.autograd.grad(loss, self.basic_model.parameters(),create_graph=True)

                grads = [citerien(x, y) for x, y in zip(grad_w, grad_target)]
                loss_w = sum(grads)

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


    fusion_train()
