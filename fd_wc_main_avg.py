import copy

import torch
import torch.nn as nn

from fetch_data import fetch_scaler_data, fetch_mnist_data
from MNmodel import OneLayerCNN, LinearClassifier, MLP, CNNClassifier, CNN
import naiveModel
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from data import noniid_partition_loader

fbsz = 64

model = CNN()

def fd_fusion_train(client):
    train_dataset, test_dataset = fetch_mnist_data()
    # train_dataset, test_dataset = fetch_scaler_data()

    noniid_client_train_loader = noniid_partition_loader(train_dataset, bsz=fbsz)
    dataset_len = len(noniid_client_train_loader)

    # Download and load the test data

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    #
    # optimizer_test = optim.Adam(model_clone_test.parameters(), lr=0.001)
    # Training loop
    num_epochs = 100

    grads_diff = initial_grad()
    # test_acc(model_clone_test, test_loader)
    global_model = model
    for epoch in range(num_epochs):
        # print(f"epoch: {epoch:.2f}")

        # model.train()
        # model_clone_test.train()
        for d in range(dataset_len):
            training_images = []
            sum_grad = initial_grad()
            for i, (images, labels) in enumerate(noniid_client_train_loader[d]):
                final_image, grads = train_local(criterion, global_model, grads_diff, images, labels)
                training_images.append(final_image)
                # training_images = final_image.repeat(fbsz, 1, 1)
                sum_grad = [g1 + g2 for g1, g2 in zip(grads, sum_grad)]


                if i % client == client-1:
                    train_image = torch.cat(training_images, dim=0)
                    avg_grad = [tensor / client for tensor in sum_grad]



                    optimizer.zero_grad()
                    result = global_model(train_image)
                        # index = torch.argmax(result, dim=1)
                        # torch.full((client,), labels[0])
                    loss = criterion(result, torch.tensor(torch.full((client,), labels[0])))
                        # loss = criterion(result, torch.tensor(labels[:client]))
                    loss.backward()
                    optimizer.step()

                    grads_new = []
                    for name, W in global_model.named_parameters():
                        grad = W.grad.detach()
                        grads_new.append(grad)

                    grads_diff = [(g1 - g2) for g1, g2 in zip(avg_grad, grads_new)]

                    training_images = []
                    sum_grad = initial_grad()

                # print("p......")
            # test_acc(model_clone_test, test_loader)
            # test_acc(model, test_loader)
                # test_acc(model_clone_test, test_loader)
                # if index == labels[0]:
                #     print("result are correct")
                # else:
                #     print("error")
            # test_acc(model, test_loader)
        test_acc(model, test_loader)


def initial_grad():
    grads_diff = []
    for name, W in model.named_parameters():
        z = torch.zeros_like(W)
        grads_diff.append(z)
    return grads_diff


def train_local(criterion, global_model, grads_diff, images, labels):

    fmodel = fusion_model(global_model)


    grads = []
    bs = images.shape[0]
    for e in range(bs):
        local_model = copy.deepcopy(global_model)
        optimizer_clone = optim.SGD(local_model.parameters(), lr=0.001)
        optimizer_clone.zero_grad()
        outputs = local_model(images[e].unsqueeze(0))
        loss_clone = criterion(outputs, labels[e].unsqueeze(0))
        loss_clone.backward()
        optimizer_clone.step()
        avg_images = torch.mean(images, dim=0)
        i=0
        for name, W in local_model.named_parameters():
            grad = W.grad.detach()
            if e == 0 :
                grads.append(grad)
            else:
                grads[i] += grad
            i+=1
    grads = [x/bs for x in grads]
    grads = [g1 + g2 for g1, g2 in zip(grads, grads_diff)]
    final_image = fmodel(avg_images.unsqueeze(0), grads, torch.tensor([labels[0]]))
    return final_image, grads


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
    print(f"{test_accuracy:.2f},")


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
                grads = [citerien(x, y) for x, y in zip(grad_w, grad_target)]
                loss_w = sum(grads)
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
    plt.grid(False)  # Ensure that no grid is displayed
    plt.axis('off')  # Hide the axis
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
    print("fed avg client = 1")
    fd_fusion_train(4)
