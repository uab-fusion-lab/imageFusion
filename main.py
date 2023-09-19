import copy

import torch
import torch.nn as nn
import naiveModel
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

model = naiveModel.SimpleConvNet()
model_path = './model/naiveModel.pt'

def pre_train():
    # Load CIFAR-10 dataset and apply transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    # Load CIFAR-10 dataset and apply transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create a dictionary to hold datasets for each label
    label_datasets = {}

    # Loop through the dataset and separate by labels
    for data, label in train_dataset:
        if label not in label_datasets:
            label_datasets[label] = []
        label_datasets[label].append((data, label))

    # Create data loaders for each label dataset
    batch_size = 8
    label_data_loaders = {}
    for label, data_samples in label_datasets.items():
        label_data_loaders[label] = torch.utils.data.DataLoader(data_samples, batch_size=batch_size, shuffle=True)

    label_datasets_test = {}

    # Loop through the dataset and separate by labels
    for data, label in test_dataset:
        if label not in label_datasets_test:
            label_datasets_test[label] = []
        label_datasets_test[label].append((data, label))

    label_data_loaders_test = {}
    for label, data_samples in label_datasets.items():
        label_data_loaders_test[label] = torch.utils.data.DataLoader(data_samples, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_clone_test = naiveModel.SimpleConvNet()
    model_clone_test.load_state_dict(copy.deepcopy(model.state_dict()))

    optimizer_test = optim.Adam(model_clone_test.parameters(), lr=0.001)
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model_clone = naiveModel.SimpleConvNet()
        model_clone.load_state_dict(copy.deepcopy(model.state_dict()))



        fmodel = fusion_model(model_clone)
        model.train()
        for i, (images, labels) in enumerate(label_data_loaders[1]):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_images = torch.mean(images,dim=0)
            final_image = fmodel(avg_images.unsqueeze(0), torch.mean(outputs, dim=0))

            result = model_clone_test(final_image)
            index = torch.argmax(result, dim=1)
            loss_test = criterion(result, torch.tensor([1]))
            loss_test.backward()
            optimizer_test.step()
            if index == labels[0]:
                print("result are correct")
            else:
                print("error")

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in label_data_loaders_test[1]:
                outputs = model(images)
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
        self.step_size = 1.0 / 255
        self.epsilon = 8.0 / 255
        self.num_steps = 100


    def forward(self, input, target):  # do forward in the module.py
        # if not args.attack :
        #    return self.basic_model(input), input

        x = input.detach()

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.mse_loss(logits, target, size_average=False)
                # print(loss)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() - self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)

            # x = torch.clamp(x, 0, 1)

        return x


if __name__ == '__main__':
    # pre_train()

    load_model()
    print(model)
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Size: {param.size()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params}")

    conv_params = list(model.conv1.parameters())

    # Calculate the total number of parameters in the convolutional layer
    total_conv_params = sum(p.numel() for p in conv_params)
    print(f"Total parameters in the convolutional layer: {total_conv_params}")

    fc1_params = list(model.fc1.parameters())

    # Calculate the total number of parameters in the convolutional layer
    total_fc1_params = sum(p.numel() for p in fc1_params)
    print(f"Total parameters in the fully connected layer: {total_fc1_params}")

    fusion_train()
