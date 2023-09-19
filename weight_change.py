import torch.nn as nn
import torch


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

def weight_loss(model):
    return sum([w.pow(2).sum() for w in model.parameters()])


model = SimpleNN()


def train_weight():
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):  # example with 100 epochs
        optimizer.zero_grad()

        loss = weight_loss(model)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')


if __name__ == '__main__':
    train_weight()