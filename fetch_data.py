import pandas as pd
from ucimlrepo import fetch_ucirepo
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        dataframe: Your data in a pandas DataFrame.
        transform: Optional transformations to apply to the features.
        """
        # Assuming the 'label' column contains numerical labels
        self.features = dataframe.drop('label', axis=1).values
        self.labels = dataframe['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        features_tensor = torch.tensor(features, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor
def fetch_scaler_data():
    # Example for reading a CSV file
    train_data = pd.read_csv('./data/image+segmentation/segmentation.data')
    test_data = pd.read_csv('./data/image+segmentation/segmentation.test')
    train_dataset = convert(test_data)
    test_dataset = convert(train_data)

    return train_dataset, test_dataset


def convert(df):
    labels = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
    label_to_num = {label: num for num, label in enumerate(labels)}

    # Assuming df is your DataFrame and the label is the index
    df = df.reset_index()


    # Now, the original index (your labels) is a regular column, typically named 'index'.
    # If you want to rename this new column, you can do:
    df = df.rename(columns={'index': 'label'})
    df['label'] = df['label'].map(label_to_num)



    dataset = CustomDataset(df)
    return dataset

def fetch_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

