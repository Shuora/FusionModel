import torch

import torch.nn as nn

import torch.nn.functional as F



class CNN2D(nn.Module):

    """
    Standard 2D-CNN for traffic image classification (28x28).
    """

    def __init__(self, num_classes=10, input_channels=3):

        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        

                                                

        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.3)

        

    def forward(self, x):

                                        

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x



if __name__ == "__main__":

    model = CNN2D(num_classes=10)

    dummy_img = torch.randn(2, 3, 28, 28)

    output = model(dummy_img)

    print(f"Output shape: {output.shape}")

