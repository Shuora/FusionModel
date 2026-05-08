import torch

import torch.nn as nn

import torch.nn.functional as F



class DeepPacket(nn.Module):

    """
    DeepPacket: A Novel Deep Learning Framework for Packet-based Traffic Classification.
    Actually implementing a 1D-CNN architecture common in traffic classification.
    """

    def __init__(self, num_classes=10, input_len=784):

        super().__init__()

                                      

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool1d(2)

        

                                                      

        self.fc_input_dim = 64 * (input_len // 4)

        self.fc1 = nn.Linear(self.fc_input_dim, 512)

        self.fc2 = nn.Linear(512, num_classes)

        

    def forward(self, x):

                                                      

        x = x.float().unsqueeze(1)                        

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x



if __name__ == "__main__":

    model = DeepPacket(num_classes=10, input_len=784)

    dummy_input = torch.randint(0, 256, (2, 784))

    output = model(dummy_input)

    print(f"Output shape: {output.shape}")

