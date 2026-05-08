import torch

import torch.nn as nn



class LSTMClassifier(nn.Module):

    def __init__(self, num_classes=10, input_len=784, embed_dim=32, hidden_dim=128, num_layers=2):

        super().__init__()

        self.embedding = nn.Embedding(259, embed_dim)                                          

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        

    def forward(self, x):

                             

        x = self.embedding(x)                              

        lstm_out, _ = self.lstm(x)                                   

                                          

        pooled = torch.mean(lstm_out, dim=1)

        logits = self.fc(pooled)

        return logits



if __name__ == "__main__":

    model = LSTMClassifier(num_classes=10)

    dummy_input = torch.randint(0, 259, (2, 784))

    output = model(dummy_input)

    print(f"Output shape: {output.shape}")

