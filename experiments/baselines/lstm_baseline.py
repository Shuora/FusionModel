import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes=10, input_len=784, embed_dim=32, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(259, embed_dim) # 0-255 bytes + 256 PAD, 257 CLS, 258 SEP
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x) # (batch, seq_len, hidden_dim * 2)
        # Global average pooling over time
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits

if __name__ == "__main__":
    model = LSTMClassifier(num_classes=10)
    dummy_input = torch.randint(0, 259, (2, 784))
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
