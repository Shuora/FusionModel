import torch

import torch.nn as nn

import sys

from pathlib import Path



                                                 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



from src.fusion_common import CharBERTTextEncoder



class CharBERTAblation(nn.Module):

    """
    CharBERT single-branch ablation model.
    Matches the time branch architecture in AttentionFusionModel.
    """

    def __init__(self, num_classes=10, char_hidden_size=128):

        super().__init__()

        

                                                        

        self.text_encoder = CharBERTTextEncoder(

            feature_dim=char_hidden_size,

            seq_len=784,

            hidden_size=char_hidden_size,

            num_layers=2,

            num_heads=4,

            dropout=0.3

        )

        

                                                                     

        self.classifier_head = nn.Sequential(

            nn.Linear(char_hidden_size, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes),

        )



    def forward(self, x):

                               

        feats = self.text_encoder(x)

        logits = self.classifier_head(feats)

        return logits



if __name__ == "__main__":

    model = CharBERTAblation(num_classes=10)

    dummy_input = torch.randint(0, 256, (2, 784))

    output = model(dummy_input)

    print(f"Output shape: {output.shape}")                    

