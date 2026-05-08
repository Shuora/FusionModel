import torch

import torch.nn as nn

from transformers import MobileViTForImageClassification, MobileViTConfig

import logging



logger = logging.getLogger(__name__)



class MobileViTAblation(nn.Module):

    """
    MobileViT single-branch ablation model.
    Matches the space branch architecture in AttentionFusionModel.
    """

    def __init__(self, num_classes=10):

        super().__init__()

        

                                           

        mv_cfg = MobileViTConfig()

                                                                                   

        mobilevit_feature_dim = mv_cfg.neck_hidden_sizes[-1] if hasattr(mv_cfg, "neck_hidden_sizes") else 640

        mv_cfg.num_labels = mobilevit_feature_dim

        

        self.mobilevit = MobileViTForImageClassification(mv_cfg)

                                                  

        self.mobilevit.classifier = nn.Linear(mobilevit_feature_dim, mobilevit_feature_dim)



                                                                     

        self.classifier_head = nn.Sequential(

            nn.Linear(mobilevit_feature_dim, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes),

        )



    def forward(self, x):

                                        

                               

        if x.shape[1] == 1:

            x = x.repeat(1, 3, 1, 1)

            

                                        

                                                                         

        feats = self.mobilevit(x).logits

        

                                        

        logits = self.classifier_head(feats)

        return logits



if __name__ == "__main__":

    model = MobileViTAblation(num_classes=10)

    dummy_input = torch.randn(2, 3, 28, 28)

    output = model(dummy_input)

    print(f"Output shape: {output.shape}")                    

