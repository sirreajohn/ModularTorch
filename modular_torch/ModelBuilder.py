import torch
import torch.nn as nn

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class TinyVgg(nn.Module):
    """Creates TinyVGG model. 
            see architecture at: https://poloclub.github.io/cnn-explainer/#article-relu
            
        args:
            in_size: int: input size for model.\n
            out_size: int: out_features size (typically num of classes.).\n
            hidden_units: int: number of feature maps to compute on each layer.
    """
    def __init__(self, in_size: int, out_size: int, hidden_units: int = 90) -> None:

        super().__init__()
        self.hidden_channels = hidden_units
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels = in_size, out_channels = self.hidden_channels , kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.hidden_channels , out_channels = self.hidden_channels , kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(in_channels = self.hidden_channels , out_channels = self.hidden_channels , kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.hidden_channels , out_channels = self.hidden_channels , kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 90*56*56, out_features = out_size)
        )
    
    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = self.linear_stack(x)
        return x
    


def get_vit_pretrained_model_with_custom_head(num_classes: int = 3):

    weights = ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = vit_b_16(weights = weights)


    ## Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    
    ## Custom Head
    model.heads = nn.Sequential(
        nn.Linear(in_features = 768, out_features = num_classes)
    )

    return model, transforms


def get_custom_effnet_b2(num_classes: int = 3):
    
    weights = EfficientNet_B2_Weights.DEFAULT
    eff_net_transforms = weights.transforms()

    eff_net_model = efficientnet_b2(weights = weights)

    ## freezing weights
    for params in eff_net_model.features.parameters():
        params.requires_grad = False

    ## adding custom head
    eff_net_model.classifier = nn.Sequential(
        nn.Dropout(p = 0.3, inplace = True),
        nn.Linear(in_features = 1408, out_features = num_classes)
    )

    return eff_net_model, eff_net_transforms