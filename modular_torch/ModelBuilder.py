import torch
import torch.nn as nn

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
