from torch import nn
from torchvision import models 
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, model_name="resnet50", pretrained=False, trainable=False, num_classes=None):
        super(ImageEncoder, self).__init__()
        
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = trainable
        
        if num_classes is not None:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model.fc = nn.Identity()
        
        
    def forward(self, x):
        return self.model(x)


    
