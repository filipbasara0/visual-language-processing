import torch.nn as nn
from torchvision.models import resnet50, resnet101


class ResNetFeatureExtractor(nn.Module):

    def __init__(self, feature_map_size, out_features_size):
        super(ResNetFeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])

        self.input_proj = nn.Conv2d(feature_map_size,
                                    out_features_size,
                                    kernel_size=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.input_proj(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        return x
