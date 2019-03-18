import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch
from smooth_filter import smooth_filter
from wct import wct


class PhotoWCT(nn.Module):

    def __init__(self):
        super(PhotoWCT, self).__init__()

        self.encoder = VGGEncoder()
        self.decoders = nn.ModuleDict({i: VGGDecoder(i) for i in [1, 2, 3, 4]})

    def forward(self, content, style):
        """
        Arguments:
            content: a float tensor with shape [].
            style: a float tensor with shape [].
        """
        with torch.no_grad():

            style_features, _ = self.encoder(style)
            x = content

            for i in [1, 2, 3, 4]:
                features, pooling_indices = self.encoder(x, level=i)
                f = wct(features[i], style_features[i])
                x = self.decoders[i](f, pooling_indices)

        return x
