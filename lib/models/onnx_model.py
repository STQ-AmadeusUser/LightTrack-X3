import torch
import numpy as np
import torch.nn as nn

from .backbone.backbone import ChildNet_FCN
from .neck.neck import BN_adj, PW_Corr_adj
from .head.head import head_subnet


class ONNXModel(nn.Module):
    def __init__(self):
        super(ONNXModel, self).__init__()
        self.features = ChildNet_FCN()
        self.neck = BN_adj(num_channel=96)
        self.feature_fusor = PW_Corr_adj(num_kernel=64, adj_channel=128)
        self.head = head_subnet()

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_corr = self.feature_fusor(zf, xf)
        # supernet head
        cls, reg = self.head(feat_corr)
        return cls, reg
