import torch
import torch.nn as nn
from dinov2 import dinov2_vit_base_14, dinov2_vit_large_14, dinov2_vit_small_14
from dino_head import DINOHead

class DINOV2(nn.Module):
    def __init__(self, out_dim, nlayers=3, backbone='vit_base', **kwargs):
        super().__init__()

        if backbone == 'vit_base':
            self.backbone = dinov2_vit_base_14(**kwargs)
            self.in_dim = 768
        elif backbone == 'vit_large':
            self.backbone = dinov2_vit_large_14(**kwargs)
            self.in_dim = 1024
        elif backbone == 'vit_small':
            self.backbone = dinov2_vit_small_14(**kwargs)
            self.in_dim = 384

        self.head = DINOHead(in_dim=self.in_dim, out_dim=out_dim, nlayers=nlayers)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    net = DINOV2(5, backbone='vit_large')
    summary(net, input_size=(1, 3, 224, 224))