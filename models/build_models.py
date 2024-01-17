import torch
import torch.nn as nn
from dinov2 import dinov2_vit_base_14, dinov2_vit_large_14, dinov2_vit_small_14, dinov2_vit_giant2
from dino_head import DINOHead
from timm.models import register_model


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
        elif backbone == 'vit_giant':
            self.backbone = dinov2_vit_giant2(**kwargs)
            self.in_dim = 1536

        self.head = DINOHead(in_dim=self.in_dim, out_dim=out_dim, nlayers=nlayers)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(1)
        x = self.head(x)
        return x


@register_model
def DinoViT_samll(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return DINOV2(nlayers=3, backbone='vit_small', **kwargs)


@register_model
def DinoViT_base(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return DINOV2(nlayers=3, backbone='vit_base', **kwargs)


@register_model
def DinoViT_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return DINOV2(nlayers=3, backbone='vit_large', **kwargs)

@register_model
def DinoViT_giant(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return DINOV2(nlayers=3, backbone='vit_giant', **kwargs)


# if __name__ == '__main__':
#     from torchinfo import summary
#     net = DinoViT_base(out_dim=5)
#     summary(net, input_size=(1, 3, 224, 224))