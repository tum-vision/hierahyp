import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer



from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule
from mmcv.runner import force_fp32

from mmseg.ops import resize
from ..losses import accuracy

import psutil
import time

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, norm_cfg, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                build_norm_layer(norm_cfg, dim_in)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
            
    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)
    
class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPContrastHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.
    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPContrastHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0

        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.proj_head = ProjectionHead(dim_in=2048, norm_cfg=self.norm_cfg)
        self.register_buffer("step", torch.zeros(1))

    def forward(self, inputs, offsets, normals, curvature):
        """Forward function."""
        ball = gt.PoincareBall(c=1.0)
        self.step+=1
        embedding = self.proj_head(inputs[-1])
        embedding_euclidian = embedding

        embedding = embedding.permute((0,2,3,1))
        embedding = ball.projx(embedding)
        embedding = embedding.permute((0,3,1,2))

        x = self._transform_inputs(inputs)

        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

      
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
           c1_output = self.c1_bottleneck(inputs[0])
           output = resize(
               input=output,
               size=c1_output.shape[2:],
               mode='bilinear',
               align_corners=self.align_corners)
           output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)


        if self.hyperbolic:
            output = output.permute((0,2,3,1))
            out_proj = ball.expmap0(output)        
            output = self.embedding_space.run_log_torch(out_proj,offsets,normals, 1.0)
            output = output.permute((0,3,1,2))
        else:
            output = self.cls_seg(output)
 
        return output
    
