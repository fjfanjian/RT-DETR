"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

DINOv3 full adapter for RT-DETRv2.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from ...core import register
from .dinov3_ms_deform_attn import MSDeformAttn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for height, width in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        reference_points_list.append(torch.stack((ref_x, ref_y), -1))

    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None]


def deform_inputs(x, patch_size):
    _, _, height, width = x.shape
    spatial_shapes = torch.as_tensor(
        [(height // 8, width // 8), (height // 16, width // 16), (height // 32, width // 32)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(height // patch_size, width // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(height // patch_size, width // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(height // 8, width // 8), (height // 16, width // 16), (height // 32, width // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs1, deform_inputs2


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, height, width):
        batch_size, num_tokens, channels = x.shape
        stride32_tokens = num_tokens // 21
        x1 = x[:, 0:16 * stride32_tokens, :].transpose(1, 2).reshape(batch_size, channels, height * 2, width * 2).contiguous()
        x2 = x[:, 16 * stride32_tokens:20 * stride32_tokens, :].transpose(1, 2).reshape(batch_size, channels, height, width).contiguous()
        x3 = x[:, 20 * stride32_tokens:, :].transpose(1, 2).reshape(batch_size, channels, height // 2, width // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        return torch.cat([x1, x2, x3], dim=1)


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path_prob=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, height, width):
        def _inner_forward(query_tensor, feat_tensor):
            attn = self.attn(
                self.query_norm(query_tensor),
                reference_points,
                self.feat_norm(feat_tensor),
                spatial_shapes,
                level_start_index,
                None,
            )
            query_tensor = query_tensor + attn
            if self.with_cffn:
                query_tensor = query_tensor + self.drop_path(self.ffn(self.ffn_norm(query_tensor), height, width))
            return query_tensor

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat)
        return _inner_forward(query, feat)


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path_prob=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path_prob=drop_path_prob,
            with_cp=with_cp,
        )
        self.extra_extractors = None
        if extra_extractor:
            self.extra_extractors = nn.ModuleList([
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    drop=drop,
                    drop_path_prob=drop_path_prob,
                    with_cp=with_cp,
                )
                for _ in range(2)
            ])

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, height_c, width_c, height_toks, width_toks):
        del cls, deform_inputs1, height_toks, width_toks
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            height=height_c,
            width=width_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    height=height_c,
                    width=width_c,
                )
        return x, c, None


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x_tensor):
            c1 = self.stem(x_tensor)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            batch_size, dim, _, _ = c1.shape
            c2 = c2.view(batch_size, dim, -1).transpose(1, 2)
            c3 = c3.view(batch_size, dim, -1).transpose(1, 2)
            c4 = c4.view(batch_size, dim, -1).transpose(1, 2)
            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


@register()
class DINOv3FPNAdapter(nn.Module):
    """Complete DINOv3 adapter with 4-scale outputs for RT-DETRv2."""

    def __init__(
        self,
        backbone,
        hidden_dim=256,
        interaction_indexes=None,
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        drop_path_rate=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.patch_size = backbone.patch_size if isinstance(backbone.patch_size, int) else backbone.patch_size[0]
        self.embed_dim = backbone.embed_dim
        self.pretrain_size = (pretrain_size, pretrain_size) if isinstance(pretrain_size, int) else tuple(pretrain_size)
        self.interaction_indexes = interaction_indexes or self._build_default_interaction_indexes()
        self.add_vit_feature = add_vit_feature

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        self.interactions = nn.ModuleList([
            InteractionBlockWithCls(
                dim=self.embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                drop_path_prob=drop_path_rate,
                extra_extractor=(idx == len(self.interaction_indexes) - 1 and use_extra_extractor),
                with_cp=with_cp,
            )
            for idx in range(len(self.interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(self.embed_dim)
        self.norm2 = nn.BatchNorm2d(self.embed_dim)
        self.norm3 = nn.BatchNorm2d(self.embed_dim)
        self.norm4 = nn.BatchNorm2d(self.embed_dim)
        self.output_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for _ in range(4)
        ])

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.output_proj.apply(self._init_weights)
        torch.nn.init.normal_(self.level_embed)

    def _build_default_interaction_indexes(self):
        depth = len(self.backbone.blocks)
        if depth < 4:
            raise ValueError(f"DINOv3 backbone depth must be >= 4, but got {depth}")
        return list(range(depth - 4, depth))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        height_c, width_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        height_toks, width_toks = height_c, width_c

        all_layers = self.backbone.get_intermediate_layers(
            x,
            n=self.interaction_indexes,
            return_class_token=True,
        )

        batch_size, _, dim = all_layers[0][0].shape
        outs = []
        for idx, layer in enumerate(self.interactions):
            patch_tokens, cls_token = all_layers[idx]
            _, c, _ = layer(
                patch_tokens,
                c,
                cls_token,
                deform_inputs1,
                deform_inputs2,
                height_c,
                width_c,
                height_toks,
                width_toks,
            )
            outs.append(patch_tokens.transpose(1, 2).reshape(batch_size, dim, height_toks, width_toks).contiguous())

        c2_size = c2.size(1)
        c3_size = c3.size(1)
        c2 = c[:, 0:c2_size, :]
        c3 = c[:, c2_size:c2_size + c3_size, :]
        c4 = c[:, c2_size + c3_size:, :]

        c2 = c2.transpose(1, 2).reshape(batch_size, dim, height_c * 2, width_c * 2).contiguous()
        c3 = c3.transpose(1, 2).reshape(batch_size, dim, height_c, width_c).contiguous()
        c4 = c4.transpose(1, 2).reshape(batch_size, dim, height_c // 2, width_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature and len(outs) >= 4:
            x1, x2, x3, x4 = outs[-4:]
            x1 = F.interpolate(x1, size=(4 * height_c, 4 * width_c), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=(2 * height_c, 2 * width_c), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=(height_c, width_c), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=(height_c // 2, width_c // 2), mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        features = [
            self.norm1(c1),
            self.norm2(c2),
            self.norm3(c3),
            self.norm4(c4),
        ]
        return [proj(feat) for proj, feat in zip(self.output_proj, features)]
