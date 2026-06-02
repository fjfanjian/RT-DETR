"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from ...core import register


__all__ = ['HybridEncoder', 'SparseHybridEncoder', 'PassThroughEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def _project_features(self, feats):
        assert len(feats) == len(self.in_channels)
        return [self.input_proj[i](feat) for i, feat in enumerate(feats)]

    def _encode_projected_features(self, proj_feats):
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        return proj_feats

    def _fuse_projected_features(self, proj_feats):
        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs

    def forward(self, feats):
        proj_feats = self._project_features(feats)
        proj_feats = self._encode_projected_features(proj_feats)
        return self._fuse_projected_features(proj_feats)


@register()
class SparseHybridEncoder(HybridEncoder):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[256, 256, 256, 256],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v2',
                 saliency_levels=[2, 3],
                 saliency_weights=[0.7, 0.3],
                 anchor_level=2,
                 window_size=8,
                 window_padding=2,
                 active_window_ratio=0.45,
                 min_windows=4,
                 max_windows=16,
                 dense_fallback_ratio=0.65,
                 saliency_eps=1e-6,
                 saliency_detach=True,
                 force_dense=False):
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
        )
        self.saliency_levels = list(saliency_levels)
        self.saliency_weights = list(saliency_weights)
        self.anchor_level = anchor_level
        self.window_size = int(window_size)
        self.window_padding = int(window_padding)
        self.active_window_ratio = float(active_window_ratio)
        self.min_windows = int(min_windows)
        self.max_windows = int(max_windows)
        self.dense_fallback_ratio = float(dense_fallback_ratio)
        self.saliency_eps = saliency_eps
        self.saliency_detach = saliency_detach
        self.force_dense = force_dense
        self.last_saliency_map = None
        self.last_sparse_stats = None

        assert len(self.saliency_levels) > 0, 'saliency_levels must not be empty'
        assert len(self.saliency_levels) == len(self.saliency_weights), (
            'saliency_levels and saliency_weights must have the same length'
        )
        assert 0 <= self.anchor_level < len(self.in_channels), (
            f'anchor_level must be in [0, {len(self.in_channels) - 1}]'
        )
        assert self.window_size > 0, 'window_size must be positive'
        assert self.window_padding >= 0, 'window_padding must be non-negative'
        assert 0.0 < self.active_window_ratio <= 1.0, 'active_window_ratio must be in (0, 1]'
        assert self.min_windows >= 1, 'min_windows must be positive'
        assert self.max_windows >= self.min_windows, 'max_windows must be >= min_windows'
        assert 0.0 <= self.dense_fallback_ratio <= 1.0, 'dense_fallback_ratio must be in [0, 1]'

    def _activation_variance_saliency(self, feat):
        feat_for_saliency = feat.detach() if self.saliency_detach else feat
        saliency = feat_for_saliency.float().var(dim=1, unbiased=False, keepdim=True)
        return saliency.to(dtype=feat.dtype)

    def _normalize_saliency(self, saliency):
        flat = saliency.flatten(2)
        min_value = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        max_value = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        dynamic_range = max_value - min_value
        denom = dynamic_range.clamp_min(self.saliency_eps)

        normalized = (saliency - min_value) / denom
        normalized = torch.where(
            dynamic_range > self.saliency_eps,
            normalized,
            torch.zeros_like(normalized),
        )
        return normalized.clamp_(0.0, 1.0)

    def _extract_saliency_map(self, proj_feats):
        target_size = proj_feats[self.anchor_level].shape[-2:]
        weighted_maps = []

        for level, weight in zip(self.saliency_levels, self.saliency_weights):
            assert 0 <= level < len(proj_feats), (
                f'saliency level {level} is out of range for {len(proj_feats)} feature levels'
            )
            saliency = self._activation_variance_saliency(proj_feats[level])
            saliency = self._normalize_saliency(saliency)

            if saliency.shape[-2:] != target_size:
                saliency = F.interpolate(
                    saliency,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )

            weighted_maps.append(saliency * float(weight))

        saliency = torch.stack(weighted_maps, dim=0).sum(dim=0)
        return self._normalize_saliency(saliency)

    def _pad_saliency_for_windows(self, saliency):
        _, _, height, width = saliency.shape
        padded_height = int(math.ceil(height / self.window_size) * self.window_size)
        padded_width = int(math.ceil(width / self.window_size) * self.window_size)
        if padded_height == height and padded_width == width:
            return saliency, (height, width)

        pad_h = padded_height - height
        pad_w = padded_width - width
        padded = F.pad(saliency, (0, pad_w, 0, pad_h), mode='replicate')
        return padded, (height, width)

    def _select_active_windows(self, saliency):
        padded_saliency, original_size = self._pad_saliency_for_windows(saliency)
        pooled = F.avg_pool2d(padded_saliency, kernel_size=self.window_size, stride=self.window_size)
        batch_size, _, grid_h, grid_w = pooled.shape
        num_windows = grid_h * grid_w
        active_count = int(round(num_windows * self.active_window_ratio))
        active_count = max(self.min_windows, active_count)
        active_count = min(self.max_windows, active_count, num_windows)

        flat_scores = pooled.flatten(2)
        _, topk_idx = torch.topk(flat_scores, k=active_count, dim=-1)
        window_coords = []
        active_ratios = []

        for batch_idx in range(batch_size):
            coords = []
            for window_idx in topk_idx[batch_idx, 0].tolist():
                row = window_idx // grid_w
                col = window_idx % grid_w
                coords.append((row, col))
            window_coords.append(coords)
            active_ratios.append(len(coords) / float(num_windows))

        return window_coords, active_ratios, original_size

    def _anchor_to_level_bounds(self, anchor_start, anchor_end, level_idx, limit):
        anchor_stride = self.feat_strides[self.anchor_level]
        level_stride = self.feat_strides[level_idx]
        pixel_start = anchor_start * anchor_stride
        pixel_end = anchor_end * anchor_stride
        level_start = pixel_start // level_stride
        level_end = int(math.ceil(pixel_end / level_stride))
        level_start = max(0, min(level_start, limit))
        level_end = max(level_start + 1, min(level_end, limit))
        return level_start, level_end

    def _build_window_regions(self, anchor_hw, window_row, window_col, feat_shapes):
        anchor_h, anchor_w = anchor_hw
        anchor_y0 = window_row * self.window_size
        anchor_y1 = min(anchor_y0 + self.window_size, anchor_h)
        anchor_x0 = window_col * self.window_size
        anchor_x1 = min(anchor_x0 + self.window_size, anchor_w)

        crop_anchor_y0 = max(0, anchor_y0 - self.window_padding)
        crop_anchor_y1 = min(anchor_h, anchor_y1 + self.window_padding)
        crop_anchor_x0 = max(0, anchor_x0 - self.window_padding)
        crop_anchor_x1 = min(anchor_w, anchor_x1 + self.window_padding)

        crop_slices = []
        core_slices = []
        for level_idx, feat_shape in enumerate(feat_shapes):
            _, _, level_h, level_w = feat_shape
            crop_y0, crop_y1 = self._anchor_to_level_bounds(crop_anchor_y0, crop_anchor_y1, level_idx, level_h)
            crop_x0, crop_x1 = self._anchor_to_level_bounds(crop_anchor_x0, crop_anchor_x1, level_idx, level_w)
            out_y0, out_y1 = self._anchor_to_level_bounds(anchor_y0, anchor_y1, level_idx, level_h)
            out_x0, out_x1 = self._anchor_to_level_bounds(anchor_x0, anchor_x1, level_idx, level_w)

            crop_slices.append((crop_y0, crop_y1, crop_x0, crop_x1))
            core_slices.append((out_y0 - crop_y0, out_y1 - crop_y0, out_x0 - crop_x0, out_x1 - crop_x0))

        return crop_slices, core_slices

    def _scatter_sparse_windows(self, encoded_feats, base_outs, saliency):
        window_coords, active_ratios, anchor_hw = self._select_active_windows(saliency)
        feat_shapes = [feat.shape for feat in encoded_feats]
        dense_fallback = []
        dense_outs = None

        for batch_idx, coords in enumerate(window_coords):
            active_ratio = active_ratios[batch_idx]
            fallback_to_dense = self.force_dense or active_ratio >= self.dense_fallback_ratio
            dense_fallback.append(fallback_to_dense)
            if fallback_to_dense:
                if dense_outs is None:
                    dense_outs = self._fuse_projected_features([feat for feat in encoded_feats])
                for level_idx, dense_feat in enumerate(dense_outs):
                    base_outs[level_idx][batch_idx:batch_idx + 1] = dense_feat[batch_idx:batch_idx + 1]
                continue

            for window_row, window_col in coords:
                crop_slices, core_slices = self._build_window_regions(anchor_hw, window_row, window_col, feat_shapes)
                crop_feats = []
                for level_idx, feat in enumerate(encoded_feats):
                    y0, y1, x0, x1 = crop_slices[level_idx]
                    crop_feats.append(feat[batch_idx:batch_idx + 1, :, y0:y1, x0:x1])

                fused_crop_feats = self._fuse_projected_features(crop_feats)
                for level_idx, fused_crop in enumerate(fused_crop_feats):
                    out_y0, _, out_x0, _ = crop_slices[level_idx]
                    core_y0, core_y1, core_x0, core_x1 = core_slices[level_idx]
                    base_outs[level_idx][
                        batch_idx:batch_idx + 1,
                        :,
                        out_y0 + core_y0:out_y0 + core_y1,
                        out_x0 + core_x0:out_x0 + core_x1,
                    ] = fused_crop[:, :, core_y0:core_y1, core_x0:core_x1]

        return base_outs, active_ratios, dense_fallback

    def forward(self, feats):
        proj_feats = self._project_features(feats)
        self.last_saliency_map = self._extract_saliency_map(proj_feats)
        encoded_feats = self._encode_projected_features([feat for feat in proj_feats])
        if self.force_dense:
            outs = self._fuse_projected_features(encoded_feats)
            batch_size = outs[0].shape[0]
            self.last_sparse_stats = {
                'active_ratios': [1.0] * batch_size,
                'dense_fallback': [True] * batch_size,
            }
            return outs

        base_outs = [feat.clone() for feat in encoded_feats]
        outs, active_ratios, dense_fallback = self._scatter_sparse_windows(
            encoded_feats,
            base_outs,
            self.last_saliency_map,
        )
        self.last_sparse_stats = {
            'active_ratios': active_ratios,
            'dense_fallback': dense_fallback,
        }
        return outs


@register()
class PassThroughEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[256, 256, 256, 256],
                 feat_strides=[4, 8, 16, 32],
                 eval_spatial_size=None,
                 strict_shape=True):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.eval_spatial_size = eval_spatial_size
        self.strict_shape = strict_shape
        self.out_channels = list(in_channels)
        self.out_strides = list(feat_strides)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels), \
            f'Expected {len(self.in_channels)} features, but got {len(feats)}'

        base_height = None
        base_width = None
        base_stride = None
        outs = []
        for idx, (feat, expected_channels, stride) in enumerate(zip(feats, self.in_channels, self.feat_strides)):
            assert feat.ndim == 4, f'Expected 4D feature map, but got {feat.ndim}D at level {idx}'
            _, channels, height, width = feat.shape
            assert channels == expected_channels, \
                f'Expected {expected_channels} channels at level {idx}, but got {channels}'

            if idx == 0:
                base_height = height
                base_width = width
                base_stride = stride
            elif self.strict_shape:
                expected_height = base_height * base_stride // stride
                expected_width = base_width * base_stride // stride
                assert height == expected_height and width == expected_width, \
                    f'Expected feature shape {(expected_height, expected_width)} at level {idx}, but got {(height, width)}'

            outs.append(feat)

        return outs
