"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Pure PyTorch multi-scale deformable attention for the DINOv3 adapter.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ...zoo.rtdetr.utils import deformable_attention_core_func


def _is_power_of_2(value: int) -> bool:
    if (not isinstance(value, int)) or value <= 0:
        raise ValueError(f"invalid input for _is_power_of_2: {value} (type: {type(value)})")
    return (value & (value - 1) == 0) and value != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")

        d_per_head = d_model // n_heads
        if not _is_power_of_2(d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn so each attention head dimension is a power of 2."
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        del input_level_start_index

        batch_size, query_length, _ = query.shape
        _, input_length, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == input_length

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)

        value = value.view(
            batch_size,
            input_length,
            self.n_heads,
            int(self.ratio * self.d_model) // self.n_heads,
        )
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            batch_size,
            query_length,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )

        output = deformable_attention_core_func(value, input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)