"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

DINOv3 backbone with the full adapter implementation.
"""

import torch
import torch.nn as nn

from ...core import register

from .dinov3_model import DINOv3Model
from .dinov3_adapter import DINOv3FPNAdapter


@register()
class DINOv3Backbone(nn.Module):
    """DINOv3 backbone with full adapter interaction and 4-scale outputs."""

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained_path: str = None,
        layers_to_use: int = 4,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        interaction_indexes=None,
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 12,
        drop_path_rate: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 1.0,
        add_vit_feature: bool = True,
        use_extra_extractor: bool = True,
        with_cp: bool = False,
    ):
        super().__init__()

        # DINOv3 骨干网络
        self.dinov3 = DINOv3Model(
            name=model_name,
            pretrained_path=pretrained_path,
            layers_to_use=layers_to_use,
            freeze_backbone=freeze_backbone,
        )

        self.adapter = DINOv3FPNAdapter(
            backbone=self.dinov3.backbone,
            hidden_dim=hidden_dim,
            interaction_indexes=interaction_indexes,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            with_cp=with_cp,
        )

        self.strides = [4, 8, 16, 32]
        self.channels = [hidden_dim, hidden_dim, hidden_dim, hidden_dim]

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            outputs: 特征列表 List[Tensor], 每层 [B, 256, H/4, H/8, H/16, H/32]
        """
        return self.adapter(x)


if __name__ == '__main__':
    # 测试代码
    backbone = DINOv3Backbone(
        model_name='dinov3_vitb16',
        layers_to_use=4,
        freeze_backbone=True,
        hidden_dim=256
    )

    data = torch.rand(1, 3, 640, 640)

    backbone.eval()
    with torch.no_grad():
        outputs = backbone(data)

    for i, output in enumerate(outputs):
        stride = [4, 8, 16, 32][i]
        print(f"Output {i} (stride={stride}): {output.shape}")
