"""
Minimal extra_modules/block.py for the supplied YOLO YAML.

Retained modules:
- MKEConv
- Bottleneck_MKEC
- C3k_MKEC
- C3k2_MKEC
- DyDCNv2

The file intentionally removes unrelated experimental modules and their imports.
"""

import torch
import torch.nn as nn
from einops import rearrange

from ..modules.conv import Conv
from ..modules.block import Bottleneck, C3k, C3k2


try:
    from mmcv.cnn import build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
except ImportError:
    build_norm_layer = None
    ModulatedDeformConv2d = None


__all__ = (
    "MKEConv",
    "Bottleneck_MKEC",
    "C3k_MKEC",
    "C3k2_MKEC",
    "DyDCNv2",
)


_DEFAULT_DCN_NORM_CFG = {
    "type": "GN",
    "num_groups": 16,
    "requires_grad": True,
}


class DyDCNv2(nn.Module):
    """
    Modulated deformable convolution used by DTID.

    DTID predicts the offset and mask externally, then passes them
    into this module. GroupNorm is enabled by default, matching the original
    implementation. Pass norm_cfg=None to disable normalization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_cfg=_DEFAULT_DCN_NORM_CFG,
    ):
        super().__init__()

        if ModulatedDeformConv2d is None or build_norm_layer is None:
            raise ImportError(
                "DyDCNv2 requires MMCV with compiled mmcv.ops support. "
                "Install a compatible full MMCV build before using "
                "DTID."
            )

        if (
            norm_cfg is not None
            and norm_cfg.get("type") == "GN"
            and out_channels % norm_cfg.get("num_groups", 16) != 0
        ):
            raise ValueError(
                f"out_channels={out_channels} must be divisible by "
                f"num_groups={norm_cfg.get('num_groups', 16)}."
            )

        self.with_norm = norm_cfg is not None
        bias = not self.with_norm

        self.conv = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )

        if self.with_norm:
            self.norm = build_norm_layer(
                norm_cfg,
                out_channels,
            )[1]

    def forward(self, x, offset, mask):
        x = self.conv(
            x.contiguous(),
            offset,
            mask,
        )

        if self.with_norm:
            x = self.norm(x)

        return x


class MKEConv(nn.Module):
    """
    Efficient Multi-Scale Convolution Plus.

    The input channels are divided into four groups. Each group is processed
    by a different kernel size (1, 3, 5, 7), and the results are concatenated
    and fused by a 1x1 convolution.
    """

    def __init__(
        self,
        channel=256,
        kernels=(1, 3, 5, 7),
    ):
        super().__init__()

        self.groups = len(kernels)

        if self.groups < 1:
            raise ValueError("kernels must contain at least one kernel size.")

        if channel % self.groups != 0:
            raise ValueError(
                f"channel={channel} must be divisible by "
                f"groups={self.groups}."
            )

        min_ch = channel // self.groups

        if min_ch < 16:
            raise ValueError(
                f"Each MKEConv branch requires at least 16 channels. "
                f"Received channel={channel}, groups={self.groups}, "
                f"branch_channels={min_ch}."
            )

        self.convs = nn.ModuleList(
            [
                Conv(
                    c1=min_ch,
                    c2=min_ch,
                    k=kernel_size,
                )
                for kernel_size in kernels
            ]
        )

        self.conv_1x1 = Conv(
            c1=channel,
            c2=channel,
            k=1,
        )

    def forward(self, x):
        if x.shape[1] % self.groups != 0:
            raise ValueError(
                f"Input channels={x.shape[1]} must be divisible by "
                f"groups={self.groups}."
            )

        x_group = rearrange(
            x,
            "bs (g ch) h w -> bs ch h w g",
            g=self.groups,
        )

        x_convs = torch.stack(
            [
                conv(x_group[..., index])
                for index, conv in enumerate(self.convs)
            ],
            dim=0,
        )

        x_convs = rearrange(
            x_convs,
            "g bs ch h w -> bs (g ch) h w",
        )

        return self.conv_1x1(x_convs)


class Bottleneck_MKEC(Bottleneck):
    """
    Bottleneck whose second feature transformation is MKEConv.

    MKEConv preserves its channel count, so this block uses e=1.0.
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=1.0,
    ):
        if e != 1.0:
            raise ValueError(
                "Bottleneck_MKEC requires e=1.0 because MKEConv "
                "does not change the channel count."
            )

        super().__init__(
            c1,
            c2,
            shortcut,
            g,
            k,
            e,
        )

        self.cv1 = Conv(
            c1,
            c2,
            k[0],
            1,
        )
        self.cv2 = MKEConv(c2)


class C3k_MKEC(C3k):
    """C3k block composed of Bottleneck_MKEC blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        k=3,
    ):
        super().__init__(
            c1,
            c2,
            n,
            shortcut,
            g,
            e,
            k,
        )

        hidden_channels = int(c2 * e)

        self.m = nn.Sequential(
            *(
                Bottleneck_MKEC(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    g=g,
                    k=(k, k),
                    e=1.0,
                )
                for _ in range(n)
            )
        )


class C3k2_MKEC(C3k2):
    """
    YOLO11 C3k2 variant using MKEConv.

    YAML example:
        - [-1, 2, C3k2_MKEC, [1024, True]]
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        super().__init__(
            c1,
            c2,
            n,
            c3k,
            e,
            g,
            shortcut,
        )

        self.m = nn.ModuleList(
            [
                (
                    C3k_MKEC(
                        self.c,
                        self.c,
                        n=2,
                        shortcut=shortcut,
                        g=g,
                    )
                    if c3k
                    else Bottleneck_MKEC(
                        self.c,
                        self.c,
                        shortcut=shortcut,
                        g=g,
                        e=1.0,
                    )
                )
                for _ in range(n)
            ]
        )
