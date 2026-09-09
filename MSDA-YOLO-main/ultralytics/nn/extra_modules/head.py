import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import DFL
from ..modules.conv import autopad
from .block import DyDCNv2
from ultralytics.utils.tal import dist2bbox, make_anchors


__all__ = (
    "Scale",
    "Conv_GN",
    "TaskSpecificFeatureRecalibrationModule",
    "DTID",
)


class Scale(nn.Module):
    """Learnable scale factor for bounding-box regression."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Conv_GN(nn.Module):
    """Convolution followed by GroupNorm and activation."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
    ):
        super().__init__()

        if c2 % 16 != 0:
            raise ValueError(
                f"Conv_GN output channels must be divisible by 16, "
                f"but received c2={c2}."
            )

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.gn = nn.GroupNorm(16, c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class TaskSpecificFeatureRecalibrationModule(nn.Module):
    """Recalibrate task-specific features for classification or regression."""

    def __init__(
        self,
        feat_channels,
        stacked_convs,
        la_down_rate=8,
    ):
        super().__init__()

        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs

        hidden_channels = self.in_channels // la_down_rate
        if hidden_channels < 1:
            raise ValueError(
                f"Invalid attention hidden channels: {hidden_channels}."
            )

        self.la_conv1 = nn.Conv2d(
            self.in_channels,
            hidden_channels,
            1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(
            hidden_channels,
            self.stacked_convs,
            1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = Conv_GN(
            self.in_channels,
            self.feat_channels,
            1,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.la_conv1.weight, mean=0, std=0.001)
        nn.init.normal_(self.la_conv2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.la_conv2.bias)
        nn.init.normal_(
            self.reduction_conv.conv.weight,
            mean=0,
            std=0.01,
        )

    def forward(self, feat, avg_feat=None):
        b, _, h, w = feat.shape

        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        conv_weight = (
            weight.reshape(
                b,
                1,
                self.stacked_convs,
                1,
            )
            * self.reduction_conv.conv.weight.reshape(
                1,
                self.feat_channels,
                self.stacked_convs,
                self.feat_channels,
            )
        )

        conv_weight = conv_weight.reshape(
            b,
            self.feat_channels,
            self.in_channels,
        )

        feat = feat.reshape(
            b,
            self.in_channels,
            h * w,
        )

        feat = torch.bmm(
            conv_weight,
            feat,
        ).reshape(
            b,
            self.feat_channels,
            h,
            w,
        )

        feat = self.reduction_conv.gn(feat)
        feat = self.reduction_conv.act(feat)

        return feat


class DTID(nn.Module):
    """
    Task Dynamic Align Detection Head.

    YAML:
        - [[23, 27, 31], 1, DTID, [nc, 512]]
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(
        self,
        nc=80,
        hidc=256,
        ch=(),
    ):
        super().__init__()

        if not ch:
            raise ValueError(
                "DTID requires the input channel list ch."
            )

        if hidc % 32 != 0:
            raise ValueError(
                f"hidc must be divisible by 32, but received {hidc}."
            )

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # P3/P4/P5 channel alignment.
        self.channel_align = nn.ModuleList(
            [
                nn.Conv2d(
                    c,
                    hidc,
                    1,
                    bias=False,
                )
                for c in ch
            ]
        )

        # Shared feature extraction.
        self.share_conv = nn.Sequential(
            Conv_GN(
                hidc,
                hidc // 2,
                3,
            ),
            Conv_GN(
                hidc // 2,
                hidc // 2,
                3,
            ),
        )

        # Classification and regression task-specific feature recalibration.
        self.cls_decomp = TaskSpecificFeatureRecalibrationModule(
            hidc // 2,
            2,
            16,
        )
        self.reg_decomp = TaskSpecificFeatureRecalibrationModule(
            hidc // 2,
            2,
            16,
        )

        # Regression spatial alignment.
        self.DyDCNV2 = DyDCNv2(
            hidc // 2,
            hidc // 2,
        )
        self.spatial_conv_offset = nn.Conv2d(
            hidc,
            3 * 3 * 3,
            3,
            padding=1,
        )
        self.offset_dim = 2 * 3 * 3

        # Classification probability alignment.
        self.cls_prob_conv1 = nn.Conv2d(
            hidc,
            hidc // 4,
            1,
        )
        self.cls_prob_conv2 = nn.Conv2d(
            hidc // 4,
            1,
            3,
            padding=1,
        )

        # Output branches.
        self.cv2 = nn.Conv2d(
            hidc // 2,
            4 * self.reg_max,
            1,
        )
        self.cv3 = nn.Conv2d(
            hidc // 2,
            self.nc,
            1,
        )

        self.scale = nn.ModuleList(
            [
                Scale(1.0)
                for _ in ch
            ]
        )

        self.dfl = (
            DFL(self.reg_max)
            if self.reg_max > 1
            else nn.Identity()
        )

    def forward(self, x):
        if len(x) != self.nl:
            raise ValueError(
                f"DTID expected {self.nl} feature maps, "
                f"but received {len(x)}."
            )

        for i in range(self.nl):
            # Compatible with both new and legacy checkpoints.
            channel_align = getattr(self, "channel_align", None)

            if channel_align is not None:
                x_aligned = channel_align[i](x[i])
            else:
                # Legacy FLIR checkpoint has no channel_align.
                # Its original detection head directly consumed x[i].
                x_aligned = x[i]

            # Shared feature extraction and concatenation.
            if channel_align is not None:
                # New head: shared convolutions are connected sequentially.
                stack_res_list = [
                    self.share_conv[0](x_aligned)
                ]
                stack_res_list.extend(
                    module(stack_res_list[-1])
                    for module in self.share_conv[1:]
                )
            else:
                # Legacy FLIR checkpoint:
                # self.share_conv is organized per feature level:
                #   share_conv[0] handles P3
                #   share_conv[1] handles P4
                #   share_conv[2] handles P5
                # Each level contains a Sequential of shared convolutions.
                level_share_conv = self.share_conv[i]

                if not isinstance(
                    level_share_conv,
                    (nn.Sequential, nn.ModuleList),
                ):
                    raise RuntimeError(
                        "Legacy DTID expected share_conv[i] to be "
                        "a Sequential or ModuleList, but received "
                        f"{type(level_share_conv).__name__}."
                    )

                if len(level_share_conv) == 0:
                    raise RuntimeError(
                        f"Legacy DTID share_conv[{i}] is empty."
                    )

                stack_res_list = [
                    level_share_conv[0](x_aligned)
                ]
                stack_res_list.extend(
                    module(stack_res_list[-1])
                    for module in level_share_conv[1:]
                )

            feat = torch.cat(
                stack_res_list,
                dim=1,
            )

            # Task-specific feature recalibration.
            avg_feat = F.adaptive_avg_pool2d(
                feat,
                (1, 1),
            )
            cls_feat = self.cls_decomp(
                feat,
                avg_feat,
            )
            reg_feat = self.reg_decomp(
                feat,
                avg_feat,
            )

            # Regression spatial alignment.
            offset_and_mask = self.spatial_conv_offset(feat)

            offset = offset_and_mask[
                :,
                : self.offset_dim,
                :,
                :,
            ]
            mask = offset_and_mask[
                :,
                self.offset_dim :,
                :,
                :,
            ].sigmoid()

            reg_feat = self.DyDCNV2(
                reg_feat,
                offset,
                mask,
            )

            # Classification probability alignment.
            cls_prob = self.cls_prob_conv2(
                F.relu(
                    self.cls_prob_conv1(feat),
                    inplace=True,
                )
            ).sigmoid()

            box_output = self.scale[i](
                self.cv2(reg_feat)
            )
            cls_output = self.cv3(
                cls_feat * cls_prob
            )

            x[i] = torch.cat(
                (
                    box_output,
                    cls_output,
                ),
                dim=1,
            )

        if self.training:
            return x

        shape = x[0].shape

        x_cat = torch.cat(
            [
                feature.view(
                    shape[0],
                    self.no,
                    -1,
                )
                for feature in x
            ],
            dim=2,
        )

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                value.transpose(0, 1)
                for value in make_anchors(
                    x,
                    self.stride,
                    0.5,
                )
            )
            self.shape = shape

        if (
            self.export
            and self.format
            in (
                "saved_model",
                "pb",
                "tflite",
                "edgetpu",
                "tfjs",
            )
        ):
            box = x_cat[
                :,
                : self.reg_max * 4,
            ]
            cls = x_cat[
                :,
                self.reg_max * 4 :,
            ]
        else:
            box, cls = x_cat.split(
                (
                    self.reg_max * 4,
                    self.nc,
                ),
                dim=1,
            )

        dbox = self.decode_bboxes(box)

        if (
            self.export
            and self.format
            in (
                "tflite",
                "edgetpu",
            )
        ):
            img_h = shape[2]
            img_w = shape[3]

            img_size = torch.tensor(
                [
                    img_w,
                    img_h,
                    img_w,
                    img_h,
                ],
                device=box.device,
            ).reshape(
                1,
                4,
                1,
            )

            norm = self.strides / (
                self.stride[0]
                * img_size
            )

            dbox = dist2bbox(
                self.dfl(box) * norm,
                self.anchors.unsqueeze(0)
                * norm[:, :2],
                xywh=True,
                dim=1,
            )

        y = torch.cat(
            (
                dbox,
                cls.sigmoid(),
            ),
            dim=1,
        )

        return (
            y
            if self.export
            else (
                y,
                x,
            )
        )

    def bias_init(self):
        """Initialize detection output biases after stride setup."""

        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[
            : self.nc
        ] = math.log(
            5
            / self.nc
            / (640 / 16) ** 2
        )

    def decode_bboxes(self, bboxes):
        """Decode distributional bounding-box predictions."""

        return (
            dist2bbox(
                self.dfl(bboxes),
                self.anchors.unsqueeze(0),
                xywh=True,
                dim=1,
            )
            * self.strides
        )
