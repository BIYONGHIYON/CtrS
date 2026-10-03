"""Grouped spectral SSA-MRN with RGB-guided 12/17-feature variants, K=4."""

import torch
from torch import nn
from torch.nn import functional as F

from .rgb_hsi import RGBLatentCore, RGBHSISsaMRN
from .interp23 import interp23tap, Interp23


# ---------------------------------------------------------------------
# RGB04 original grouped core: 12 spectral features
# ---------------------------------------------------------------------
class RGBGroupedCore(RGBLatentCore):
    def __init__(self):
        super().__init__(12, 4, vectorized=True)

        for blocks in (
            self.SSA_blocks,
            self.SSA_blocks1,
            self.SSA_blocks2,
        ):
            for block in blocks:
                block.conv1t6 = nn.Conv2d(1, 4, 3, padding=1)

        # RGB04: internal 23-tap upsampling
        for name, ratio in [
            ("upsample1", 4),
            ("upsample100", 2),
            ("upsample101", 2),
            ("upsample102", 2),
        ]:
            setattr(self, name, Interp23(ratio))

        # RGB04: internal average-pooling downsampling
        for name, ratio in [
            ("downsample1", 4),
            ("downsample2", 4),
            ("downsample100", 2),
            ("downsample200", 2),
        ]:
            setattr(self, name, nn.AvgPool2d(ratio))

    def _attend(
        self,
        pan,
        features,
        ms,
        blocks,
        fusion,
        activation,
    ):
        n, _, h, w = pan.shape

        guide = F.conv2d(
            pan,
            torch.cat([b.conv1t6.weight for b in blocks]),
            torch.cat([b.conv1t6.bias for b in blocks]),
            padding=1,
            groups=3,
        ).relu()

        guide = F.conv2d(
            guide,
            torch.cat([b.conv6t6.weight for b in blocks]),
            torch.cat([b.conv6t6.bias for b in blocks]),
            padding=1,
            groups=12,
        )

        inputs = torch.cat(
            [
                features[:, None].expand(-1, 12, -1, -1, -1),
                ms[:, :, None],
            ],
            dim=2,
        ).reshape(n, 156, h, w)

        projected = F.conv2d(
            inputs,
            torch.cat([b.conv7t6_3.weight for b in blocks]),
            torch.cat([b.conv7t6_3.bias for b in blocks]),
            padding=1,
            groups=12,
        )

        g = guide.reshape(n, 48, h * w)

        attention = (
            g
            * projected.transpose(-1, -2).reshape(
                n,
                48,
                h * w,
            )
        ).reshape(n, 48, h, w)

        attention = (
            attention.transpose(-1, -2)
            .reshape(n, 48, h * w)
            .softmax(-1)
        )

        return activation(
            fusion(
                (g * attention).reshape(
                    n,
                    48,
                    h,
                    w,
                )
            )
        )


# ---------------------------------------------------------------------
# Original grouped single-core model (RGB04 compatibility)
# ---------------------------------------------------------------------
class RGBGroupedSsaMRN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Conv2d(
            204,
            12,
            1,
            groups=12,
        )

        self.core = RGBGroupedCore()

        self.decoder = nn.Conv2d(
            12,
            204,
            1,
            groups=12,
        )

        nn.init.normal_(
            self.decoder.weight,
            std=1e-3,
        )
        nn.init.zeros_(
            self.decoder.bias
        )

    def forward(
        self,
        rgb,
        lr_hsi,
    ):
        if (
            rgb.shape[1] != 3
            or lr_hsi.shape[1] != 204
            or rgb.shape[-2:]
            != tuple(
                s * 4
                for s in lr_hsi.shape[-2:]
            )
        ):
            raise ValueError(
                "Expected RGB HR and 204-band HSI "
                "at x4 spatial ratio"
            )

        latent = self.encoder(lr_hsi)

        residual = self.core(
            rgb,
            interp23tap(latent, 4),
            latent,
        )

        return (
            interp23tap(lr_hsi, 4)
            + self.decoder(residual)
        )


# ---------------------------------------------------------------------
# RGB04: one scalar RGB channel -> 12 features
# ---------------------------------------------------------------------
class ScalarGrouped12Core(RGBGroupedCore):
    """
    One scalar RGB guide drives all twelve spectral branches.
    """

    def __init__(self):
        super().__init__()

        self.guide_channels = 1

        self.conv1t1 = nn.Conv2d(
            1,
            1,
            3,
            padding=1,
        )

        # 12 latent features + 2 auxiliary channels
        self.cov2t64 = nn.Conv2d(
            14,
            64,
            3,
            padding=1,
        )

    def _attend(
        self,
        guide,
        features,
        ms,
        blocks,
        fusion,
        activation,
    ):
        # RGB04 behavior:
        # expand one scalar guide to three identical guide inputs
        return super()._attend(
            guide.expand(
                -1,
                3,
                -1,
                -1,
            ),
            features,
            ms,
            blocks,
            fusion,
            activation,
        )


# ---------------------------------------------------------------------
# RGB06: one scalar RGB channel -> 17 features
# Only feature count differs from RGB04.
# 204 bands / 17 features = 12 bands per feature.
# ---------------------------------------------------------------------
class ScalarGrouped17Core(RGBLatentCore):
    """
    One scalar RGB guide drives all seventeen spectral branches.
    K=4 and RGB04 resampling are preserved.
    """

    def __init__(self):
        super().__init__(
            17,
            4,
            vectorized=True,
        )

        self.guide_channels = 1

        self.conv1t1 = nn.Conv2d(
            1,
            1,
            3,
            padding=1,
        )

        # 17 latent features + 2 auxiliary channels
        self.cov2t64 = nn.Conv2d(
            19,
            64,
            3,
            padding=1,
        )

        # Each of the 17 SSA branches receives one scalar guide.
        for blocks in (
            self.SSA_blocks,
            self.SSA_blocks1,
            self.SSA_blocks2,
        ):
            for block in blocks:
                block.conv1t6 = nn.Conv2d(
                    1,
                    4,
                    3,
                    padding=1,
                )

        # Keep RGB04 23-tap upsampling.
        for name, ratio in [
            ("upsample1", 4),
            ("upsample100", 2),
            ("upsample101", 2),
            ("upsample102", 2),
        ]:
            setattr(
                self,
                name,
                Interp23(ratio),
            )

        # Keep RGB04 average-pooling downsampling.
        for name, ratio in [
            ("downsample1", 4),
            ("downsample2", 4),
            ("downsample100", 2),
            ("downsample200", 2),
        ]:
            setattr(
                self,
                name,
                nn.AvgPool2d(ratio),
            )

    def _attend(
        self,
        guide,
        features,
        ms,
        blocks,
        fusion,
        activation,
    ):
        n, _, h, w = guide.shape

        # 17 branches × K=4 = 68 guide channels
        guide_features = F.conv2d(
            guide,
            torch.cat(
                [
                    b.conv1t6.weight
                    for b in blocks
                ]
            ),
            torch.cat(
                [
                    b.conv1t6.bias
                    for b in blocks
                ]
            ),
            padding=1,
        ).relu()

        guide_features = F.conv2d(
            guide_features,
            torch.cat(
                [
                    b.conv6t6.weight
                    for b in blocks
                ]
            ),
            torch.cat(
                [
                    b.conv6t6.bias
                    for b in blocks
                ]
            ),
            padding=1,
            groups=17,
        )

        # For each branch:
        # 17 shared latent features + 1 branch-specific feature
        # = 18 channels
        #
        # 17 groups × 18 channels = 306 total input channels
        inputs = torch.cat(
            [
                features[:, None].expand(
                    -1,
                    17,
                    -1,
                    -1,
                    -1,
                ),
                ms[:, :, None],
            ],
            dim=2,
        ).reshape(
            n,
            17 * 18,
            h,
            w,
        )

        projected = F.conv2d(
            inputs,
            torch.cat(
                [
                    b.conv7t6_3.weight
                    for b in blocks
                ]
            ),
            torch.cat(
                [
                    b.conv7t6_3.bias
                    for b in blocks
                ]
            ),
            padding=1,
            groups=17,
        )

        # 17 branches × K=4 = 68
        g = guide_features.reshape(
            n,
            17 * 4,
            h * w,
        )

        attention = (
            g
            * projected.transpose(
                -1,
                -2,
            ).reshape(
                n,
                17 * 4,
                h * w,
            )
        ).reshape(
            n,
            17 * 4,
            h,
            w,
        )

        attention = (
            attention.transpose(
                -1,
                -2,
            )
            .reshape(
                n,
                17 * 4,
                h * w,
            )
            .softmax(-1)
        )

        return activation(
            fusion(
                (
                    g * attention
                ).reshape(
                    n,
                    17 * 4,
                    h,
                    w,
                )
            )
        )


# ---------------------------------------------------------------------
# RGB04 / RGB06 triple RGB model
# ---------------------------------------------------------------------
class RGBTripleGroupedSsaMRN(nn.Module):
    """
    R/G/B independently reconstruct all 204 bands
    and are combined by learned band-wise fusion.

    RGB04: latent_channels=12
    RGB06: latent_channels=17
    """

    def __init__(
        self,
        latent_channels=12,
    ):
        super().__init__()

        if latent_channels not in (
            12,
            17,
        ):
            raise ValueError(
                "Triple grouped model supports "
                "12 or 17 features"
            )

        self.latent_channels = (
            latent_channels
        )

        self.encoder = nn.Conv2d(
            204,
            latent_channels,
            1,
            groups=latent_channels,
        )

        if latent_channels == 12:
            core_class = (
                ScalarGrouped12Core
            )
        else:
            core_class = (
                ScalarGrouped17Core
            )

        self.cores = nn.ModuleList(
            [
                core_class()
                for _ in range(3)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                nn.Conv2d(
                    latent_channels,
                    204,
                    1,
                    groups=latent_channels,
                )
                for _ in range(3)
            ]
        )

        # R/G/B correction for each HSI band
        self.fusion = nn.Conv2d(
            204 * 3,
            204,
            1,
            groups=204,
        )

        for decoder in self.decoders:
            nn.init.normal_(
                decoder.weight,
                std=1e-3,
            )
            nn.init.zeros_(
                decoder.bias
            )

        # Start from equal RGB contribution.
        nn.init.constant_(
            self.fusion.weight,
            1 / 3,
        )
        nn.init.zeros_(
            self.fusion.bias
        )

    def forward(
        self,
        rgb,
        lr_hsi,
    ):
        if (
            rgb.ndim != 4
            or lr_hsi.ndim != 4
        ):
            raise ValueError(
                "Expected NCHW RGB and HSI"
            )

        if (
            rgb.shape[1] != 3
            or lr_hsi.shape[1] != 204
            or rgb.shape[0]
            != lr_hsi.shape[0]
        ):
            raise ValueError(
                "Expected matching batches, "
                "RGB=3 and HSI=204"
            )

        if (
            rgb.shape[-2:]
            != tuple(
                s * 4
                for s
                in lr_hsi.shape[-2:]
            )
        ):
            raise ValueError(
                "Expected x4 spatial ratio"
            )

        latent = self.encoder(
            lr_hsi
        )

        up = interp23tap(
            latent,
            4,
        )

        corrections = [
            decoder(
                core(
                    rgb[:, i : i + 1],
                    up,
                    latent,
                )
            )
            for i, (
                core,
                decoder,
            ) in enumerate(
                zip(
                    self.cores,
                    self.decoders,
                )
            )
        ]

        # Interleave:
        # band0-R, band0-G, band0-B,
        # band1-R, band1-G, band1-B, ...
        n, c, h, w = (
            corrections[0].shape
        )

        per_band = torch.stack(
            corrections,
            dim=2,
        ).reshape(
            n,
            c * 3,
            h,
            w,
        )

        return (
            interp23tap(
                lr_hsi,
                4,
            )
            + self.fusion(
                per_band
            )
        )


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------
def build_rgb_hsi_model(config):
    kind = config.get(
        "model_type",
        "latent_rgb",
    )

    if kind in (
        "rgb_grouped12_23tap",
        "rgb_triple_grouped12_23tap",
        "rgb_triple_grouped17_23tap",
    ):
        if kind == (
            "rgb_triple_grouped17_23tap"
        ):
            expected_channels = 17
        else:
            expected_channels = 12

        if (
            config["latent_channels"]
            != expected_channels
            or config["ssai_dimension"]
            != 4
        ):
            raise ValueError(
                "Grouped model requires "
                f"{expected_channels} features "
                "and K=4"
            )

        if (
            config.get("upsampler")
            != "23tap"
        ):
            raise ValueError(
                "Grouped model requires "
                "upsampler=23tap"
            )

        if kind == (
            "rgb_grouped12_23tap"
        ):
            return RGBGroupedSsaMRN()

        return RGBTripleGroupedSsaMRN(
            latent_channels=expected_channels
        )

    if kind != "latent_rgb":
        raise ValueError(
            "Unknown model_type: "
            + kind
        )

    return RGBHSISsaMRN(
        204,
        config["latent_channels"],
        config["ssai_dimension"],
        config.get(
            "vectorized_ssa",
            True,
        ),
    )