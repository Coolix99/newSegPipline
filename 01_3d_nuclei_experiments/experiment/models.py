"""Eight comparable foreground+3D flow architectures.

All forward signatures: model(image[B,1,Z,Y,X], profile[B,8], features=None)
returns (foreground logits[B,2,Z,Y,X], unit flow[B,3,Z,Y,X]).
"""
import torch
from torch import nn
from torch.nn import functional as F

MODELS = (
    'N0_original', 'N1_residual', 'N2_anisotropic', 'N3_aniso_residual',
    'N4_xy_hybrid', 'N5_microsam_xy', 'N6_microsam_xyz', 'N7_cpdino_xy')
PRETRAINED = MODELS[-3:]


def group_norm(ch):
    return nn.GroupNorm(min(8, ch) if ch % min(8, ch) == 0 else 1, ch)


class Block(nn.Module):
    def __init__(self, ci, co, residual=False, aniso=False, batchnorm=False):
        super().__init__()
        kernel = (1, 3, 3) if aniso else 3
        padding = (0, 1, 1) if aniso else 1
        normal = (lambda c: nn.BatchNorm3d(c)) if batchnorm else group_norm
        self.net = nn.Sequential(nn.Conv3d(ci, co, kernel, padding=padding), normal(co), nn.ReLU(inplace=True),
                                 nn.Conv3d(co, co, kernel, padding=padding), normal(co))
        self.shortcut = nn.Conv3d(ci, co, 1) if residual and ci != co else nn.Identity()
        self.residual = residual

    def forward(self, x):
        y = self.net(x)
        return F.relu(y + self.shortcut(x) if self.residual else y, inplace=True)


class XY2D(nn.Module):
    """Shared 2D XY encoder, applied to each Z slice, trained end-to-end."""
    def __init__(self, channels=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())

    def forward(self, x):
        b, c, z, y, w = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(b*z, c, y, w)
        out = self.net(slices)
        return out.reshape(b, z, out.shape[1], y, w).permute(0, 2, 1, 3, 4)


class SegNet(nn.Module):
    def __init__(self, kind, base=16):
        super().__init__()
        if kind not in MODELS: raise ValueError(kind)
        self.kind = kind
        self.pretrained = kind in PRETRAINED
        self.xy = XY2D(8) if kind == 'N4_xy_hybrid' else None
        widths = (base, base*2, base*4, base*8)
        residual = kind in ('N1_residual', 'N3_aniso_residual') or self.pretrained
        anisotropic = kind in ('N2_anisotropic', 'N3_aniso_residual')
        c0, c1, c2, c3 = widths
        batchnorm = False  # exact legacy variant is implemented in legacy.py
        self.encoder = nn.ModuleList([
            Block(1 + (8 if self.xy else 0), c0, residual, aniso=anisotropic, batchnorm=batchnorm),
            Block(c0, c1, residual, aniso=anisotropic, batchnorm=batchnorm),
            Block(c1, c2, residual, batchnorm=batchnorm),
            Block(c2, c3, residual, batchnorm=batchnorm),
        ])
        self.pool0 = (1, 2, 2) if anisotropic else (2, 2, 2)
        self.context = nn.Linear(8, c3)
        # Pretrained low-res features fused at encoder scale 1/4; no giant upsampling.
        self.feature_project = nn.Conv3d(768 if kind == 'N7_cpdino_xy' else 256, c2, 1) if self.pretrained else None
        self.decoder = nn.ModuleList([
            Block(c3+c2, c2, residual, batchnorm=batchnorm),
            Block(c2+c1, c1, residual, batchnorm=batchnorm),
            Block(c1+c0, c0, residual, batchnorm=batchnorm),
        ])
        self.seg_head = nn.Conv3d(c0, 2, 1)
        self.flow_head = nn.Conv3d(c0, 3, 1)

    def forward(self, image, profile, features=None):
        if self.pretrained and features is None:
            raise ValueError(f'{self.kind} requires real cached pretrained features')
        if self.xy is not None:
            x = torch.cat([image, self.xy(image)], dim=1)
        else:
            x = image
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](F.max_pool3d(x0, self.pool0))
        x2 = self.encoder[2](F.max_pool3d(x1, 2))
        if self.feature_project is not None:
            feat = F.interpolate(features, size=x2.shape[2:], mode='trilinear', align_corners=False)
            x2 = x2 + self.feature_project(feat)
        x3 = self.encoder[3](F.max_pool3d(x2, 2))
        x3 = x3 + self.context(profile).reshape(profile.shape[0], -1, 1, 1, 1)
        up = x3
        for block, skip in zip(self.decoder, (x2, x1, x0)):
            up = F.interpolate(up, size=skip.shape[2:], mode='trilinear', align_corners=False)
            up = block(torch.cat([up, skip], dim=1))
        return self.seg_head(up), F.normalize(self.flow_head(up), p=2, dim=1, eps=1e-6)


def build_model(name, base=16):
    if name == 'N0_original':
        from .legacy import LegacyUNet3D
        return LegacyUNet3D()
    return SegNet(name, base=base)
