import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _kaiming_uniform(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)


def _gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


class KernelNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        kn_cfg    = cfg["model"]["kernel_net"]
        ch        = kn_cfg["channels"]    # [32, 64, 128]
        k         = kn_cfg["kernel_size"] # 5
        fc_hidden = kn_cfg["fc_hidden"]   # 512

        self.encoder = nn.Sequential(
            nn.Conv2d(3,     ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), nn.LeakyReLU(0.01), nn.AdaptiveAvgPool2d(1),
        )
        self.fc1 = nn.Linear(ch[2], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 3 * k * k)
        self._kernel_size = k
        self._init_weights(cfg)

    def _init_weights(self, cfg: dict):
        kn_cfg = cfg["model"]["kernel_net"]
        sigma  = kn_cfg["init"]["gaussian_sigma"]
        w_std  = kn_cfg["init"]["weight_std"]
        k      = kn_cfg["kernel_size"]

        self.encoder.apply(_kaiming_uniform)
        _kaiming_uniform(self.fc1)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=w_std)
        gauss = _gaussian_kernel_2d(k, sigma).flatten()
        bias  = np.tile(gauss, 3)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.from_numpy(bias))

    def forward(self, L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        f_L = self.encoder(L).flatten(1)
        f_R = self.encoder(R).flatten(1)
        f   = (f_L + f_R) / 2.0
        h   = F.leaky_relu(self.fc1(f), 0.01)
        k   = self.fc2(h)
        return k.view(k.size(0), 3, self._kernel_size, self._kernel_size)  # (B,3,5,5)


class DynamicFilter(nn.Module):
    def forward(self, x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k      = K.shape[-1]
        x_flat = x.reshape(1, B * C, H, W)
        k_flat = K.reshape(B * C, 1, k, k)
        y_flat = F.conv2d(x_flat, k_flat, padding=k // 2, groups=B * C)
        return y_flat.reshape(B, C, H, W)


class SiameseBackbone(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        bb_cfg  = cfg["model"]["backbone"]
        weights = "IMAGENET1K_V1" if bb_cfg["pretrained"] else None
        base    = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        if bb_cfg["freeze"]:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)  # (B, 512)


class GazeRegressor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1  = nn.Linear(1027, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, 3)
        self.apply(_kaiming_uniform)

    def forward(self, f_L: torch.Tensor, f_R: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_L, f_R, H], dim=1)
        h = self.drop(F.leaky_relu(self.fc1(x), 0.01))
        o = self.fc2(h)
        return F.normalize(o, dim=1)


class ProposedModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.kernel_net     = KernelNet(cfg)
        self.dynamic_filter = DynamicFilter()
        self.backbone       = SiameseBackbone(cfg)
        reg_cfg = cfg["model"]["regressor"]
        self.regressor = GazeRegressor(reg_cfg["hidden_dim"], reg_cfg["dropout"])

    def forward(self, L: torch.Tensor, R: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        K   = self.kernel_net(L, R)
        L_  = self.dynamic_filter(L, K)
        R_  = self.dynamic_filter(R, K)
        f_L = self.backbone(L_)
        f_R = self.backbone(R_)
        return self.regressor(f_L, f_R, H)


class BaselineModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone  = SiameseBackbone(cfg)
        reg_cfg = cfg["model"]["regressor"]
        self.regressor = GazeRegressor(hidden_dim=reg_cfg["hidden_dim"], dropout=reg_cfg["dropout"])

    def forward(self, L: torch.Tensor, R: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        f_L = self.backbone(L)
        f_R = self.backbone(R)
        return self.regressor(f_L, f_R, H)


def build_model(cfg: dict) -> nn.Module:
    variant = cfg["model"]["variant"]
    if variant == "proposed":
        return ProposedModel(cfg)
    if variant == "baseline":
        return BaselineModel(cfg)
    raise ValueError(f"알 수 없는 model.variant: {variant!r}")
