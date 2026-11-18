import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2 with optional Dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
            
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Multimodal PyTorch U-Net Architecture for Brain Tumor Segmentation.
    Supports variable input channels (e.g. 1, 2, 3, or 4 for ablation studies)
    and test-time Monte Carlo Dropout sampling.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
        dropout_rate: float = 0.2
    ):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Contracting Path
        curr_in = in_channels
        for feat in features:
            self.encoder.append(DoubleConv(curr_in, feat, dropout_rate=dropout_rate if feat >= 256 else 0.0))
            curr_in = feat

        # Bottleneck Layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_rate=dropout_rate)

        # Decoder Expanding Path
        for feat in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feat * 2, feat, dropout_rate=dropout_rate if feat >= 256 else 0.0))

        # Final 1x1 Conv Output Head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder Forward Pass
        for enc_block in self.encoder:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse Skip Connections for Decoder
        skip_connections = skip_connections[::-1]

        # Decoder Forward Pass
        for idx in range(0, len(self.decoder), 2):
            up_conv = self.decoder[idx]
            double_conv = self.decoder[idx + 1]

            x = up_conv(x)
            skip = skip_connections[idx // 2]

            # Match spatial dimensions if necessary due to odd padding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            # Concatenate along channel dimension
            concat_x = torch.cat((skip, x), dim=1)
            x = double_conv(concat_x)

        # Output Logits
        return self.final_conv(x)

    def enable_mc_dropout(self):
        """Enables dropout layers during inference for Monte Carlo Dropout uncertainty sampling."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
