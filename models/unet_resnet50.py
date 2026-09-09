"""
U-Net com backbone ResNet-50 (Pipeline 1 e base da Pipeline 3).

Implementação própria e original que utiliza a ResNet-50 pré-treinada
(ImageNet) da torchvision como encoder, conectada a um decoder no estilo
U-Net construído manualmente, com skip connections extraídas dos estágios
intermediários da ResNet. Suporta transfer learning (encoder congelado) e
fine-tuning (encoder treinável).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class BlocoDecoder(nn.Module):
    """Bloco de decodificação: upsampling + concatenação com skip + convolução dupla."""

    def __init__(self, canais_entrada: int, canais_skip: int, canais_saida: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(canais_entrada, canais_saida, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(canais_saida + canais_skip, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet50(nn.Module):
    """U-Net com encoder ResNet-50 pré-treinado, para segmentação binária de feridas.

    Args:
        canais_saida: Número de classes de saída (1 para segmentação binária).
        pretrained: Se True, carrega os pesos pré-treinados em ImageNet.
        fine_tuning: Se True, todos os parâmetros do encoder permanecem
            treináveis. Se False, o encoder é congelado (apenas o decoder
            é treinado — transfer learning "puro").
    """

    def __init__(self, canais_saida: int = 1, pretrained: bool = True, fine_tuning: bool = True) -> None:
        super().__init__()

        pesos = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=pesos)

        # Estágios da ResNet-50 utilizados como fontes de skip connections.
        self.estagio0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64 canais,  /2
        self.pool0 = backbone.maxpool                                              # /4
        self.estagio1 = backbone.layer1  # 256 canais,  /4
        self.estagio2 = backbone.layer2  # 512 canais,  /8
        self.estagio3 = backbone.layer3  # 1024 canais, /16
        self.estagio4 = backbone.layer4  # 2048 canais, /32

        if not fine_tuning:
            for parametro in self.parameters():
                parametro.requires_grad = False

        self.decoder4 = BlocoDecoder(canais_entrada=2048, canais_skip=1024, canais_saida=512)
        self.decoder3 = BlocoDecoder(canais_entrada=512, canais_skip=512, canais_saida=256)
        self.decoder2 = BlocoDecoder(canais_entrada=256, canais_skip=256, canais_saida=128)
        self.decoder1 = BlocoDecoder(canais_entrada=128, canais_skip=64, canais_saida=64)

        self.upsample_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.saida = nn.Conv2d(32, canais_saida, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.estagio0(x)          # /2,  64 canais
        x = self.pool0(skip0)             # /4
        skip1 = self.estagio1(x)          # /4,  256 canais
        skip2 = self.estagio2(skip1)      # /8,  512 canais
        skip3 = self.estagio3(skip2)      # /16, 1024 canais
        x = self.estagio4(skip3)          # /32, 2048 canais

        x = self.decoder4(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x, skip0)

        x = self.upsample_final(x)
        x = self.conv_final(x)
        return self.saida(x)
