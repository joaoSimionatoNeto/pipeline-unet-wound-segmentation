"""
U-Net clássica (Pipeline 2 e base da Pipeline 4).

Implementação própria e original da arquitetura U-Net descrita por
Ronneberger et al. (2015), construída inteiramente do zero, sem qualquer
backbone pré-treinado. Utilizada como baseline para comparação com a versão
que utiliza ResNet-50 como encoder.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BlocoConvolucionalDuplo(nn.Module):
    """Bloco básico da U-Net: duas convoluções 3x3 seguidas de BatchNorm e ReLU."""

    def __init__(self, canais_entrada: int, canais_saida: int) -> None:
        super().__init__()
        self.bloco = nn.Sequential(
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloco(x)


class CaminhoDescendente(nn.Module):
    """Etapa do encoder: max pooling seguido de bloco convolucional duplo."""

    def __init__(self, canais_entrada: int, canais_saida: int) -> None:
        super().__init__()
        self.caminho = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            BlocoConvolucionalDuplo(canais_entrada, canais_saida),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.caminho(x)


class CaminhoAscendente(nn.Module):
    """Etapa do decoder: upsampling, concatenação com skip connection e convolução."""

    def __init__(self, canais_entrada: int, canais_saida: int, usar_transposta: bool = True) -> None:
        super().__init__()
        if usar_transposta:
            self.upsample = nn.ConvTranspose2d(canais_entrada, canais_entrada // 2, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = BlocoConvolucionalDuplo(canais_entrada, canais_saida)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        diferenca_altura = skip_connection.size(2) - x.size(2)
        diferenca_largura = skip_connection.size(3) - x.size(3)
        x = nn.functional.pad(
            x,
            [
                diferenca_largura // 2,
                diferenca_largura - diferenca_largura // 2,
                diferenca_altura // 2,
                diferenca_altura - diferenca_altura // 2,
            ],
        )

        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Arquitetura U-Net clássica, construída do zero (sem transfer learning).

    Args:
        canais_entrada: Número de canais da imagem de entrada (3 para RGB).
        canais_saida: Número de classes de saída (1 para segmentação binária).
        canais_base: Número de canais na primeira camada do encoder.
        profundidade: Número de níveis de downsampling/upsampling.
    """

    def __init__(
        self,
        canais_entrada: int = 3,
        canais_saida: int = 1,
        canais_base: int = 64,
        profundidade: int = 4,
    ) -> None:
        super().__init__()

        self.entrada = BlocoConvolucionalDuplo(canais_entrada, canais_base)

        canais_por_nivel: List[int] = [canais_base * (2 ** i) for i in range(profundidade + 1)]

        self.encoder = nn.ModuleList(
            [CaminhoDescendente(canais_por_nivel[i], canais_por_nivel[i + 1]) for i in range(profundidade)]
        )

        self.decoder = nn.ModuleList(
            [
                CaminhoAscendente(canais_por_nivel[i + 1], canais_por_nivel[i])
                for i in reversed(range(profundidade))
            ]
        )

        self.saida = nn.Conv2d(canais_base, canais_saida, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        x = self.entrada(x)
        skip_connections.append(x)

        for etapa_encoder in self.encoder[:-1]:
            x = etapa_encoder(x)
            skip_connections.append(x)

        x = self.encoder[-1](x)

        for etapa_decoder, skip in zip(self.decoder, reversed(skip_connections)):
            x = etapa_decoder(x, skip)

        return self.saida(x)
