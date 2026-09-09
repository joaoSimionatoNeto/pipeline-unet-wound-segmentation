"""
Funções de perda para segmentação binária.

Implementa Binary Cross Entropy, Dice Loss, combinação BCE+Dice, Focal Loss
e Tversky Loss, todas operando sobre logits (antes da sigmoide), com uma
fábrica (`obter_funcao_perda`) que seleciona a perda a partir da configuração.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Calcula a Dice Loss (1 - Dice Score) para segmentação binária."""

    def __init__(self, suavizacao: float = 1e-6) -> None:
        super().__init__()
        self.suavizacao = suavizacao

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        probabilidades = torch.sigmoid(logits)
        probabilidades = probabilidades.view(-1)
        alvo = alvo.view(-1)

        intersecao = (probabilidades * alvo).sum()
        dice = (2.0 * intersecao + self.suavizacao) / (probabilidades.sum() + alvo.sum() + self.suavizacao)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combinação ponderada entre Binary Cross Entropy e Dice Loss."""

    def __init__(self, peso_bce: float = 0.5, peso_dice: float = 0.5) -> None:
        super().__init__()
        self.peso_bce = peso_bce
        self.peso_dice = peso_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        return self.peso_bce * self.bce(logits, alvo) + self.peso_dice * self.dice(logits, alvo)


class FocalLoss(nn.Module):
    """Focal Loss, útil para lidar com o desbalanceamento entre ferida e fundo."""

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, alvo, reduction="none")
        probabilidades = torch.sigmoid(logits)
        pt = probabilidades * alvo + (1 - probabilidades) * (1 - alvo)
        peso_focal = self.alpha * (1 - pt).pow(self.gamma)
        return (peso_focal * bce).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss, generalização da Dice Loss que pondera falsos positivos e negativos."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, suavizacao: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.suavizacao = suavizacao

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        probabilidades = torch.sigmoid(logits).view(-1)
        alvo = alvo.view(-1)

        verdadeiros_positivos = (probabilidades * alvo).sum()
        falsos_positivos = ((1 - alvo) * probabilidades).sum()
        falsos_negativos = (alvo * (1 - probabilidades)).sum()

        indice_tversky = (verdadeiros_positivos + self.suavizacao) / (
            verdadeiros_positivos + self.alpha * falsos_positivos + self.beta * falsos_negativos + self.suavizacao
        )
        return 1.0 - indice_tversky


def obter_funcao_perda(nome_perda: str, config_perdas: dict | None = None) -> nn.Module:
    """Fábrica que retorna a função de perda configurada.

    Args:
        nome_perda: Um dos valores: "bce", "dice", "dice_bce", "focal", "tversky".
        config_perdas: Dicionário opcional com hiperparâmetros específicos
            (seção `perdas` do arquivo de configuração).

    Returns:
        Instância de `nn.Module` correspondente à perda selecionada.
    """
    config_perdas = config_perdas or {}
    nome_perda = nome_perda.lower()

    mapeamento = {
        "bce": lambda: nn.BCEWithLogitsLoss(),
        "dice": lambda: DiceLoss(),
        "dice_bce": lambda: DiceBCELoss(),
        "focal": lambda: FocalLoss(**config_perdas.get("focal", {})),
        "tversky": lambda: TverskyLoss(**config_perdas.get("tversky", {})),
    }

    if nome_perda not in mapeamento:
        raise ValueError(f"Função de perda desconhecida: '{nome_perda}'. Opções válidas: {list(mapeamento.keys())}")

    return mapeamento[nome_perda]()
