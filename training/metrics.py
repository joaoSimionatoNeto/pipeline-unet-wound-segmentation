"""
Métricas de avaliação para segmentação binária.

Implementa Dice Score, IoU, Precision, Recall, F1-Score e Acurácia,
calculadas a partir de logits do modelo e das máscaras verdadeiras.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricasSegmentacao:
    """Estrutura que armazena o resultado de todas as métricas calculadas."""

    dice: float
    iou: float
    precisao: float
    revocacao: float
    f1: float
    acuracia: float

    def como_dicionario(self) -> dict:
        """Converte a estrutura em um dicionário simples, útil para logging/CSV."""
        return {
            "dice": self.dice,
            "iou": self.iou,
            "precisao": self.precisao,
            "revocacao": self.revocacao,
            "f1": self.f1,
            "acuracia": self.acuracia,
        }


@torch.no_grad()
def calcular_metricas(
    logits: torch.Tensor,
    alvo: torch.Tensor,
    limiar: float = 0.5,
    suavizacao: float = 1e-6,
) -> MetricasSegmentacao:
    """Calcula o conjunto completo de métricas de segmentação para um batch.

    Args:
        logits: Saída bruta do modelo (antes da sigmoide), shape (N, 1, H, W).
        alvo: Máscara verdadeira binária, shape (N, 1, H, W).
        limiar: Limiar de binarização aplicado às probabilidades.
        suavizacao: Constante para evitar divisão por zero.

    Returns:
        Instância de `MetricasSegmentacao` com os valores médios do batch.
    """
    probabilidades = torch.sigmoid(logits)
    predicao_binaria = (probabilidades > limiar).float()

    predicao_flat = predicao_binaria.view(-1)
    alvo_flat = alvo.view(-1)

    verdadeiros_positivos = (predicao_flat * alvo_flat).sum()
    falsos_positivos = (predicao_flat * (1 - alvo_flat)).sum()
    falsos_negativos = ((1 - predicao_flat) * alvo_flat).sum()
    verdadeiros_negativos = ((1 - predicao_flat) * (1 - alvo_flat)).sum()

    dice = (2 * verdadeiros_positivos + suavizacao) / (
        2 * verdadeiros_positivos + falsos_positivos + falsos_negativos + suavizacao
    )
    uniao = verdadeiros_positivos + falsos_positivos + falsos_negativos
    iou = (verdadeiros_positivos + suavizacao) / (uniao + suavizacao)

    precisao = (verdadeiros_positivos + suavizacao) / (verdadeiros_positivos + falsos_positivos + suavizacao)
    revocacao = (verdadeiros_positivos + suavizacao) / (verdadeiros_positivos + falsos_negativos + suavizacao)
    f1 = (2 * precisao * revocacao) / (precisao + revocacao + suavizacao)

    total_pixels = verdadeiros_positivos + falsos_positivos + falsos_negativos + verdadeiros_negativos
    acuracia = (verdadeiros_positivos + verdadeiros_negativos + suavizacao) / (total_pixels + suavizacao)

    return MetricasSegmentacao(
        dice=dice.item(),
        iou=iou.item(),
        precisao=precisao.item(),
        revocacao=revocacao.item(),
        f1=f1.item(),
        acuracia=acuracia.item(),
    )
