"""
Módulo de visualização.

Contém funções responsáveis por gerar todos os gráficos exigidos pelo
projeto: curvas de treinamento, matriz de confusão por pixel e comparações
visuais entre imagem original, máscara real e predição do modelo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Backend não interativo, seguro para ambientes sem GUI.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plotar_curva_treinamento(
    valores_treino: Sequence[float],
    valores_validacao: Sequence[float],
    titulo: str,
    rotulo_y: str,
    caminho_saida: str,
) -> None:
    """Plota e salva a curva de uma métrica ao longo das épocas.

    Args:
        valores_treino: Sequência de valores da métrica no conjunto de treino.
        valores_validacao: Sequência de valores da métrica no conjunto de validação.
        titulo: Título do gráfico.
        rotulo_y: Rótulo do eixo Y.
        caminho_saida: Caminho completo (incluindo nome do arquivo) para salvar o PNG.
    """
    epocas = range(1, len(valores_treino) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, valores_treino, label="Treino", linewidth=2)
    plt.plot(epocas, valores_validacao, label="Validação", linewidth=2)
    plt.title(titulo)
    plt.xlabel("Época")
    plt.ylabel(rotulo_y)
    plt.legend()
    plt.grid(alpha=0.3)
    _salvar_figura(caminho_saida)


def plotar_curva_learning_rate(valores_lr: Sequence[float], caminho_saida: str) -> None:
    """Plota a evolução da taxa de aprendizado ao longo das épocas.

    Args:
        valores_lr: Sequência de valores de learning rate por época.
        caminho_saida: Caminho de saída para o arquivo PNG.
    """
    epocas = range(1, len(valores_lr) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, valores_lr, color="darkorange", linewidth=2)
    plt.title("Evolução da Learning Rate")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    _salvar_figura(caminho_saida)


def plotar_matriz_confusao_pixel(
    mascaras_reais: np.ndarray,
    mascaras_preditas: np.ndarray,
    caminho_saida: str,
) -> None:
    """Gera a matriz de confusão por pixel (fundo vs. ferida).

    Args:
        mascaras_reais: Array binário achatado com os rótulos verdadeiros.
        mascaras_preditas: Array binário achatado com as predições do modelo.
        caminho_saida: Caminho de saída para o arquivo PNG.
    """
    matriz = confusion_matrix(mascaras_reais.flatten(), mascaras_preditas.flatten())
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fundo", "Ferida"],
        yticklabels=["Fundo", "Ferida"],
    )
    plt.title("Matriz de Confusão por Pixel")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    _salvar_figura(caminho_saida)


def plotar_comparacao_visual(
    imagem: np.ndarray,
    mascara_real: np.ndarray,
    mascara_predita: np.ndarray,
    caminho_saida: str,
) -> None:
    """Gera uma figura comparando imagem original, máscara real e predição.

    Args:
        imagem: Imagem original em formato HWC (RGB).
        mascara_real: Máscara de segmentação verdadeira (HW).
        mascara_predita: Máscara de segmentação predita pelo modelo (HW).
        caminho_saida: Caminho de saída para o arquivo PNG.
    """
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5))
    eixos[0].imshow(imagem)
    eixos[0].set_title("Imagem Original")
    eixos[1].imshow(mascara_real, cmap="gray")
    eixos[1].set_title("Máscara Real")
    eixos[2].imshow(mascara_predita, cmap="gray")
    eixos[2].set_title("Predição do Modelo")
    for eixo in eixos:
        eixo.axis("off")
    _salvar_figura(caminho_saida)


def plotar_comparacao_modelos(
    nomes_modelos: Sequence[str],
    valores_metrica: Sequence[float],
    nome_metrica: str,
    caminho_saida: str,
) -> None:
    """Gera um gráfico de barras comparando uma métrica entre os quatro modelos.

    Args:
        nomes_modelos: Nomes das pipelines comparadas.
        valores_metrica: Valores finais da métrica para cada pipeline.
        nome_metrica: Nome da métrica exibida (ex.: "Dice Score").
        caminho_saida: Caminho de saída para o arquivo PNG.
    """
    plt.figure(figsize=(8, 5))
    cores = sns.color_palette("viridis", len(nomes_modelos))
    barras = plt.bar(nomes_modelos, valores_metrica, color=cores)
    plt.title(f"Comparação de {nome_metrica} entre Pipelines")
    plt.ylabel(nome_metrica)
    plt.xticks(rotation=15)
    for barra, valor in zip(barras, valores_metrica):
        plt.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.3f}", ha="center", va="bottom")
    _salvar_figura(caminho_saida)


def _salvar_figura(caminho_saida: str) -> None:
    """Cria o diretório de destino se necessário e salva a figura atual."""
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
