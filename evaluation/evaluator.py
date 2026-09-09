"""
Módulo de avaliação.

Implementa a classe `Evaluator`, responsável por avaliar um modelo treinado
no conjunto de teste, salvar métricas em CSV/JSON, gerar gráficos e produzir
comparações visuais entre imagem original, máscara real e predição.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import calcular_metricas
from utils.logger import criar_logger
from utils.visualization import plotar_comparacao_visual, plotar_matriz_confusao_pixel


class Evaluator:
    """Avalia um modelo de segmentação no conjunto de teste e consolida os resultados.

    Args:
        modelo: Modelo treinado (pesos já carregados).
        dispositivo: Dispositivo de processamento.
        nome_experimento: Identificador da pipeline avaliada.
        diretorio_resultados: Diretório raiz onde os resultados serão salvos.
        limiar_binarizacao: Limiar aplicado às probabilidades para gerar a máscara binária.
    """

    def __init__(
        self,
        modelo: nn.Module,
        dispositivo: torch.device,
        nome_experimento: str,
        diretorio_resultados: str = "results",
        limiar_binarizacao: float = 0.5,
    ) -> None:
        self.modelo = modelo.to(dispositivo).eval()
        self.dispositivo = dispositivo
        self.nome_experimento = nome_experimento
        self.limiar_binarizacao = limiar_binarizacao

        self.diretorio_saida = Path(diretorio_resultados) / nome_experimento
        self.diretorio_saida.mkdir(parents=True, exist_ok=True)

        self.logger = criar_logger(f"evaluator.{nome_experimento}", diretorio_logs=str(Path(diretorio_resultados) / "logs"))

    @torch.no_grad()
    def avaliar(self, loader_teste: DataLoader, salvar_predicoes: bool = True, maximo_visualizacoes: int = 12) -> Dict[str, float]:
        """Executa a avaliação completa no conjunto de teste.

        Args:
            loader_teste: DataLoader do conjunto de teste.
            salvar_predicoes: Se True, salva comparações visuais e matriz de confusão.
            maximo_visualizacoes: Número máximo de comparações visuais a salvar.

        Returns:
            Dicionário com as métricas médias calculadas no conjunto de teste.
        """
        registros_por_amostra: List[Dict[str, float]] = []
        todas_mascaras_reais = []
        todas_mascaras_preditas = []
        indice_visualizacao = 0

        for imagens, mascaras in loader_teste:
            imagens = imagens.to(self.dispositivo)
            mascaras = mascaras.to(self.dispositivo)

            logits = self.modelo(imagens)
            metricas = calcular_metricas(logits, mascaras, limiar=self.limiar_binarizacao)
            registros_por_amostra.append(metricas.como_dicionario())

            probabilidades = torch.sigmoid(logits)
            predicoes_binarias = (probabilidades > self.limiar_binarizacao).float()

            todas_mascaras_reais.append(mascaras.cpu().numpy().astype(np.uint8))
            todas_mascaras_preditas.append(predicoes_binarias.cpu().numpy().astype(np.uint8))

            if salvar_predicoes and indice_visualizacao < maximo_visualizacoes:
                indice_visualizacao = self._salvar_visualizacoes(
                    imagens, mascaras, predicoes_binarias, indice_visualizacao, maximo_visualizacoes
                )

        metricas_medias = pd.DataFrame(registros_por_amostra).mean().to_dict()
        self._salvar_metricas(registros_por_amostra, metricas_medias)

        if salvar_predicoes:
            mascaras_reais_concat = np.concatenate([m.flatten() for m in todas_mascaras_reais])
            mascaras_preditas_concat = np.concatenate([m.flatten() for m in todas_mascaras_preditas])
            plotar_matriz_confusao_pixel(
                mascaras_reais_concat, mascaras_preditas_concat, str(self.diretorio_saida / "matriz_confusao.png")
            )

        self.logger.info(f"Avaliação concluída para '{self.nome_experimento}': {metricas_medias}")
        return metricas_medias

    def _salvar_visualizacoes(
        self,
        imagens: torch.Tensor,
        mascaras_reais: torch.Tensor,
        mascaras_preditas: torch.Tensor,
        indice_atual: int,
        maximo: int,
    ) -> int:
        """Salva comparações visuais (imagem, máscara real, predição) para o batch atual."""
        for i in range(imagens.size(0)):
            if indice_atual >= maximo:
                break

            imagem_np = imagens[i].cpu().numpy().transpose(1, 2, 0)
            imagem_np = np.clip(imagem_np, 0, 1)
            mascara_real_np = mascaras_reais[i, 0].cpu().numpy()
            mascara_predita_np = mascaras_preditas[i, 0].cpu().numpy()

            caminho_saida = self.diretorio_saida / "comparacoes" / f"amostra_{indice_atual:03d}.png"
            plotar_comparacao_visual(imagem_np, mascara_real_np, mascara_predita_np, str(caminho_saida))
            indice_atual += 1

        return indice_atual

    def _salvar_metricas(self, registros_por_amostra: List[Dict[str, float]], metricas_medias: Dict[str, float]) -> None:
        """Persiste as métricas por amostra e as médias finais em CSV e JSON."""
        dataframe = pd.DataFrame(registros_por_amostra)
        dataframe.to_csv(self.diretorio_saida / "metricas_por_amostra.csv", index=False)

        with open(self.diretorio_saida / "metricas_finais.json", "w", encoding="utf-8") as arquivo:
            json.dump(metricas_medias, arquivo, indent=2, ensure_ascii=False)

        pd.DataFrame([metricas_medias]).to_csv(self.diretorio_saida / "metricas_finais.csv", index=False)


def gerar_relatorio_comparativo(resultados_por_pipeline: Dict[str, Dict[str, float]], diretorio_saida: str) -> pd.DataFrame:
    """Gera um relatório consolidado comparando todas as pipelines avaliadas.

    Args:
        resultados_por_pipeline: Dicionário {nome_pipeline: {metrica: valor}}.
        diretorio_saida: Diretório onde o relatório (CSV) será salvo.

    Returns:
        DataFrame consolidado com uma linha por pipeline.
    """
    tabela = pd.DataFrame(resultados_por_pipeline).T
    tabela.index.name = "pipeline"
    tabela = tabela.sort_values(by="dice", ascending=False)

    caminho = Path(diretorio_saida)
    caminho.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(caminho / "relatorio_comparativo_final.csv")

    with open(caminho / "relatorio_comparativo_final.json", "w", encoding="utf-8") as arquivo:
        json.dump(tabela.to_dict(orient="index"), arquivo, indent=2, ensure_ascii=False)

    return tabela
