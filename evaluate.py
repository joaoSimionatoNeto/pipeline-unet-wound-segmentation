"""
Script principal de avaliação.

Carrega os checkpoints `_best.pth` de cada pipeline treinada, avalia no
conjunto de teste e gera o relatório comparativo final entre as quatro
abordagens de segmentação de feridas.

Uso:
    python evaluate.py --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from evaluation.evaluator import Evaluator, gerar_relatorio_comparativo
from train import NOMES_PIPELINES, carregar_configuracao, construir_datasets, construir_modelo
from utils.logger import criar_logger
from utils.seed import detectar_dispositivo, fixar_seed_global
from utils.visualization import plotar_comparacao_modelos

import torch


def avaliar_pipeline(nome_pipeline: str, config: Dict[str, Any]) -> Dict[str, float] | None:
    """Carrega o melhor checkpoint da pipeline e executa a avaliação no conjunto de teste."""
    logger = criar_logger(f"evaluate.{nome_pipeline}", config["caminhos"]["logs"])
    config_pipeline = config["pipelines"][nome_pipeline]

    if not config_pipeline.get("ativo", True):
        logger.info(f"Pipeline '{nome_pipeline}' desativada. Pulando avaliação.")
        return None

    caminho_checkpoint = Path(config["caminhos"]["checkpoints"]) / f"{nome_pipeline}_best.pth"
    if not caminho_checkpoint.exists():
        logger.warning(f"Checkpoint não encontrado para '{nome_pipeline}': {caminho_checkpoint}. Pulando.")
        return None

    dispositivo = detectar_dispositivo(config["projeto"]["dispositivo"])

    _, _, dataset_teste = construir_datasets(config, usar_opencv=config_pipeline["usar_opencv"])
    loader_teste = DataLoader(
        dataset_teste,
        batch_size=config["treinamento"]["batch_size"],
        shuffle=False,
        num_workers=config["treinamento"]["num_workers"],
    )

    modelo = construir_modelo(nome_pipeline, config_pipeline)
    checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])

    avaliador = Evaluator(
        modelo=modelo,
        dispositivo=dispositivo,
        nome_experimento=nome_pipeline,
        diretorio_resultados=config["caminhos"]["resultados"],
        limiar_binarizacao=config["avaliacao"]["limiar_binarizacao"],
    )

    metricas = avaliador.avaliar(loader_teste, salvar_predicoes=config["avaliacao"]["salvar_predicoes"])
    logger.info(f"Métricas finais de '{nome_pipeline}': {metricas}")
    return metricas


def main() -> None:
    """Ponto de entrada de linha de comando para avaliação e comparação final."""
    parser = argparse.ArgumentParser(description="Avaliação e comparação das pipelines de segmentação de feridas.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    argumentos = parser.parse_args()

    config = carregar_configuracao(argumentos.config)
    fixar_seed_global(config["projeto"]["seed"])

    resultados_por_pipeline: Dict[str, Dict[str, float]] = {}

    for nome_pipeline in NOMES_PIPELINES:
        metricas = avaliar_pipeline(nome_pipeline, config)
        if metricas is not None:
            resultados_por_pipeline[nome_pipeline] = metricas

    if not resultados_por_pipeline:
        print("Nenhuma pipeline avaliada. Verifique se os checkpoints existem.")
        return

    tabela_comparativa = gerar_relatorio_comparativo(resultados_por_pipeline, config["caminhos"]["resultados"])
    print("\n=== Relatório Comparativo Final ===")
    print(tabela_comparativa)

    plotar_comparacao_modelos(
        list(resultados_por_pipeline.keys()),
        [resultados_por_pipeline[p]["dice"] for p in resultados_por_pipeline],
        "Dice Score",
        str(Path(config["caminhos"]["resultados"]) / "comparacao_dice_final.png"),
    )
    plotar_comparacao_modelos(
        list(resultados_por_pipeline.keys()),
        [resultados_por_pipeline[p]["iou"] for p in resultados_por_pipeline],
        "IoU",
        str(Path(config["caminhos"]["resultados"]) / "comparacao_iou_final.png"),
    )


if __name__ == "__main__":
    main()
