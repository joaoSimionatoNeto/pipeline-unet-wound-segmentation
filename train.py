"""
Script principal de treinamento.

Orquestra o treinamento das quatro pipelines de segmentaÃ§Ã£o de feridas:
    1. U-Net + ResNet-50
    2. U-Net clÃ¡ssica
    3. OpenCV + U-Net + ResNet-50
    4. OpenCV + U-Net clÃ¡ssica

Uso:
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --pipeline unet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.augmentations import construir_pipeline_augmentation
from datasets.data_split import dividir_dados
from datasets.partitions import listar_pares
from datasets.wound_dataset import WoundDataset, listar_amostras_com_mascara
from models.unet import UNet
from models.unet_resnet50 import UNetResNet50
from preprocessing.opencv_pipeline import ConfiguracaoPreProcessamento
from training.losses import obter_funcao_perda
from training.trainer import Trainer
from utils.logger import criar_logger
from utils.model_utils import substituir_batchnorm_por_groupnorm
from utils.seed import determinar_batch_size_disponivel, detectar_dispositivo, fixar_seed_global
from utils.visualization import plotar_curva_learning_rate, plotar_curva_treinamento

NOMES_PIPELINES = ["unet", "unet_resnet50", "unet_opencv", "unet_resnet50_opencv"]


def carregar_configuracao(caminho_config: str) -> Dict[str, Any]:
    """Carrega o arquivo de configuraÃ§Ã£o YAML."""
    with open(caminho_config, "r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def construir_modelo(nome_pipeline: str, config_pipeline: Dict[str, Any], batch_size: int = 1) -> torch.nn.Module:
    """Instancia a arquitetura correspondente Ã  pipeline solicitada.

    Todas as camadas `BatchNorm2d` (inclusive as do encoder ResNet-50
    prÃ©-treinado, nas pipelines que o utilizam) sÃ£o substituÃ­das por
    `GroupNorm`. Isso Ã© necessÃ¡rio porque, com a acumulaÃ§Ã£o de gradientes
    processando uma imagem por vez, o batch efetivo por passo de GPU Ã© 1 â€”
    e as estatÃ­sticas de mÃ©dia/variÃ¢ncia do BatchNorm ficam instÃ¡veis e
    pouco representativas nesse regime. O GroupNorm normaliza por amostra
    e Ã© independente do tamanho do batch, tornando o treinamento estÃ¡vel.
    """
    if "resnet50" in nome_pipeline:
        modelo: torch.nn.Module = UNetResNet50(
            canais_saida=1,
            pretrained=config_pipeline.get("encoder_pretrained", True),
            fine_tuning=config_pipeline.get("fine_tuning", True),
            groupnorm_decoder=batch_size <= 2,
            congelar_batchnorm_encoder=batch_size <= 2,
        )
    else:
        modelo = UNet(
            canais_entrada=3,
            canais_saida=1,
            canais_base=config_pipeline.get("canais_base", 64),
            profundidade=config_pipeline.get("profundidade", 4),
        )

    return modelo if "resnet50" in nome_pipeline else substituir_batchnorm_por_groupnorm(modelo)


def construir_datasets(
    config: Dict[str, Any], usar_opencv: bool
) -> tuple[WoundDataset, WoundDataset, WoundDataset]:
    """Monta datasets das pastas de particao quando os tres splits existem."""
    paths = config["caminhos"]
    extension = config["dados"].get("extensao_imagem")
    split_keys = {
        "train": ("imagens_treino", "mascaras_treino"),
        "validation": ("imagens_validacao", "mascaras_validacao"),
        "test": ("imagens_teste", "mascaras_teste"),
    }
    direct_splits = all(
        paths.get(image_key) and paths.get(mask_key)
        and Path(paths[image_key]).is_dir() and Path(paths[mask_key]).is_dir()
        for image_key, mask_key in split_keys.values()
    )
    if direct_splits:
        (imagens_treino, mascaras_treino) = listar_pares(
            Path(paths["imagens_treino"]), Path(paths["mascaras_treino"]), extension
        )
        (imagens_val, mascaras_val) = listar_pares(
            Path(paths["imagens_validacao"]), Path(paths["mascaras_validacao"]), extension
        )
        (imagens_teste, mascaras_teste) = listar_pares(
            Path(paths["imagens_teste"]), Path(paths["mascaras_teste"]), extension
        )
        if not imagens_treino or not imagens_val or not imagens_teste:
            raise RuntimeError("Uma das pastas de treino, validacao ou teste nao contem pares imagem/mascara.")
    else:
        caminhos_imagens, caminhos_mascaras = listar_amostras_com_mascara(
            paths["imagens_treino"], paths["mascaras_treino"], extensao=extension
        )
        if not caminhos_imagens:
            raise RuntimeError("Nenhuma amostra de treino com mascara foi encontrada.")
        (imagens_treino, mascaras_treino), (imagens_val, mascaras_val), (imagens_teste, mascaras_teste) = dividir_dados(
            caminhos_imagens,
            caminhos_mascaras,
            proporcao_treino=config["dados"]["proporcao_treino"],
            proporcao_validacao=config["dados"]["proporcao_validacao"],
            proporcao_teste=config["dados"]["proporcao_teste"],
            seed=config["projeto"]["seed"],
        )
    configuracao_opencv = None
    if usar_opencv:
        configuracao_opencv = ConfiguracaoPreProcessamento.a_partir_de_dicionario(
            config["pre_processamento_opencv"]
        )

    tamanho_imagem = config["treinamento"]["image_size"]
    transformacoes_treino = construir_pipeline_augmentation(config["data_augmentation"])

    dataset_treino = WoundDataset(
        imagens_treino, mascaras_treino, tamanho_imagem, transformacoes_treino, usar_opencv, configuracao_opencv,
        normalizar_imagenet=not usar_opencv and config.get("modelo", {}).get("imagenet_normalizacao", False)
    )
    dataset_validacao = WoundDataset(
        imagens_val, mascaras_val, tamanho_imagem, None, usar_opencv, configuracao_opencv,
        normalizar_imagenet=not usar_opencv and config.get("modelo", {}).get("imagenet_normalizacao", False)
    )
    dataset_teste = WoundDataset(
        imagens_teste, mascaras_teste, tamanho_imagem, None, usar_opencv, configuracao_opencv,
        normalizar_imagenet=not usar_opencv and config.get("modelo", {}).get("imagenet_normalizacao", False)
    )

    return dataset_treino, dataset_validacao, dataset_teste


def executar_pipeline(nome_pipeline: str, config: Dict[str, Any], caminho_retomada: str | None = None) -> None:
    """Executa o ciclo completo de treinamento de uma pipeline especÃ­fica."""
    logger = criar_logger(f"train.{nome_pipeline}", config["caminhos"]["logs"])
    config_pipeline = config["pipelines"][nome_pipeline]

    if not config_pipeline.get("ativo", True):
        logger.info(f"Pipeline '{nome_pipeline}' estÃ¡ desativada na configuraÃ§Ã£o. Pulando.")
        return

    logger.info(f"=== Iniciando pipeline: {nome_pipeline} ===")

    dispositivo = detectar_dispositivo(config["projeto"]["dispositivo"])
    logger.info(f"Dispositivo selecionado: {dispositivo}")

    dataset_treino, dataset_validacao, _ = construir_datasets(config, usar_opencv=config_pipeline["usar_opencv"])

    batch_size_configurado = int(config["treinamento"]["batch_size"])
    batch_size = determinar_batch_size_disponivel(
        batch_size_configurado, dispositivo, int(config["treinamento"]["image_size"])
    )
    if batch_size != batch_size_configurado:
        logger.warning(
            f"Batch reduzido de {batch_size_configurado} para {batch_size} devido Ã  memÃ³ria livre da GPU."
        )
    num_workers = config["treinamento"]["num_workers"]

    pin_memory = dispositivo.type == "cuda"
    loader_treino = DataLoader(
        dataset_treino, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    loader_validacao = DataLoader(
        dataset_validacao, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    modelo = construir_modelo(nome_pipeline, config_pipeline, batch_size)
    funcao_perda = obter_funcao_perda(config["treinamento"]["loss"], config.get("perdas", {}))

    treinador = Trainer(
        modelo=modelo,
        funcao_perda=funcao_perda,
        dispositivo=dispositivo,
        config_treinamento=config["treinamento"],
        nome_experimento=nome_pipeline,
        diretorio_checkpoints=config["caminhos"]["checkpoints"],
        diretorio_logs=config["caminhos"]["logs"],
    )
    logger.info(
        f"Batch por passo de GPU: {batch_size} | Passos de acumulaÃ§Ã£o de gradiente: {treinador.accum_steps} | "
        f"Batch efetivo: {batch_size * treinador.accum_steps}"
    )

    diretorio_experimento = config_pipeline["diretorio_experimento"]
    treinador.salvar_configuracao_experimento(config, diretorio_experimento)

    epoca_inicial = 1
    if caminho_retomada:
        epoca_inicial = treinador.carregar_checkpoint(caminho_retomada)

    historico = treinador.treinar(loader_treino, loader_validacao, epoca_inicial=epoca_inicial)

    diretorio_graficos = Path(config["caminhos"]["resultados"]) / nome_pipeline
    plotar_curva_treinamento(
        historico.perda_treino, historico.perda_validacao, "Curva de Loss", "Loss",
        str(diretorio_graficos / "curva_loss.png"),
    )
    plotar_curva_treinamento(
        historico.dice_treino, historico.dice_validacao, "Curva de Dice Score", "Dice",
        str(diretorio_graficos / "curva_dice.png"),
    )
    plotar_curva_treinamento(
        historico.iou_treino, historico.iou_validacao, "Curva de IoU", "IoU",
        str(diretorio_graficos / "curva_iou.png"),
    )
    plotar_curva_learning_rate(historico.learning_rates, str(diretorio_graficos / "curva_learning_rate.png"))

    logger.info(f"=== Pipeline '{nome_pipeline}' concluÃ­da. ===")


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description="Treinamento das pipelines de segmentaÃ§Ã£o de feridas.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Caminho do arquivo YAML de configuraÃ§Ã£o.")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="todas",
        choices=NOMES_PIPELINES + ["todas"],
        help="Pipeline especÃ­fica a treinar, ou 'todas' para treinar as quatro sequencialmente.",
    )
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="Retoma a pipeline a partir do checkpoint *_last.pth.",
    )
    parser.add_argument(
        "--checkpoint-retomada",
        type=str,
        help="Caminho de um checkpoint especÃ­fico para retomada. Implica --retomar.",
    )
    argumentos = parser.parse_args()

    config = carregar_configuracao(argumentos.config)
    fixar_seed_global(config["projeto"]["seed"])

    pipelines_a_executar = NOMES_PIPELINES if argumentos.pipeline == "todas" else [argumentos.pipeline]

    for nome_pipeline in pipelines_a_executar:
        caminho_retomada = argumentos.checkpoint_retomada
        if argumentos.retomar and caminho_retomada is None:
            caminho_retomada = str(Path(config["caminhos"]["checkpoints"]) / f"{nome_pipeline}_last.pth")
        executar_pipeline(nome_pipeline, config, caminho_retomada=caminho_retomada)


if __name__ == "__main__":
    main()


