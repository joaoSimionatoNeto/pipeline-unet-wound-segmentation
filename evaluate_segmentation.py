"""Avaliação e inferência em lote da segmentação de feridas.

O módulo mantém o formato de checkpoint usado por ``training/trainer.py`` e
não altera o fluxo de treinamento. Pode ser usado como biblioteca ou CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from models.unet import UNet
from models.unet_resnet50 import UNetResNet50
from utils.model_utils import substituir_batchnorm_por_groupnorm
from utils.seed import detectar_dispositivo

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRICAS_LIMIAR = {"dice", "iou", "acuracia", "f1"}


@dataclass
class ResultadoAvaliacao:
    arquivo: str
    dice: float
    iou: float
    acuracia: float
    precisao: float
    recall: float
    especificidade: float
    f1: float
    verdadeiros_positivos: int
    verdadeiros_negativos: int
    falsos_positivos: int
    falsos_negativos: int
    area_manual_pixels: int
    area_predita_pixels: int
    erro: str = ""

    def como_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def calcular_metricas_mascara(predicao: np.ndarray, manual: np.ndarray, arquivo: str = "") -> ResultadoAvaliacao:
    """Calcula métricas binárias com convenções explícitas para casos vazios."""
    pred = np.asarray(predicao, dtype=bool)
    alvo = np.asarray(manual, dtype=bool)
    if pred.shape != alvo.shape:
        raise ValueError(f"Dimensões incompatíveis: predição {pred.shape}, manual {alvo.shape}.")
    tp = int(np.count_nonzero(pred & alvo))
    tn = int(np.count_nonzero(~pred & ~alvo))
    fp = int(np.count_nonzero(pred & ~alvo))
    fn = int(np.count_nonzero(~pred & alvo))
    total = pred.size

    def divisao(numerador: int, denominador: int, ambos_vazios: float = 0.0) -> float:
        return ambos_vazios if denominador == 0 else numerador / denominador

    uniao = tp + fp + fn
    precisao = divisao(tp, tp + fp, 1.0 if tp + fp == 0 and fn == 0 else 0.0)
    recall = divisao(tp, tp + fn, 1.0 if tp + fn == 0 and fp == 0 else 0.0)
    especificidade = divisao(tn, tn + fp, 1.0)
    dice = divisao(2 * tp, 2 * tp + fp + fn, 1.0 if uniao == 0 else 0.0)
    iou = divisao(tp, uniao, 1.0 if uniao == 0 else 0.0)
    acuracia = divisao(tp + tn, total, 0.0)
    f1 = divisao(2 * precisao * recall, precisao + recall, 0.0)
    return ResultadoAvaliacao(
        arquivo=arquivo, dice=dice, iou=iou, acuracia=acuracia,
        precisao=precisao, recall=recall, especificidade=especificidade, f1=f1,
        verdadeiros_positivos=tp, verdadeiros_negativos=tn,
        falsos_positivos=fp, falsos_negativos=fn,
        area_manual_pixels=int(np.count_nonzero(alvo)),
        area_predita_pixels=int(np.count_nonzero(pred)),
    )


def binarizar(probabilidades: np.ndarray, limiar: float) -> np.ndarray:
    """Converte probabilidades em máscara booleana usando ``>= limiar``."""
    if not 0.0 <= limiar <= 1.0:
        raise ValueError("O limiar deve estar entre 0 e 1.")
    return np.asarray(probabilidades) >= limiar


def calcular_escala_pixels_por_cm(pontos: np.ndarray) -> float | None:
    """Obtém a mediana da distância entre marcas consecutivas da régua."""
    pontos = np.asarray(pontos, dtype=np.float32)
    if pontos.ndim != 2 or pontos.shape[0] < 2 or pontos.shape[1] != 2:
        return None
    distancias = np.linalg.norm(np.diff(pontos, axis=0), axis=1)
    distancias = distancias[np.isfinite(distancias) & (distancias > 0)]
    return float(np.median(distancias)) if len(distancias) else None


def area_pixels_para_cm2(area_pixels: float, pixels_por_cm: float | None) -> float | None:
    """Converte uma área somente quando a escala foi detectada validamente."""
    if pixels_por_cm is None or not np.isfinite(pixels_por_cm) or pixels_por_cm <= 0:
        return None
    return float(area_pixels / (pixels_por_cm**2))


def listar_imagens(diretorio: str | Path) -> list[Path]:
    caminho = Path(diretorio)
    if not caminho.is_dir():
        raise FileNotFoundError(f"Diretório de imagens inexistente: {caminho}")
    return sorted(p for p in caminho.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM)


def localizar_mascara(caminho_imagem: Path, diretorio_mascaras: str | Path) -> Path:
    diretorio = Path(diretorio_mascaras)
    caminho = diretorio / caminho_imagem.name
    if caminho.exists():
        return caminho

    mascaras_com_mesmo_nome = sorted(diretorio.glob(f"{caminho_imagem.stem}.*"))
    if mascaras_com_mesmo_nome:
        return mascaras_com_mesmo_nome[0]

    raise FileNotFoundError(f"Máscara manual ausente para {caminho_imagem.name}: {diretorio}")


def filtrar_imagens_com_mascara(
    imagens: Iterable[Path], diretorio_mascaras: str | Path, logger: logging.Logger
) -> list[Path]:
    """Retorna somente imagens que possuem máscara manual correspondente."""
    imagens_com_mascara: list[Path] = []
    for imagem in imagens:
        try:
            localizar_mascara(imagem, diretorio_mascaras)
        except FileNotFoundError as erro:
            logger.warning("%s", erro)
            continue
        imagens_com_mascara.append(imagem)
    return imagens_com_mascara


def construir_modelo(arquitetura: str, tamanho_base: int = 64, profundidade: int = 4) -> torch.nn.Module:
    if arquitetura == "unet":
        modelo: torch.nn.Module = UNet(3, 1, tamanho_base, profundidade)
    elif arquitetura == "unet_resnet50":
        modelo = UNetResNet50(canais_saida=1, pretrained=False, fine_tuning=True)
    else:
        raise ValueError(f"Arquitetura desconhecida: {arquitetura}")
    return substituir_batchnorm_por_groupnorm(modelo)


def carregar_modelo(
    caminho_pesos: str | Path,
    arquitetura: str,
    dispositivo: torch.device,
    tamanho_base: int = 64,
    profundidade: int = 4,
) -> torch.nn.Module:
    caminho = Path(caminho_pesos)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de pesos inexistente: {caminho}")
    modelo = construir_modelo(arquitetura, tamanho_base, profundidade)
    checkpoint = torch.load(caminho, map_location=dispositivo)
    estado = checkpoint.get("modelo_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        modelo.load_state_dict(estado)
    except (RuntimeError, TypeError, AttributeError) as erro:
        raise ValueError(
            "Pesos incompatíveis com a arquitetura. Verifique --architecture, "
            "--base-channels e --depth."
        ) from erro
    return modelo.to(dispositivo).eval()


def _imagem_para_tensor(imagem: np.ndarray, tamanho: int) -> torch.Tensor:
    redimensionada = cv2.resize(imagem, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
    return torch.from_numpy((redimensionada.astype(np.float32) / 255.0).transpose(2, 0, 1).copy())


@torch.no_grad()
def predizer_probabilidade(modelo: torch.nn.Module, imagem: np.ndarray, tamanho: int, dispositivo: torch.device) -> np.ndarray:
    if imagem is None or imagem.ndim != 3:
        raise ValueError("Imagem inválida ou sem três canais.")
    tensor = _imagem_para_tensor(imagem, tamanho).unsqueeze(0).to(dispositivo)
    saida = modelo(tensor)
    if isinstance(saida, (tuple, list)):
        saida = saida[0]
    if saida.ndim != 4 or saida.shape[1] != 1:
        raise ValueError(f"Saída da rede incompatível: shape {tuple(saida.shape)}.")
    valores = saida[:, 0]
    if float(valores.detach().min()) < 0.0 or float(valores.detach().max()) > 1.0:
        valores = torch.sigmoid(valores)
    return cv2.resize(valores[0].cpu().numpy(), (imagem.shape[1], imagem.shape[0]), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def predizer_probabilidades(
    modelo: torch.nn.Module, imagens: Sequence[np.ndarray], tamanho: int, dispositivo: torch.device
) -> list[np.ndarray]:
    """Executa inferência em um batch e restaura cada máscara ao tamanho original."""
    if not imagens:
        return []
    tensor = torch.stack([_imagem_para_tensor(imagem, tamanho) for imagem in imagens]).to(dispositivo)
    saida = modelo(tensor)
    if isinstance(saida, (tuple, list)):
        saida = saida[0]
    valores = saida[:, 0]
    if float(valores.detach().min()) < 0.0 or float(valores.detach().max()) > 1.0:
        valores = torch.sigmoid(valores)
    return [
        cv2.resize(probabilidade.cpu().numpy(), (imagem.shape[1], imagem.shape[0]), interpolation=cv2.INTER_LINEAR)
        for probabilidade, imagem in zip(valores, imagens)
    ]


def sobrepor_mascaras(imagem_bgr: np.ndarray, manual: np.ndarray | None, predicao: np.ndarray) -> np.ndarray:
    """Gera uma visualização: verde=manual, vermelho=predição, amarelo=interseção."""
    base = imagem_bgr.copy()
    overlay = base.copy()
    if manual is not None:
        overlay[manual & ~predicao] = (0, 180, 0)
        overlay[manual & predicao] = (0, 220, 220)
    if manual is None:
        overlay[predicao] = (0, 0, 220)
    else:
        overlay[predicao & ~manual] = (0, 0, 220)
    return cv2.addWeighted(base, 0.55, overlay, 0.45, 0)


def _resumo(resultados: Sequence[ResultadoAvaliacao]) -> dict[str, object]:
    resumo: dict[str, object] = {"arquivo": "RESUMO"}
    campos = ("dice", "iou", "acuracia", "precisao", "recall", "especificidade", "f1")
    for campo in campos:
        valores = [
            valor for resultado in resultados
            if math.isfinite(valor := float(getattr(resultado, campo)))
        ]
        resumo[f"{campo}_media"] = statistics.mean(valores) if valores else float("nan")
        resumo[f"{campo}_mediana"] = statistics.median(valores) if valores else float("nan")
        resumo[f"{campo}_desvio_padrao"] = statistics.stdev(valores) if len(valores) > 1 else 0.0
    return resumo


def salvar_csv(resultados: Sequence[ResultadoAvaliacao], caminho: str | Path) -> None:
    linhas = [r.como_dict() for r in resultados]
    linhas.append(_resumo(resultados))
    chaves = list(dict.fromkeys(k for linha in linhas for k in linha))
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=chaves, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)


def buscar_melhor_limiar(
    probabilidades: Sequence[np.ndarray],
    manuais: Sequence[np.ndarray],
    inicio: float,
    fim: float,
    passo: float,
    metrica: str,
) -> tuple[float, list[dict[str, float]]]:
    if metrica not in METRICAS_LIMIAR or passo <= 0 or inicio > fim:
        raise ValueError("Intervalo/passo inválido ou métrica de limiar não suportada.")
    limiares = np.arange(inicio, fim + passo / 2, passo)
    tabela = []
    for limiar in limiares:
        resultados = [calcular_metricas_mascara(binarizar(p, float(limiar)), m) for p, m in zip(probabilidades, manuais)]
        media = float(np.mean([getattr(r, metrica) for r in resultados])) if resultados else float("nan")
        tabela.append({"threshold": float(limiar), metrica: media})
    if not tabela:
        raise ValueError("Nenhum limiar foi gerado.")
    melhor = max(tabela, key=lambda linha: linha[metrica])["threshold"]
    return melhor, tabela


def avaliar_imagens(
    imagens: Iterable[Path],
    diretorio_mascaras: str | Path,
    modelo: torch.nn.Module,
    dispositivo: torch.device,
    diretorio_saida: str | Path,
    tamanho: int,
    limiar: float,
    salvar_overlays: bool = True,
    batch_size: int = 1,
) -> list[ResultadoAvaliacao]:
    saida = Path(diretorio_saida)
    mascara_dir = saida / "predicted_masks"
    overlay_dir = saida / "overlays"
    mascara_dir.mkdir(parents=True, exist_ok=True)
    if salvar_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    resultados: list[ResultadoAvaliacao] = []
    logger = logging.getLogger("evaluate_segmentation")
    caminhos = list(imagens)
    for inicio in range(0, len(caminhos), max(1, batch_size)):
        lote = caminhos[inicio:inicio + max(1, batch_size)]
        imagens_validas: list[np.ndarray] = []
        dados_lote: list[tuple[Path, np.ndarray, np.ndarray]] = []
        for caminho in lote:
            try:
                imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
                if imagem is None:
                    raise ValueError("imagem corrompida ou formato não suportado")
                caminho_manual = localizar_mascara(caminho, diretorio_mascaras)
                manual = cv2.imread(str(caminho_manual), cv2.IMREAD_GRAYSCALE)
                if manual is None:
                    raise ValueError("máscara manual corrompida")
                if manual.shape != imagem.shape[:2]:
                    raise ValueError(f"dimensões diferentes entre imagem {imagem.shape[:2]} e máscara {manual.shape}")
                imagens_validas.append(imagem)
                dados_lote.append((caminho, imagem, manual > 127))
            except (OSError, ValueError, RuntimeError) as erro:
                logger.error("%s: %s", caminho.name, erro)
                resultados.append(ResultadoAvaliacao(caminho.name, *([float("nan")] * 7), *([0] * 6), str(erro)))
        try:
            probabilidades = predizer_probabilidades(modelo, imagens_validas, tamanho, dispositivo)
        except (RuntimeError, ValueError) as erro:
            for caminho, _, _ in dados_lote:
                logger.error("%s: %s", caminho.name, erro)
                resultados.append(ResultadoAvaliacao(caminho.name, *([float("nan")] * 7), *([0] * 6), str(erro)))
            continue
        for (caminho, imagem, manual), prob in zip(dados_lote, probabilidades):
            pred = binarizar(prob, limiar)
            resultado = calcular_metricas_mascara(pred, manual, caminho.name)
            cv2.imwrite(str(mascara_dir / caminho.name), pred.astype(np.uint8) * 255)
            if salvar_overlays:
                cv2.imwrite(str(overlay_dir / caminho.name), sobrepor_mascaras(imagem, manual, pred))
            resultados.append(resultado)
    return resultados


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--images_dir", type=Path, default=Path("data/validacao/imagens"))
    origem.add_argument("--image", type=Path)
    parser.add_argument("--masks_dir", "--mask-dir", type=Path, default=Path("data/validacao/mascaras"))
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("unet", "unet_resnet50"), default="unet_resnet50")
    parser.add_argument("--batch_size", type=int, default=1, help="Reservado para compatibilidade; a avaliação é segura por imagem.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_start", type=float)
    parser.add_argument("--threshold_end", type=float)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--threshold_metric", choices=sorted(METRICAS_LIMIAR), default="dice")
    parser.add_argument("--find_best_threshold", action="store_true")
    parser.add_argument("--no_overlays", action="store_true")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _argumentos()
    if args.image is None and args.images_dir is None:
        raise ValueError("Informe --images_dir ou --image.")
    if args.image is not None:
        imagens = [args.image]
        diretorio_mascaras = args.mask.parent if args.mask else args.masks_dir
    else:
        imagens = listar_imagens(args.images_dir)
        diretorio_mascaras = args.masks_dir
    if args.mask is not None and args.image is not None:
        diretorio_mascaras = args.mask.parent
        if args.mask.name != args.image.name:
            raise ValueError("A máscara individual deve ter o mesmo nome da imagem.")
    dispositivo = detectar_dispositivo(args.device)
    modelo = carregar_modelo(args.weights, args.architecture, dispositivo, args.base_channels, args.depth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(args.output_dir / "evaluation.log"), level=logging.INFO)
    logger = logging.getLogger("evaluate_segmentation")
    if args.image is None:
        imagens = filtrar_imagens_com_mascara(imagens, diretorio_mascaras, logger)
        if not imagens:
            raise RuntimeError(
                f"Nenhuma imagem com máscara correspondente foi encontrada em {diretorio_mascaras}."
            )

    limiar = args.threshold
    if args.find_best_threshold:
        probabilidades, manuais = [], []
        for imagem_path in imagens:
            imagem = cv2.imread(str(imagem_path), cv2.IMREAD_COLOR)
            manual_path = localizar_mascara(imagem_path, diretorio_mascaras)
            manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
            if imagem is None or manual is None or imagem.shape[:2] != manual.shape:
                continue
            probabilidades.append(predizer_probabilidade(modelo, imagem, args.image_size, dispositivo))
            manuais.append(manual > 127)
        limiar, tabela = buscar_melhor_limiar(
            probabilidades, manuais, args.threshold_start if args.threshold_start is not None else 0.1,
            args.threshold_end if args.threshold_end is not None else 0.9, args.threshold_step, args.threshold_metric,
        )
        (args.output_dir / "threshold_search").mkdir(exist_ok=True)
        threshold_path = args.output_dir / "threshold_search" / "threshold_results.csv"
        with open(threshold_path, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=["threshold", args.threshold_metric])
            escritor.writeheader()
            escritor.writerows(tabela)
        (args.output_dir / "best_threshold.json").write_text(
            json.dumps({"threshold": limiar, "metric": args.threshold_metric}, indent=2), encoding="utf-8"
        )
        print(f"Melhor limiar: {limiar:.4f}\nMétrica utilizada: {args.threshold_metric}")
    resultados = avaliar_imagens(
        imagens, diretorio_mascaras, modelo, dispositivo, args.output_dir, args.image_size,
        limiar, not args.no_overlays, args.batch_size,
    )
    salvar_csv(resultados, args.output_dir / "segmentation_results.csv")
    print(f"Processadas: {len(resultados)} imagens; resultados: {args.output_dir / 'segmentation_results.csv'}")


if __name__ == "__main__":
    main()
