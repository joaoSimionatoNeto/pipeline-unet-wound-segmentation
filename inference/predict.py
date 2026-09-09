"""
Script de inferência.

Carrega um checkpoint treinado (qualquer uma das quatro pipelines) e gera a
máscara de segmentação predita para uma nova imagem de ferida.

Exemplo de uso via linha de comando:

    python inference/predict.py \
        --imagem caminho/para/imagem.png \
        --checkpoint checkpoints/unet_resnet50_best.pth \
        --arquitetura unet_resnet50 \
        --saida resultado_mascara.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.unet import UNet
from models.unet_resnet50 import UNetResNet50
from preprocessing.opencv_pipeline import ConfiguracaoPreProcessamento, PipelineOpenCV
from utils.seed import detectar_dispositivo


def construir_modelo(arquitetura: str) -> torch.nn.Module:
    """Instancia a arquitetura solicitada, pronta para receber os pesos do checkpoint.

    Args:
        arquitetura: Um dos valores: "unet", "unet_resnet50".

    Returns:
        Instância do modelo (não treinado, pesos serão carregados em seguida).
    """
    if arquitetura == "unet_resnet50":
        return UNetResNet50(canais_saida=1, pretrained=False, fine_tuning=True)
    if arquitetura == "unet":
        return UNet(canais_entrada=3, canais_saida=1)
    raise ValueError(f"Arquitetura desconhecida: '{arquitetura}'.")


def predizer_mascara(
    caminho_imagem: str,
    caminho_checkpoint: str,
    arquitetura: str,
    tamanho_imagem: int = 512,
    usar_opencv: bool = False,
    configuracao_opencv: ConfiguracaoPreProcessamento | None = None,
    limiar: float = 0.5,
) -> np.ndarray:
    """Executa a inferência completa e retorna a máscara binária predita.

    Args:
        caminho_imagem: Caminho para a imagem de entrada.
        caminho_checkpoint: Caminho para o arquivo `.pth` do modelo treinado.
        arquitetura: Arquitetura correspondente ao checkpoint ("unet" ou "unet_resnet50").
        tamanho_imagem: Tamanho para redimensionamento da imagem de entrada.
        usar_opencv: Se True, aplica o pipeline de pré-processamento OpenCV.
        configuracao_opencv: Configuração do pipeline OpenCV (obrigatória se usar_opencv=True).
        limiar: Limiar de binarização aplicado à saída da rede.

    Returns:
        Máscara binária (uint8, valores 0 ou 255) no tamanho original da imagem.
    """
    dispositivo = detectar_dispositivo("auto")

    modelo = construir_modelo(arquitetura)
    checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])
    modelo.to(dispositivo).eval()

    imagem_original = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
    if imagem_original is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")

    altura_original, largura_original = imagem_original.shape[:2]

    if usar_opencv:
        if configuracao_opencv is None:
            raise ValueError("configuracao_opencv é obrigatória quando usar_opencv=True.")
        imagem_processada = PipelineOpenCV(configuracao_opencv).processar(imagem_original)
        if imagem_processada.dtype != np.uint8:
            imagem_processada = np.clip(imagem_processada * 255.0, 0, 255).astype(np.uint8)
    else:
        imagem_processada = cv2.resize(imagem_original, (tamanho_imagem, tamanho_imagem))

    imagem_float = imagem_processada.astype(np.float32) / 255.0
    tensor_imagem = torch.from_numpy(imagem_float.transpose(2, 0, 1)).unsqueeze(0).float().to(dispositivo)

    with torch.no_grad():
        logits = modelo(tensor_imagem)
        probabilidades = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mascara_binaria = (probabilidades > limiar).astype(np.uint8) * 255
    mascara_redimensionada = cv2.resize(
        mascara_binaria, (largura_original, altura_original), interpolation=cv2.INTER_NEAREST
    )
    return mascara_redimensionada


def main() -> None:
    """Ponto de entrada de linha de comando para inferência."""
    parser = argparse.ArgumentParser(description="Inferência de segmentação de feridas.")
    parser.add_argument("--imagem", required=True, help="Caminho da imagem de entrada.")
    parser.add_argument("--checkpoint", required=True, help="Caminho do checkpoint (.pth).")
    parser.add_argument("--arquitetura", required=True, choices=["unet", "unet_resnet50"])
    parser.add_argument("--saida", required=True, help="Caminho de saída para a máscara predita.")
    parser.add_argument("--usar-opencv", action="store_true", help="Ativa o pré-processamento OpenCV.")
    parser.add_argument("--tamanho-imagem", type=int, default=512)
    parser.add_argument("--limiar", type=float, default=0.5)
    argumentos = parser.parse_args()

    configuracao_opencv = None
    if argumentos.usar_opencv:
        configuracao_opencv = ConfiguracaoPreProcessamento.a_partir_de_dicionario(
            {
                "redimensionar": {"ativo": True, "tamanho": [argumentos.tamanho_imagem, argumentos.tamanho_imagem]},
                "clahe": {"ativo": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
                "remocao_ruido_gaussiano": {"ativo": True, "kernel_size": 5},
                "correcao_iluminacao": {"ativo": True},
            }
        )

    mascara = predizer_mascara(
        caminho_imagem=argumentos.imagem,
        caminho_checkpoint=argumentos.checkpoint,
        arquitetura=argumentos.arquitetura,
        tamanho_imagem=argumentos.tamanho_imagem,
        usar_opencv=argumentos.usar_opencv,
        configuracao_opencv=configuracao_opencv,
        limiar=argumentos.limiar,
    )

    Path(argumentos.saida).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(argumentos.saida, mascara)
    print(f"Máscara salva em: {argumentos.saida}")


if __name__ == "__main__":
    main()
