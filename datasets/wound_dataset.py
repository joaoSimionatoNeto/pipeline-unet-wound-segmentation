"""
Dataset de segmentação de feridas.

Define a classe `WoundDataset`, responsável por carregar pares
(imagem, máscara), aplicar o pré-processamento OpenCV opcional e o pipeline
de data augmentation via Albumentations, retornando tensores prontos para o
PyTorch. Utiliza exclusivamente amostras que possuem máscara de segmentação.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.opencv_pipeline import ConfiguracaoPreProcessamento, PipelineOpenCV


class WoundDataset(Dataset):
    """Dataset PyTorch para pares de imagem e máscara de segmentação de feridas.

    Args:
        caminhos_imagens: Lista de caminhos para as imagens de entrada.
        caminhos_mascaras: Lista de caminhos para as máscaras correspondentes
            (mesma ordem de `caminhos_imagens`).
        tamanho_imagem: Tamanho (altura=largura) para o qual a imagem é
            redimensionada antes de entrar na rede.
        transformacoes: Pipeline de data augmentation do Albumentations
            (opcional). Deve aceitar `image=` e `mask=` como argumentos.
        usar_opencv: Se True, aplica o `PipelineOpenCV` de pré-processamento
            antes das transformações de augmentation.
        configuracao_opencv: Configuração do pipeline OpenCV, obrigatória
            quando `usar_opencv=True`.
    """

    def __init__(
        self,
        caminhos_imagens: List[str],
        caminhos_mascaras: List[str],
        tamanho_imagem: int = 512,
        transformacoes: Optional[Callable] = None,
        usar_opencv: bool = False,
        configuracao_opencv: Optional[ConfiguracaoPreProcessamento] = None,
    ) -> None:
        if len(caminhos_imagens) != len(caminhos_mascaras):
            raise ValueError("As listas de imagens e máscaras devem ter o mesmo tamanho.")

        self.caminhos_imagens = caminhos_imagens
        self.caminhos_mascaras = caminhos_mascaras
        self.tamanho_imagem = tamanho_imagem
        self.transformacoes = transformacoes
        self.usar_opencv = usar_opencv

        self.pipeline_opencv: Optional[PipelineOpenCV] = None
        if usar_opencv:
            if configuracao_opencv is None:
                raise ValueError("configuracao_opencv é obrigatória quando usar_opencv=True.")
            self.pipeline_opencv = PipelineOpenCV(configuracao_opencv)

    def __len__(self) -> int:
        return len(self.caminhos_imagens)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor]:
        imagem, mascara = self._carregar_par(indice)

        if self.pipeline_opencv is not None:
            imagem = self.pipeline_opencv.processar(imagem)
            # Garante uint8 para compatibilidade com Albumentations quando a
            # normalização OpenCV ainda não foi aplicada em float.
            if imagem.dtype != np.uint8:
                imagem = np.clip((imagem * 255.0) if imagem.max() <= 1.0 else imagem, 0, 255).astype(np.uint8)
        else:
            imagem = cv2.resize(imagem, (self.tamanho_imagem, self.tamanho_imagem))

        mascara = cv2.resize(mascara, (self.tamanho_imagem, self.tamanho_imagem), interpolation=cv2.INTER_NEAREST)

        if self.transformacoes is not None:
            aumentado = self.transformacoes(image=imagem, mask=mascara)
            imagem, mascara = aumentado["image"], aumentado["mask"]

        tensor_imagem = self._converter_imagem_para_tensor(imagem)
        tensor_mascara = self._converter_mascara_para_tensor(mascara)

        return tensor_imagem, tensor_mascara

    def _carregar_par(self, indice: int) -> Tuple[np.ndarray, np.ndarray]:
        """Carrega a imagem (BGR) e a máscara binária (uint8) correspondentes."""
        caminho_imagem = self.caminhos_imagens[indice]
        caminho_mascara = self.caminhos_mascaras[indice]

        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
        if imagem is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_imagem}")

        mascara = cv2.imread(caminho_mascara, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            raise FileNotFoundError(f"Não foi possível carregar a máscara: {caminho_mascara}")

        mascara = (mascara > 127).astype(np.uint8) * 255
        return imagem, mascara

    @staticmethod
    def _converter_imagem_para_tensor(imagem: np.ndarray) -> torch.Tensor:
        """Converte a imagem (HWC) para tensor PyTorch (CHW), float32."""
        if imagem.dtype == np.uint8:
            imagem = imagem.astype(np.float32) / 255.0
        tensor = torch.from_numpy(imagem.transpose(2, 0, 1).copy()).float()
        return tensor

    @staticmethod
    def _converter_mascara_para_tensor(mascara: np.ndarray) -> torch.Tensor:
        """Converte a máscara (HW) para tensor PyTorch (1, H, W), binária."""
        mascara_binaria = (mascara > 127).astype(np.float32)
        return torch.from_numpy(mascara_binaria).unsqueeze(0)


def listar_amostras_com_mascara(diretorio_imagens: str, diretorio_mascaras: str, extensao: str = ".png") -> Tuple[List[str], List[str]]:
    """Lista apenas as amostras que possuem imagem E máscara correspondentes.

    Conforme exigido pelo projeto, a pipeline de treinamento utiliza
    exclusivamente imagens que possuem máscara de segmentação associada.

    Args:
        diretorio_imagens: Diretório contendo as imagens brutas.
        diretorio_mascaras: Diretório contendo as máscaras de segmentação.
        extensao: Extensão de arquivo esperada (ex.: ".png").

    Returns:
        Tupla (caminhos_imagens, caminhos_mascaras), ordenados e pareados.
    """
    caminho_imagens = Path(diretorio_imagens)
    caminho_mascaras = Path(diretorio_mascaras)

    imagens_disponiveis = sorted(caminho_imagens.glob(f"*{extensao}"))

    caminhos_imagens: List[str] = []
    caminhos_mascaras: List[str] = []

    for caminho_imagem in imagens_disponiveis:
        caminho_mascara_correspondente = caminho_mascaras / caminho_imagem.name
        if caminho_mascara_correspondente.exists():
            caminhos_imagens.append(str(caminho_imagem))
            caminhos_mascaras.append(str(caminho_mascara_correspondente))

    return caminhos_imagens, caminhos_mascaras
