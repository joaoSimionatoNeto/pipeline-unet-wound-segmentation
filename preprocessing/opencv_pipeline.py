"""
Pipeline de pré-processamento com OpenCV.

Implementa uma sequência configurável de técnicas clássicas de processamento
de imagens, utilizada pelas Pipelines 3 e 4 (etapa OpenCV antes da rede
neural). Cada técnica pode ser ativada/desativada individualmente via
arquivo de configuração YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass
class ConfiguracaoPreProcessamento:
    """Estrutura de configuração para o pipeline de pré-processamento OpenCV."""

    parametros: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def a_partir_de_dicionario(cls, dados: Dict[str, Any]) -> "ConfiguracaoPreProcessamento":
        """Cria a configuração a partir do dicionário lido do YAML."""
        return cls(parametros=dados or {})

    def esta_ativo(self, etapa: str) -> bool:
        """Verifica se determinada etapa está ativada na configuração."""
        return bool(self.parametros.get(etapa, {}).get("ativo", False))

    def obter(self, etapa: str, chave: str, padrao: Any = None) -> Any:
        """Obtém um parâmetro específico de uma etapa, com valor padrão."""
        return self.parametros.get(etapa, {}).get(chave, padrao)


class PipelineOpenCV:
    """Executa uma sequência configurável de operações de pré-processamento.

    Cada método `_etapa_*` implementa uma técnica isolada, permitindo fácil
    manutenção, testes unitários e composição. A ordem de execução segue a
    ordem lógica recomendada para pré-processamento de imagens médicas de
    feridas: redimensionamento -> correções fotométricas -> filtragem de
    ruído -> realce -> conversão de espaço de cor -> normalização.
    """

    def __init__(self, configuracao: ConfiguracaoPreProcessamento) -> None:
        self.config = configuracao

    def processar(self, imagem: np.ndarray) -> np.ndarray:
        """Aplica todas as etapas ativas do pipeline sobre uma imagem.

        Args:
            imagem: Imagem de entrada no formato HWC, canais BGR ou RGB
                (uint8), conforme lida pelo OpenCV.

        Returns:
            Imagem processada, em float32, já normalizada se configurado.
        """
        resultado = imagem.copy()

        if self.config.esta_ativo("redimensionar"):
            resultado = self._etapa_redimensionar(resultado)

        if self.config.esta_ativo("correcao_iluminacao"):
            resultado = self._etapa_correcao_iluminacao(resultado)

        if self.config.esta_ativo("equalizacao_histograma"):
            resultado = self._etapa_equalizacao_histograma(resultado)

        if self.config.esta_ativo("clahe"):
            resultado = self._etapa_clahe(resultado)

        if self.config.esta_ativo("ajuste_contraste"):
            resultado = self._etapa_ajuste_contraste(resultado)

        if self.config.esta_ativo("remocao_ruido_gaussiano"):
            resultado = self._etapa_remocao_ruido_gaussiano(resultado)

        if self.config.esta_ativo("filtro_mediano"):
            resultado = self._etapa_filtro_mediano(resultado)

        if self.config.esta_ativo("sharpening"):
            resultado = self._etapa_sharpening(resultado)

        if self.config.esta_ativo("conversao_espaco_cor"):
            resultado = self._etapa_conversao_espaco_cor(resultado)

        if self.config.esta_ativo("normalizacao"):
            resultado = self._etapa_normalizacao(resultado)

        return resultado

    # ------------------------------------------------------------------
    # Etapas individuais de pré-processamento
    # ------------------------------------------------------------------

    def _etapa_redimensionar(self, imagem: np.ndarray) -> np.ndarray:
        """Redimensiona a imagem para o tamanho configurado."""
        largura, altura = self.config.obter("redimensionar", "tamanho", [512, 512])
        return cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_LINEAR)

    def _etapa_equalizacao_histograma(self, imagem: np.ndarray) -> np.ndarray:
        """Aplica equalização de histograma no canal de luminância (YCrCb)."""
        ycrcb = cv2.cvtColor(imagem, cv2.COLOR_BGR2YCrCb)
        canal_y, canal_cr, canal_cb = cv2.split(ycrcb)
        canal_y = cv2.equalizeHist(canal_y)
        ycrcb_equalizado = cv2.merge([canal_y, canal_cr, canal_cb])
        return cv2.cvtColor(ycrcb_equalizado, cv2.COLOR_YCrCb2BGR)

    def _etapa_clahe(self, imagem: np.ndarray) -> np.ndarray:
        """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clip_limit = self.config.obter("clahe", "clip_limit", 2.0)
        tile_grid = tuple(self.config.obter("clahe", "tile_grid_size", [8, 8]))

        lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
        canal_l, canal_a, canal_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        canal_l = clahe.apply(canal_l)
        lab_resultado = cv2.merge([canal_l, canal_a, canal_b])
        return cv2.cvtColor(lab_resultado, cv2.COLOR_LAB2BGR)

    def _etapa_remocao_ruido_gaussiano(self, imagem: np.ndarray) -> np.ndarray:
        """Remove ruído aplicando um filtro Gaussiano."""
        kernel_size = self.config.obter("remocao_ruido_gaussiano", "kernel_size", 5)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 0)

    def _etapa_filtro_mediano(self, imagem: np.ndarray) -> np.ndarray:
        """Aplica filtro mediano, eficaz contra ruído do tipo sal e pimenta."""
        kernel_size = self.config.obter("filtro_mediano", "kernel_size", 5)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(imagem, kernel_size)

    def _etapa_ajuste_contraste(self, imagem: np.ndarray) -> np.ndarray:
        """Ajusta contraste e brilho via transformação linear (alpha, beta)."""
        alpha = self.config.obter("ajuste_contraste", "alpha", 1.2)
        beta = self.config.obter("ajuste_contraste", "beta", 10)
        return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

    def _etapa_sharpening(self, imagem: np.ndarray) -> np.ndarray:
        """Aplica um kernel de realce de bordas (sharpening)."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(imagem, -1, kernel)

    def _etapa_correcao_iluminacao(self, imagem: np.ndarray) -> np.ndarray:
        """Corrige iluminação não uniforme utilizando subtração de fundo (top-hat)."""
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        fundo = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, kernel)
        fundo_bgr = cv2.cvtColor(fundo, cv2.COLOR_GRAY2BGR).astype(np.float32)
        imagem_corrigida = imagem.astype(np.float32) - fundo_bgr + 128.0
        return np.clip(imagem_corrigida, 0, 255).astype(np.uint8)

    def _etapa_conversao_espaco_cor(self, imagem: np.ndarray) -> np.ndarray:
        """Converte a imagem para o espaço de cor de destino configurado."""
        destino = self.config.obter("conversao_espaco_cor", "destino", "LAB")
        conversoes = {
            "LAB": cv2.COLOR_BGR2LAB,
            "HSV": cv2.COLOR_BGR2HSV,
            "GRAY": cv2.COLOR_BGR2GRAY,
        }
        codigo = conversoes.get(destino.upper())
        if codigo is None:
            return imagem
        convertida = cv2.cvtColor(imagem, codigo)
        if destino.upper() == "GRAY":
            convertida = cv2.cvtColor(convertida, cv2.COLOR_GRAY2BGR)
        return convertida

    def _etapa_normalizacao(self, imagem: np.ndarray) -> np.ndarray:
        """Normaliza a imagem para float32 no intervalo [0, 1], usando média/desvio padrão."""
        imagem_float = imagem.astype(np.float32) / 255.0
        media = np.array(self.config.obter("normalizacao", "media", [0.485, 0.456, 0.406]), dtype=np.float32)
        desvio = np.array(self.config.obter("normalizacao", "desvio_padrao", [0.229, 0.224, 0.225]), dtype=np.float32)
        return (imagem_float - media) / desvio
