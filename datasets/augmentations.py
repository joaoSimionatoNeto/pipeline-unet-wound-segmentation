"""
Módulo de Data Augmentation.

Constrói o pipeline de aumentação de dados a partir da configuração YAML,
utilizando a biblioteca Albumentations. Cada técnica pode ser
ativada/desativada e parametrizada individualmente.
"""

from __future__ import annotations

from typing import Any, Dict
import inspect

import albumentations as A


def construir_pipeline_augmentation(
    config_augmentation: Dict[str, Any], tamanho_referencia: int = 512
) -> A.Compose:
    """Constrói um `albumentations.Compose` a partir da configuração fornecida.

    Args:
        config_augmentation: Dicionário lido da seção `data_augmentation` do
            arquivo de configuração YAML.

    Returns:
        Pipeline `A.Compose` pronto para ser aplicado a pares imagem/máscara.
    """
    transformacoes = []

    if not config_augmentation.get("ativo", True):
        return A.Compose(transformacoes)

    def ativo(nome: str) -> bool:
        return bool(config_augmentation.get(nome, {}).get("ativo", False))

    def parametro(nome: str, chave: str, padrao: Any) -> Any:
        return config_augmentation.get(nome, {}).get(chave, padrao)

    # O strength escala a intensidade dos ajustes, sem alterar suas
    # probabilidades. A força 1.0 preserva os limites declarados no YAML.
    forca = float(config_augmentation.get("augmentation_strength", 1.0))

    def escalar_limite(valor: Any, fator: float) -> Any:
        if isinstance(valor, (list, tuple)):
            return type(valor)(float(item) * fator for item in valor)
        return float(valor) * fator

    if ativo("horizontal_flip"):
        transformacoes.append(A.HorizontalFlip(p=parametro("horizontal_flip", "probabilidade", 0.5)))

    if ativo("vertical_flip"):
        transformacoes.append(A.VerticalFlip(p=parametro("vertical_flip", "probabilidade", 0.5)))

    if ativo("rotacao_90"):
        transformacoes.append(A.RandomRotate90(p=parametro("rotacao_90", "probabilidade", 0.5)))

    if ativo("rotation"):
        transformacoes.append(
            A.Rotate(limit=parametro("rotation", "limite_graus", 30), p=parametro("rotation", "probabilidade", 0.5))
        )

    if ativo("shift_scale_rotate"):
        transformacoes.append(
            A.Affine(
                translate_percent=(
                    -parametro("shift_scale_rotate", "shift_limit", 0.0625),
                    parametro("shift_scale_rotate", "shift_limit", 0.0625),
                ),
                scale=(
                    1 - parametro("shift_scale_rotate", "scale_limit", 0.1),
                    1 + parametro("shift_scale_rotate", "scale_limit", 0.1),
                ),
                rotate=(
                    -parametro("shift_scale_rotate", "rotate_limit", 15),
                    parametro("shift_scale_rotate", "rotate_limit", 15),
                ),
                p=parametro("shift_scale_rotate", "probabilidade", 0.5),
            )
        )

    if ativo("random_brightness") or ativo("random_contrast"):
        brilho_ativo = ativo("random_brightness")
        contraste_ativo = ativo("random_contrast")
        if brilho_ativo:
            transformacoes.append(
                A.RandomBrightnessContrast(
                    brightness_limit=escalar_limite(
                        parametro("random_brightness", "brightness_limit", 0.2), forca
                    ),
                    contrast_limit=0,
                    p=parametro("random_brightness", "probabilidade", 1.0),
                )
            )
        if contraste_ativo:
            transformacoes.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0,
                    contrast_limit=escalar_limite(
                        parametro("random_contrast", "contrast_limit", 0.2), forca
                    ),
                    p=parametro("random_contrast", "probabilidade", 1.0),
                )
            )

    # Compatibilidade com configurações anteriores que agrupavam brilho e contraste.
    elif ativo("random_brightness_contrast"):
        transformacoes.append(
            A.RandomBrightnessContrast(
                brightness_limit=escalar_limite(
                    parametro("random_brightness_contrast", "brightness_limit", 0.2), forca
                ),
                contrast_limit=escalar_limite(
                    parametro("random_brightness_contrast", "contrast_limit", 0.2), forca
                ),
                p=parametro("random_brightness_contrast", "probabilidade", 0.4),
            )
        )

    if ativo("hue_saturation_value"):
        transformacoes.append(
            A.HueSaturationValue(
                hue_shift_limit=parametro("hue_saturation_value", "hue_shift_limit", 10),
                sat_shift_limit=parametro("hue_saturation_value", "sat_shift_limit", 15),
                val_shift_limit=parametro("hue_saturation_value", "val_shift_limit", 10),
                p=parametro("hue_saturation_value", "probabilidade", 0.3),
            )
        )

    if ativo("gamma"):
        transformacoes.append(
            A.RandomGamma(
                gamma_limit=tuple(parametro("gamma", "limite", (80, 120))),
                p=parametro("gamma", "probabilidade", 0.3),
            )
        )

    if ativo("clahe"):
        transformacoes.append(A.CLAHE(p=parametro("clahe", "probabilidade", 0.3)))

    if ativo("gaussian_noise"):
        transformacoes.append(A.GaussNoise(p=parametro("gaussian_noise", "probabilidade", 0.2)))

    if ativo("blur"):
        transformacoes.append(A.Blur(blur_limit=3, p=parametro("blur", "probabilidade", 0.2)))

    if ativo("cutout"):
        # CoarseDropout altera apenas pixels da imagem; a máscara fica intacta.
        argumentos_cutout = {
            "num_holes_range": tuple(parametro("cutout", "num_holes_range", (1, 8))),
            "hole_height_range": tuple(parametro("cutout", "hole_height_range", (0.02, 0.12))),
            "hole_width_range": tuple(parametro("cutout", "hole_width_range", (0.02, 0.12))),
            "fill": parametro("cutout", "fill", 0),
            "p": parametro("cutout", "probabilidade", 0.3),
        }
        # Albumentations 1.x usava dimensões em pixels e nomes diferentes.
        assinatura_cutout = inspect.signature(A.CoarseDropout).parameters
        if "num_holes_range" not in assinatura_cutout:
            n_holes = argumentos_cutout.pop("num_holes_range")
            argumentos_cutout.pop("hole_height_range")
            argumentos_cutout.pop("hole_width_range")
            argumentos_cutout["min_holes"], argumentos_cutout["max_holes"] = n_holes
            argumentos_cutout["min_height"], argumentos_cutout["max_height"] = (16, 64)
            argumentos_cutout["min_width"], argumentos_cutout["max_width"] = (16, 64)
            if "fill" in assinatura_cutout:
                argumentos_cutout["fill_value"] = argumentos_cutout.pop("fill")
            else:
                argumentos_cutout.pop("fill", None)
        else:
            # A API 2.x diferencia intervalos relativos (float) de tamanhos
            # em pixels (int). Esta configuração usa frações da dimensão da imagem.
            for chave in ("hole_height_range", "hole_width_range"):
                intervalo = argumentos_cutout[chave]
                if all(isinstance(valor, int) for valor in intervalo):
                    continue
                # Frações da resolução padrão 512x512 convertidas para pixels.
                # O Albumentations instalado valida esses campos como >= 1.
                argumentos_cutout[chave] = tuple(
                    max(1, round(float(valor) * tamanho_referencia)) for valor in intervalo
                )
        transformacoes.append(A.CoarseDropout(**argumentos_cutout))

    if ativo("grid_dropout"):
        transformacoes.append(
            A.GridDropout(
                ratio=parametro("grid_dropout", "ratio", 0.3),
                p=parametro("grid_dropout", "probabilidade", 0.5),
            )
        )

    if ativo("elastic_transform"):
        transformacoes.append(A.ElasticTransform(p=parametro("elastic_transform", "probabilidade", 0.2)))

    if ativo("grid_distortion"):
        transformacoes.append(
            A.GridDistortion(
                distort_limit=escalar_limite(
                    parametro("grid_distortion", "distort_limit", 0.3), forca
                ),
                p=parametro("grid_distortion", "probabilidade", 0.2),
            )
        )

    # A aplicação a 100% das amostras controla o Compose; cada transformação
    # ainda respeita sua probabilidade individual.
    probabilidade_pipeline = 1.0 if config_augmentation.get("aplicar_em_todas_imagens_treino", False) else 0.0
    return A.Compose(transformacoes, p=probabilidade_pipeline)
