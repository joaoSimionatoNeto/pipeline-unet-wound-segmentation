"""
Módulo de Data Augmentation.

Constrói o pipeline de aumentação de dados a partir da configuração YAML,
utilizando a biblioteca Albumentations. Cada técnica pode ser
ativada/desativada e parametrizada individualmente.
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A


def construir_pipeline_augmentation(config_augmentation: Dict[str, Any]) -> A.Compose:
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

    if ativo("horizontal_flip"):
        transformacoes.append(A.HorizontalFlip(p=parametro("horizontal_flip", "probabilidade", 0.5)))

    if ativo("vertical_flip"):
        transformacoes.append(A.VerticalFlip(p=parametro("vertical_flip", "probabilidade", 0.5)))

    if ativo("rotation"):
        transformacoes.append(
            A.Rotate(limit=parametro("rotation", "limite_graus", 30), p=parametro("rotation", "probabilidade", 0.5))
        )

    if ativo("shift_scale_rotate"):
        transformacoes.append(
            A.ShiftScaleRotate(
                shift_limit=parametro("shift_scale_rotate", "shift_limit", 0.0625),
                scale_limit=parametro("shift_scale_rotate", "scale_limit", 0.1),
                rotate_limit=parametro("shift_scale_rotate", "rotate_limit", 15),
                p=parametro("shift_scale_rotate", "probabilidade", 0.5),
            )
        )

    if ativo("random_brightness_contrast"):
        transformacoes.append(
            A.RandomBrightnessContrast(p=parametro("random_brightness_contrast", "probabilidade", 0.4))
        )

    if ativo("clahe"):
        transformacoes.append(A.CLAHE(p=parametro("clahe", "probabilidade", 0.3)))

    if ativo("gaussian_noise"):
        transformacoes.append(A.GaussNoise(p=parametro("gaussian_noise", "probabilidade", 0.2)))

    if ativo("blur"):
        transformacoes.append(A.Blur(blur_limit=3, p=parametro("blur", "probabilidade", 0.2)))

    if ativo("elastic_transform"):
        transformacoes.append(A.ElasticTransform(p=parametro("elastic_transform", "probabilidade", 0.2)))

    if ativo("grid_distortion"):
        transformacoes.append(A.GridDistortion(p=parametro("grid_distortion", "probabilidade", 0.2)))

    return A.Compose(transformacoes)
