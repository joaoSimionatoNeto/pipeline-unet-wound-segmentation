"""
Módulo de reprodutibilidade.

Responsável por fixar as sementes aleatórias (seeds) de todas as bibliotecas
utilizadas no projeto (Python, NumPy e PyTorch), garantindo que os
experimentos possam ser reproduzidos de forma determinística sempre que
possível.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def fixar_seed_global(seed: int = 42, determinismo_total: bool = False) -> None:
    """Fixa a seed global em todas as bibliotecas relevantes do projeto.

    Args:
        seed: Valor inteiro utilizado como semente aleatória.
        determinismo_total: Se True, força o cuDNN a operar em modo
            determinístico. Isso pode reduzir a performance de treinamento,
            mas garante reprodutibilidade total entre execuções.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if determinismo_total:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Prioriza performance quando o determinismo total não é exigido.
        torch.backends.cudnn.benchmark = True


def detectar_dispositivo(preferencia: str = "auto") -> torch.device:
    """Detecta automaticamente o melhor dispositivo de processamento disponível.

    A ordem de prioridade em modo automático é: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        preferencia: "auto", "cpu", "cuda" ou "mps".

    Returns:
        Uma instância de torch.device pronta para uso.
    """
    if preferencia != "auto":
        return torch.device(preferencia)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
