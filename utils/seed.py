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
import warnings

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
    try:
        torch.cuda.manual_seed_all(seed)
    except RuntimeError as erro:
        warnings.warn(
            "Nao foi possivel inicializar a seed CUDA; o treinamento podera usar CPU. "
            f"Detalhe: {erro}",
            RuntimeWarning,
        )

    if determinismo_total:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Prioriza performance quando o determinismo total não é exigido.
        torch.backends.cudnn.benchmark = True


def detectar_dispositivo(preferencia: str = "auto", memoria_minima_gb: float = 2.0) -> torch.device:
    """Detecta automaticamente o melhor dispositivo de processamento disponível.

    A ordem de prioridade em modo automático é: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        preferencia: "auto", "cpu", "cuda" ou "mps".

    Returns:
        Uma instância de torch.device pronta para uso.
    """
    if preferencia == "cpu":
        return torch.device("cpu")

    cuda_disponivel = False
    if preferencia in {"auto", "cuda"}:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"CUDA initialization: The NVIDIA driver.*",
                category=UserWarning,
            )
            try:
                # is_available() pode consultar apenas o NVML e nao garantir
                # que o runtime CUDA consiga criar um tensor no dispositivo.
                cuda_disponivel = torch.cuda.is_available()
                if cuda_disponivel:
                    torch.empty(1, device="cuda")
            except RuntimeError as erro:
                warnings.warn(
                    "CUDA foi detectado, mas nao pode ser inicializado. "
                    f"Usando CPU. Detalhe: {erro}",
                    RuntimeWarning,
                )
                cuda_disponivel = False

    if cuda_disponivel:
        melhor_indice = 0
        maior_memoria_livre = 0
        for indice in range(torch.cuda.device_count()):
            try:
                memoria_livre, _ = torch.cuda.mem_get_info(indice)
            except RuntimeError:
                continue
            if memoria_livre > maior_memoria_livre:
                maior_memoria_livre = memoria_livre
                melhor_indice = indice

        memoria_livre_gb = maior_memoria_livre / (1024**3)
        if preferencia == "auto" and memoria_livre_gb < memoria_minima_gb:
            warnings.warn(
                f"Nenhuma GPU possui pelo menos {memoria_minima_gb:.1f} GB livres "
                f"(maior valor: {memoria_livre_gb:.2f} GB). Usando CPU.",
                RuntimeWarning,
            )
            return torch.device("cpu")

        dispositivo = torch.device("cuda", melhor_indice)
        torch.cuda.set_device(dispositivo)
        return dispositivo
    if preferencia == "cuda":
        warnings.warn("CUDA foi solicitado, mas nao esta disponivel. Usando CPU.", RuntimeWarning)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def determinar_batch_size_disponivel(
    batch_configurado: int, dispositivo: torch.device, tamanho_imagem: int
) -> int:
    """Limita o batch de GPU conforme a memória livre observada no início da execução."""
    if dispositivo.type != "cuda":
        return batch_configurado

    memoria_livre, _ = torch.cuda.mem_get_info(dispositivo.index)
    memoria_livre_gb = memoria_livre / (1024**3)
    fator_resolucao = (tamanho_imagem / 512) ** 2
    memoria_reservada_gb = 2.0 * fator_resolucao
    memoria_por_amostra_gb = 2.5 * fator_resolucao
    capacidade = int((memoria_livre_gb - memoria_reservada_gb) / memoria_por_amostra_gb)
    return max(1, min(batch_configurado, capacidade))
