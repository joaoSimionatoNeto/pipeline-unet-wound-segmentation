"""
Módulo de logging estruturado.

Centraliza a configuração de logs do projeto, permitindo saída simultânea
para console e para arquivo, com formatação consistente em todos os módulos.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def criar_logger(nome: str, diretorio_logs: str = "logs", nivel: int = logging.INFO) -> logging.Logger:
    """Cria (ou recupera) um logger configurado para console e arquivo.

    Args:
        nome: Nome identificador do logger (geralmente o nome do módulo).
        diretorio_logs: Diretório onde o arquivo de log será salvo.
        nivel: Nível mínimo de severidade a ser registrado.

    Returns:
        Instância de logging.Logger configurada.
    """
    logger = logging.getLogger(nome)

    # Evita adicionar handlers duplicados caso o logger já exista.
    if logger.handlers:
        return logger

    logger.setLevel(nivel)

    formato = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formato)
    logger.addHandler(handler_console)

    caminho_logs = Path(diretorio_logs)
    caminho_logs.mkdir(parents=True, exist_ok=True)
    handler_arquivo = logging.FileHandler(caminho_logs / f"{nome}.log", encoding="utf-8")
    handler_arquivo.setFormatter(formato)
    logger.addHandler(handler_arquivo)

    return logger
