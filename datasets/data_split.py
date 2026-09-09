"""
Módulo de divisão dos dados.

Implementa a divisão reprodutível do conjunto de amostras (imagem + máscara)
em treino, validação e teste, com suporte opcional a estratificação por uma
coluna de metadados (ex.: tipo de ferida ou tom de pele).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split


def dividir_dados(
    caminhos_imagens: List[str],
    caminhos_mascaras: List[str],
    proporcao_treino: float = 0.70,
    proporcao_validacao: float = 0.15,
    proporcao_teste: float = 0.15,
    rotulos_estratificacao: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """Divide o conjunto de dados em treino, validação e teste de forma reprodutível.

    Args:
        caminhos_imagens: Lista de caminhos das imagens.
        caminhos_mascaras: Lista de caminhos das máscaras correspondentes.
        proporcao_treino: Fração destinada ao treino.
        proporcao_validacao: Fração destinada à validação.
        proporcao_teste: Fração destinada ao teste.
        rotulos_estratificacao: Lista opcional de rótulos (mesma ordem das
            imagens) usada para estratificar a divisão.
        seed: Semente para reprodutibilidade.

    Returns:
        Três tuplas (imagens, mascaras) para treino, validação e teste.
    """
    soma_proporcoes = proporcao_treino + proporcao_validacao + proporcao_teste
    if abs(soma_proporcoes - 1.0) > 1e-6:
        raise ValueError(f"As proporções devem somar 1.0 (atual: {soma_proporcoes}).")

    total_amostras = len(caminhos_imagens)
    if total_amostras != len(caminhos_mascaras):
        raise ValueError("As listas de imagens e máscaras devem ter o mesmo tamanho.")

    if total_amostras == 0:
        return ([], []), ([], []), ([], [])

    def selecionar(indices_selecionados: List[int]) -> Tuple[List[str], List[str]]:
        imagens = [caminhos_imagens[i] for i in indices_selecionados]
        mascaras = [caminhos_mascaras[i] for i in indices_selecionados]
        return imagens, mascaras

    indices = list(range(len(caminhos_imagens)))
    estratificacao = rotulos_estratificacao if rotulos_estratificacao else None

    if total_amostras < 3:
        return (caminhos_imagens, caminhos_mascaras), ([], []), ([], [])

    quantidade_treino = max(1, int(round(total_amostras * proporcao_treino)))
    quantidade_treino = min(quantidade_treino, total_amostras - 2)

    indices_treino, indices_temp = train_test_split(
        indices,
        train_size=quantidade_treino,
        random_state=seed,
        stratify=[estratificacao[i] for i in indices] if estratificacao else None,
    )

    if len(indices_temp) < 2:
        return selecionar(indices_treino), ([], []), ([], [])

    proporcao_relativa_validacao = proporcao_validacao / (proporcao_validacao + proporcao_teste)
    quantidade_validacao = max(1, int(round(len(indices_temp) * proporcao_relativa_validacao)))
    quantidade_validacao = min(quantidade_validacao, len(indices_temp) - 1)
    estratificacao_temp = [estratificacao[i] for i in indices_temp] if estratificacao else None

    indices_validacao, indices_teste = train_test_split(
        indices_temp,
        train_size=quantidade_validacao,
        random_state=seed,
        stratify=estratificacao_temp,
    )

    return selecionar(indices_treino), selecionar(indices_validacao), selecionar(indices_teste)
