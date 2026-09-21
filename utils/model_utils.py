"""
Utilitários para modificação de arquiteturas de modelos.

Contém a função `substituir_batchnorm_por_groupnorm`, usada para adaptar
qualquer modelo (incluindo um backbone pré-treinado, como a ResNet-50) a
cenários de treinamento com batches efetivos muito pequenos por passo de GPU
— como ocorre ao usar acumulação de gradientes processando uma imagem por
vez. Nesses cenários, o BatchNorm se torna instável porque suas estatísticas
(média/variância) são calculadas sobre o batch, e com batch=1 essas
estatísticas deixam de ser representativas. O GroupNorm normaliza dentro de
cada amostra individual (agrupando canais), portanto seu comportamento não
depende do tamanho do batch.
"""

from __future__ import annotations

import torch.nn as nn


def _maior_divisor_ate_limite(num_canais: int, limite: int) -> int:
    """Retorna o maior divisor de `num_canais` que seja <= `limite`.

    O GroupNorm exige que `num_canais` seja divisível pelo número de grupos.
    Para camadas com poucos canais (ou canais que não sejam múltiplos de 32,
    o padrão usual), buscamos o maior número de grupos válido até o limite
    desejado, garantindo compatibilidade em qualquer arquitetura.
    """
    limite = max(1, min(limite, num_canais))
    for candidato in range(limite, 0, -1):
        if num_canais % candidato == 0:
            return candidato
    return 1


def substituir_batchnorm_por_groupnorm(modulo: nn.Module, num_grupos_maximo: int = 32) -> nn.Module:
    """Substitui recursivamente todas as camadas `nn.BatchNorm2d` de `modulo` por `nn.GroupNorm`.

    Percorre `modulo` e todos os seus submódulos (incluindo os de um encoder
    pré-treinado, como a ResNet-50 usada em `UNetResNet50`) e troca cada
    `BatchNorm2d` encontrado por um `GroupNorm` equivalente em posição na
    rede, com número de grupos escolhido automaticamente (o maior divisor de
    `num_features` que seja <= `num_grupos_maximo`).

    Apenas a camada de normalização é substituída — os pesos convolucionais
    ao redor dela (inclusive os pré-treinados no ImageNet) não são alterados.
    Os parâmetros afins (`gamma`/`beta`) do GroupNorm são inicializados do
    zero (1.0 e 0.0), pois as estatísticas acumuladas de um BatchNorm não são
    diretamente aproveitáveis por um GroupNorm. O estado `requires_grad`
    (treinável ou congelado) de cada camada é preservado, para não destreinar
    acidentalmente um encoder que estava congelado (`fine_tuning=False`).

    Args:
        modulo: Módulo (ou modelo completo) a ser modificado *in-place*.
        num_grupos_maximo: Número máximo de grupos a considerar por camada.
            32 é o valor recomendado pelo artigo original do GroupNorm
            (Wu & He, 2018) e funciona bem na maioria dos casos.

    Returns:
        O próprio `modulo`, já modificado (retornado por conveniência para
        permitir `modelo = substituir_batchnorm_por_groupnorm(modelo)`).
    """
    for nome_filho, filho in modulo.named_children():
        if isinstance(filho, nn.BatchNorm2d):
            num_canais = filho.num_features
            num_grupos = _maior_divisor_ate_limite(num_canais, num_grupos_maximo)

            camada_groupnorm = nn.GroupNorm(
                num_groups=num_grupos,
                num_channels=num_canais,
                affine=filho.affine,
            )

            if filho.affine:
                # Preserva se os parâmetros afins eram treináveis ou congelados
                # (ex.: encoder congelado quando fine_tuning=False).
                requer_grad = filho.weight.requires_grad
                camada_groupnorm.weight.requires_grad = requer_grad
                camada_groupnorm.bias.requires_grad = requer_grad

            setattr(modulo, nome_filho, camada_groupnorm)
        else:
            substituir_batchnorm_por_groupnorm(filho, num_grupos_maximo)

    return modulo
