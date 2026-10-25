import numpy as np

from evaluate_segmentation import (
    ResultadoAvaliacao,
    _resumo,
    area_pixels_para_cm2,
    buscar_melhor_limiar,
    calcular_escala_pixels_por_cm,
    calcular_metricas_mascara,
    localizar_mascara,
)


def test_resumo_ignora_metricas_nao_finitas():
    resultado = calcular_metricas_mascara(
        np.array([[1, 0], [0, 0]], dtype=bool),
        np.array([[1, 0], [0, 0]], dtype=bool),
    )
    erro = ResultadoAvaliacao("erro.png", *([float("nan")] * 7), *([0] * 6), "falha")

    resumo = _resumo([resultado, erro])

    assert resumo["dice_media"] == 1.0
    assert resumo["dice_mediana"] == 1.0
    assert resumo["dice_desvio_padrao"] == 0.0


def test_metricas_mascara_basicas():
    manual = np.array([[1, 1], [0, 0]], dtype=bool)
    predicao = np.array([[1, 0], [1, 0]], dtype=bool)
    resultado = calcular_metricas_mascara(predicao, manual, "x.png")
    assert resultado.verdadeiros_positivos == 1
    assert resultado.falsos_positivos == 1
    assert resultado.falsos_negativos == 1
    assert resultado.dice == 0.5
    assert resultado.iou == 1 / 3


def test_metricas_vazias_e_conversoes():
    vazia = np.zeros((2, 2), dtype=bool)
    assert calcular_metricas_mascara(vazia, vazia).dice == 1.0
    assert calcular_escala_pixels_por_cm(np.array([[0, 0], [3, 4], [6, 4]])) == 4.0
    assert area_pixels_para_cm2(100, 10) == 1.0
    assert area_pixels_para_cm2(100, None) is None


def test_busca_melhor_limiar():
    probabilidades = [np.array([[0.2, 0.8], [0.7, 0.1]])]
    manuais = [np.array([[0, 1], [1, 0]], dtype=bool)]
    melhor, tabela = buscar_melhor_limiar(probabilidades, manuais, 0.2, 0.8, 0.3, "dice")
    assert melhor == 0.5
    assert len(tabela) == 3


def test_correspondencia_por_nome(tmp_path):
    imagem = tmp_path / "paciente_001.png"
    imagem.write_bytes(b"x")
    assert localizar_mascara(imagem, tmp_path) == imagem
