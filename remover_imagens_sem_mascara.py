from pathlib import Path
import shutil


PASTA_BASE = Path(__file__).resolve().parent


def transferir_mascaras(
    pasta_imagens: Path | str,
    pasta_mascaras_origem: Path | str | None = None,
    pasta_mascaras_destino: Path | str | None = None,
) -> int:
    """
    Procura as máscaras das imagens de uma pasta e as transfere
    para a pasta de máscaras correspondente.

    Retorna a quantidade de máscaras transferidas.
    """
    pasta_imagens = Path(pasta_imagens)
    pasta_mascaras_origem = (
        Path(pasta_mascaras_origem)
        if pasta_mascaras_origem
        else PASTA_BASE / "wound_mask"
    )
    pasta_mascaras_destino = (
        Path(pasta_mascaras_destino)
        if pasta_mascaras_destino
        else PASTA_BASE / "validation_masks"
    )

    if not pasta_imagens.is_dir():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {pasta_imagens}"
        )

    if not pasta_mascaras_origem.is_dir():
        raise FileNotFoundError(
            f"Pasta de máscaras não encontrada: {pasta_mascaras_origem}"
        )

    pasta_mascaras_destino.mkdir(parents=True, exist_ok=True)

    extensoes_validas = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    imagens = [
        arquivo
        for arquivo in pasta_imagens.iterdir()
        if arquivo.is_file()
        and arquivo.suffix.lower() in extensoes_validas
    ]

    total_transferido = 0
    mascaras_nao_encontradas = []

    for imagem in imagens:
        mascara = pasta_mascaras_origem / imagem.name

        if not mascara.is_file():
            mascaras_nao_encontradas.append(imagem.name)
            continue

        destino = pasta_mascaras_destino / mascara.name

        if destino.exists():
            print(f"Máscara já existe no destino: {destino.name}")
            continue

        shutil.move(str(mascara), str(destino))

        total_transferido += 1
        print(f"Máscara transferida: {mascara.name} -> {destino}")

    print(f"\nTotal de imagens analisadas: {len(imagens)}")
    print(f"Máscaras transferidas: {total_transferido}")
    print(f"Máscaras não encontradas: {len(mascaras_nao_encontradas)}")

    if mascaras_nao_encontradas:
        print("\nImagens sem máscara correspondente:")
        for nome in mascaras_nao_encontradas:
            print(f" - {nome}")

    return total_transferido

if __name__ == "__main__":
    transferir_mascaras(
        pasta_imagens=Path(
            r"C:\TCC - CODIGO\teste 03\wound_segmentation\data\validacao\images"
        ),
        pasta_mascaras_origem=Path(
            r"C:\TCC - CODIGO\teste 03\wound_segmentation\data\treino\mascaras"
        ),
        pasta_mascaras_destino=Path(
            r"C:\TCC - CODIGO\teste 03\wound_segmentation\data\validacao\masks"
        ),
    )