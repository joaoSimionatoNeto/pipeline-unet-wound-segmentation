from pathlib import Path
import random
import shutil


PASTA_BASE = Path(__file__).resolve().parent
EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PREFIXO_IMAGEM = "LLFW-"


def remover_imagens_com_prefixo(
    pasta_dados: Path | str,
    prefixo: str = PREFIXO_IMAGEM,
    extensoes: set[str] | None = None,
) -> int:
    """Remove imagens com o prefixo informado da pasta de dados e subpastas."""
    pasta_dados = Path(pasta_dados)
    extensoes = extensoes or EXTENSOES_IMAGEM

    if not pasta_dados.is_dir():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {pasta_dados}")

    imagens = [
        arquivo
        for arquivo in pasta_dados.rglob("*")
        if arquivo.is_file()
        and arquivo.name.startswith(prefixo)
        and arquivo.suffix.lower() in extensoes
    ]

    for imagem in imagens:
        imagem.unlink()
        print(f"Imagem removida: {imagem}")

    print(f"\nTotal de imagens removidas: {len(imagens)}")
    return len(imagens)


def separar_treino_validacao(
    pasta_dados: Path | str,
    proporcao_treino: float = 0.75,
    semente: int = 42,
) -> tuple[int, int]:
    """
    Separa pares de imagem e máscara em treino e validação.

    O pareamento usa o nome-base do arquivo, permitindo extensões diferentes
    entre a imagem e a máscara. Somente pares completos são movidos.
    """
    pasta_dados = Path(pasta_dados)

    if not 0 < proporcao_treino < 1:
        raise ValueError("A proporção de treino deve estar entre 0 e 1.")

    fontes = (
        (pasta_dados / "treino" / "imagens", pasta_dados / "treino" / "mascaras"),
        (pasta_dados / "com_regua" / "images", pasta_dados / "com_regua" / "masks"),
    )
    destino_treino = pasta_dados / "treino"
    destino_validacao = pasta_dados / "validacao"
    imagens_treino = destino_treino / "imagens"
    mascaras_treino = destino_treino / "mascaras"
    imagens_validacao = destino_validacao / "imagens"
    mascaras_validacao = destino_validacao / "mascaras"

    for imagens, mascaras in fontes:
        if not imagens.is_dir():
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {imagens}")
        if not mascaras.is_dir():
            raise FileNotFoundError(f"Pasta de máscaras não encontrada: {mascaras}")

    imagens_por_nome: dict[str, Path] = {}
    mascaras_por_nome: dict[str, Path] = {}
    imagens_sem_mascara: list[str] = []
    mascaras_sem_imagem: list[str] = []

    for pasta_imagens, pasta_mascaras in fontes:
        for imagem in pasta_imagens.iterdir():
            if imagem.is_file() and imagem.suffix.lower() in EXTENSOES_IMAGEM:
                nome = imagem.stem.casefold()
                if nome in imagens_por_nome:
                    raise ValueError(f"Imagem duplicada para o nome-base: {imagem.stem}")
                imagens_por_nome[nome] = imagem

        for mascara in pasta_mascaras.iterdir():
            if mascara.is_file() and mascara.suffix.lower() in EXTENSOES_IMAGEM:
                nome = mascara.stem.casefold()
                if nome in mascaras_por_nome:
                    raise ValueError(f"Máscara duplicada para o nome-base: {mascara.stem}")
                mascaras_por_nome[nome] = mascara

    pares = []
    for nome, imagem in imagens_por_nome.items():
        mascara = mascaras_por_nome.get(nome)
        if mascara is None:
            imagens_sem_mascara.append(imagem.name)
        else:
            pares.append((imagem, mascara))

    for nome, mascara in mascaras_por_nome.items():
        if nome not in imagens_por_nome:
            mascaras_sem_imagem.append(mascara.name)

    if not pares:
        raise ValueError("Nenhum par completo de imagem e máscara foi encontrado.")

    random.Random(semente).shuffle(pares)
    quantidade_treino = round(len(pares) * proporcao_treino)
    pares_treino = pares[:quantidade_treino]
    pares_validacao = pares[quantidade_treino:]

    for pasta in (
        imagens_treino,
        mascaras_treino,
        imagens_validacao,
        mascaras_validacao,
    ):
        pasta.mkdir(parents=True, exist_ok=True)

    movimentos = [
        (imagem, imagens_treino / imagem.name, mascara, mascaras_treino / mascara.name)
        for imagem, mascara in pares_treino
    ] + [
        (imagem, imagens_validacao / imagem.name, mascara, mascaras_validacao / mascara.name)
        for imagem, mascara in pares_validacao
    ]

    for imagem, destino_imagem, mascara, destino_mascara in movimentos:
        for origem, destino in (
            (imagem, destino_imagem),
            (mascara, destino_mascara),
        ):
            if origem.resolve() == destino.resolve():
                continue
            if destino.exists():
                raise FileExistsError(f"Arquivo já existe no destino: {destino}")

    for imagem, destino_imagem, mascara, destino_mascara in movimentos:
        if imagem.resolve() != destino_imagem.resolve():
            shutil.move(str(imagem), str(destino_imagem))
        if mascara.resolve() != destino_mascara.resolve():
            shutil.move(str(mascara), str(destino_mascara))

    print(f"Pares completos encontrados: {len(pares)}")
    print(f"Pares separados para treino: {len(pares_treino)}")
    print(f"Pares separados para validação: {len(pares_validacao)}")
    print(f"Imagens sem máscara (preservadas): {len(imagens_sem_mascara)}")
    print(f"Máscaras sem imagem (preservadas): {len(mascaras_sem_imagem)}")

    return len(pares_treino), len(pares_validacao)


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

    imagens = [
        arquivo
        for arquivo in pasta_imagens.iterdir()
        if arquivo.is_file()
        and arquivo.suffix.lower() in EXTENSOES_IMAGEM
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
    separar_treino_validacao(pasta_dados=PASTA_BASE / "data")