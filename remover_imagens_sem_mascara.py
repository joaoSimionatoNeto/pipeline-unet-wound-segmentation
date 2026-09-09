"""Remove imagens que nao possuem mascara correspondente.

Por padrao, os diretorios sao:
    data/com_mascara/imagens
    data/com_mascara/mascaras

O pareamento considera o nome completo do arquivo, incluindo a extensao.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def encontrar_imagens_sem_mascara(diretorio_imagens: Path, diretorio_mascaras: Path) -> list[Path]:
    """Retorna as imagens cujo nome nao existe no diretorio de mascaras."""
    nomes_mascaras = {caminho.name for caminho in diretorio_mascaras.iterdir() if caminho.is_file()}
    imagens = (
        caminho
        for caminho in diretorio_imagens.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
    )
    return sorted((imagem for imagem in imagens if imagem.name not in nomes_mascaras), key=lambda caminho: caminho.name)


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove da pasta de imagens os arquivos sem mascara correspondente."
    )
    parser.add_argument(
        "--imagens",
        type=Path,
        default=Path("data/com_mascara/imagens"),
        help="Diretorio com as imagens de entrada.",
    )
    parser.add_argument(
        "--mascaras",
        type=Path,
        default=Path("data/com_mascara/mascaras"),
        help="Diretorio com as mascaras correspondentes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista os arquivos que seriam removidos.",
    )
    return parser


def main() -> None:
    args = criar_parser().parse_args()

    if not args.imagens.is_dir():
        raise SystemExit(f"Diretorio de imagens nao encontrado: {args.imagens}")
    if not args.mascaras.is_dir():
        raise SystemExit(f"Diretorio de mascaras nao encontrado: {args.mascaras}")

    imagens_sem_mascara = encontrar_imagens_sem_mascara(args.imagens, args.mascaras)

    if not imagens_sem_mascara:
        print("Nenhuma imagem sem mascara correspondente foi encontrada.")
        return

    acao = "seriam removidas" if args.dry_run else "removidas"
    print(f"{len(imagens_sem_mascara)} imagem(ns) {acao}:")
    for imagem in imagens_sem_mascara:
        if args.dry_run:
            print(f"  {imagem}")
        else:
            imagem.unlink()
            print(f"  {imagem}")


if __name__ == "__main__":
    main()
