"""Preparacao e leitura de particoes fisicas de imagens e mascaras."""
from __future__ import annotations

import hashlib
import shutil
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def listar_pares(imagens_dir: Path, masks_dir: Path, extensao: str | None = None) -> tuple[list[str], list[str]]:
    """Lista imagens e mascaras correspondentes pelo nome-base."""
    if not imagens_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Diretorios de imagens/mascaras nao encontrados: {imagens_dir} / {masks_dir}")
    images = sorted(p for p in imagens_dir.iterdir() if p.is_file() and (extensao is None or p.suffix.lower() == extensao.lower()))
    masks_by_stem: dict[str, list[Path]] = {}
    for mask in masks_dir.iterdir():
        if mask.is_file():
            masks_by_stem.setdefault(mask.stem.casefold(), []).append(mask)
    image_paths: list[str] = []
    mask_paths: list[str] = []
    used_masks: set[Path] = set()
    for image in images:
        candidates = masks_by_stem.get(image.stem.casefold(), [])
        if len(candidates) > 1:
            raise ValueError(f"Pareamento ambiguo para {image}: {len(candidates)} mascaras")
        if candidates:
            image_paths.append(str(image))
            mask_paths.append(str(candidates[0]))
            used_masks.add(candidates[0])
    unmatched_images = len(images) - len(image_paths)
    orphan_masks = sum(1 for masks in masks_by_stem.values() for mask in masks if mask not in used_masks)
    if unmatched_images or orphan_masks:
        warnings.warn(f"Ignorando {unmatched_images} imagens sem mascara e {orphan_masks} mascaras sem imagem em {imagens_dir.parent}.", UserWarning)
    return image_paths, mask_paths


def _coletar(imagens_dir: Path, masks_dir: Path, extensao: str | None) -> list[tuple[Path, Path]]:
    image_paths, mask_paths = listar_pares(imagens_dir, masks_dir, extensao)
    pairs = [(Path(image), Path(mask)) for image, mask in zip(image_paths, mask_paths)]
    dedup: dict[tuple[str, str], tuple[Path, Path]] = {}
    for image, mask in pairs:
        dedup.setdefault((_sha256(image), _sha256(mask)), (image, mask))
    return list(dedup.values())


def coletar_fontes(diretorios: list[tuple[Path, Path]], extensao: str | None = None) -> list[tuple[Path, Path]]:
    """Combina origens, pareia por nome-base e remove pares duplicados."""
    gathered: list[tuple[Path, Path]] = []
    for images_dir, masks_dir in diretorios:
        gathered.extend(_coletar(images_dir, masks_dir, extensao))
    unique: dict[tuple[str, str], tuple[Path, Path]] = {}
    names: dict[str, str] = {}
    for image, mask in gathered:
        image_hash, mask_hash = _sha256(image), _sha256(mask)
        name = image.name.casefold()
        if name in names and names[name] != image_hash:
            raise ValueError(f"Colisao de nomes com conteudos diferentes: {name}")
        names[name] = image_hash
        unique.setdefault((image_hash, mask_hash), (image, mask))
    return list(unique.values())


def preparar_particoes(origens: list[tuple[Path, Path]], raiz: Path, proporcoes: dict[str, float], seed: int,
                       extensao: str | None = None, recriar: bool = False) -> Path:
    """Copia os pares para train/validation/test, reutilizando pastas validas existentes."""
    splits = ("train", "validation", "test")
    if raiz.exists() and any(raiz.iterdir()) and not recriar:
        try:
            counts = [len(listar_pares(raiz / split / "images", raiz / split / "masks", extensao)[0]) for split in splits]
        except FileNotFoundError as error:
            raise FileExistsError(f"Particoes incompletas em {raiz}; use --recriar para substitui-las") from error
        if any(count == 0 for count in counts):
            raise FileExistsError(f"Particoes vazias em {raiz}; use --recriar para substitui-las")
        return raiz
    if raiz.exists() and recriar:
        shutil.rmtree(raiz)
    if not origens:
        raise ValueError("Nenhum par imagem/mascara disponivel para particionar.")
    if set(proporcoes) != set(splits) or abs(sum(proporcoes.values()) - 1.0) > 1e-8:
        raise ValueError("As proporcoes train/validation/test devem somar 1.0")
    groups: dict[str, list[int]] = {}
    for i, (image, _) in enumerate(origens):
        groups.setdefault(_sha256(image), []).append(i)
    group_ids = list(range(len(groups)))
    if len(group_ids) < 3:
        raise ValueError("Sao necessarios pelo menos 3 conteudos de imagem distintos para tres particoes nao vazias")
    ntrain = max(1, min(len(group_ids) - 2, round(len(group_ids) * proporcoes["train"])))
    train_groups, rest = train_test_split(group_ids, train_size=ntrain, random_state=seed, shuffle=True)
    nval = max(1, min(len(rest) - 1, round(len(group_ids) * proporcoes["validation"])))
    val_groups, test_groups = train_test_split(rest, train_size=nval, random_state=seed, shuffle=True)
    group_values = list(groups.values())
    assignment = {
        sample_id: split
        for split, indexes in (("train", train_groups), ("validation", val_groups), ("test", test_groups))
        for group_index in indexes
        for sample_id in group_values[group_index]
    }
    for i, (image, mask) in enumerate(origens):
        split = assignment[i]
        for kind, source in (("images", image), ("masks", mask)):
            target = raiz / split / kind / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for split in splits:
        images, masks = listar_pares(raiz / split / "images", raiz / split / "masks", extensao)
        if not images or len(images) != len(masks):
            raise RuntimeError(f"Particao {split} vazia ou com pares incompletos")
    return raiz