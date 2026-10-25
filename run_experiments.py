"""Executa experimentos ResNet50 nas partições físicas existentes.

Uso: python run_experiments.py (requer pastas train/validation/test em data/particoes/grid_resnet50)
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import csv
import shutil
from pathlib import Path

import yaml

from datasets.partitions import listar_pares


def avaliar_experimento(root: Path, partitions_root: Path, batch_size: int, image_size: int, device_name: str) -> float:
    """Escolhe limiar na validação e mede uma única vez no teste."""
    import cv2
    import numpy as np
    import torch
    from evaluate_segmentation import (avaliar_imagens, buscar_melhor_limiar, carregar_modelo,
                                       listar_imagens, predizer_probabilidade, salvar_csv)
    from utils.seed import detectar_dispositivo

    device = detectar_dispositivo(device_name)
    weights = root / "checkpoints" / "unet_resnet50_best.pth"
    model = carregar_modelo(weights, "unet_resnet50", device, batch_size=batch_size)
    val_images, val_masks = listar_pares(partitions_root / "validation" / "images", partitions_root / "validation" / "masks")
    if not val_images:
        raise RuntimeError("A pasta de validacao nao contem pares imagem/mascara.")
    probs, masks = [], []
    for image_path, mask_path in zip(val_images, val_masks):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        probs.append(predizer_probabilidade(model, image, image_size, device, normalizar_imagenet=True))
        masks.append(mask > 127)
    threshold, table = buscar_melhor_limiar(probs, masks, 0.30, 0.70, 0.05, "dice")
    eval_dir = root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "threshold_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["threshold", "dice"])
        writer.writeheader(); writer.writerows(table)
    (eval_dir / "threshold.json").write_text(json.dumps({"threshold": threshold, "selected_on": "validation"}, indent=2), encoding="utf-8")
    test_images, test_masks = listar_pares(partitions_root / "test" / "images", partitions_root / "test" / "masks")
    if not test_images:
        raise RuntimeError("A pasta de teste nao contem pares imagem/mascara.")
    results = avaliar_imagens([Path(p) for p in test_images], Path(test_masks[0]).parent, model, device,
                              eval_dir / "test", image_size, threshold, True, batch_size, normalizar_imagenet=True)
    salvar_csv(results, eval_dir / "test_metrics.csv")
    worst = sorted((r for r in results if np.isfinite(r.dice)), key=lambda r: r.dice)[:20]
    worst_dir = eval_dir / "worst_20"
    worst_dir.mkdir(exist_ok=True)
    for result in worst:
        overlay = eval_dir / "test" / "overlays" / result.arquivo
        if overlay.exists(): shutil.copy2(overlay, worst_dir / overlay.name)
    return float(np.mean([r.dice for r in results if np.isfinite(r.dice)])) if results else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experimentos_resnet50.yaml")
    args = parser.parse_args()
    matrix = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    project_path = Path(matrix["project_config"])
    base = yaml.safe_load(project_path.read_text(encoding="utf-8-sig"))
    pconf = matrix["partitions"]
    partitions_root = Path(pconf["root"])
    for split in ("train", "validation", "test"):
        images, masks = listar_pares(partitions_root / split / "images", partitions_root / split / "masks")
        if not images or len(images) != len(masks):
            raise FileNotFoundError(f"Pasta {split} ausente ou sem pares imagem/mascara: {partitions_root / split}")
    statuses = []
    for exp in matrix["experiments"]:
        config = copy.deepcopy(base)
        config["caminhos"]["imagens_treino"] = str(partitions_root / "train" / "images")
        config["caminhos"]["mascaras_treino"] = str(partitions_root / "train" / "masks")
        config["caminhos"]["imagens_validacao"] = str(partitions_root / "validation" / "images")
        config["caminhos"]["mascaras_validacao"] = str(partitions_root / "validation" / "masks")
        config["caminhos"]["imagens_teste"] = str(partitions_root / "test" / "images")
        config["caminhos"]["mascaras_teste"] = str(partitions_root / "test" / "masks")
        config["treinamento"]["batch_size"] = int(exp["batch_size"])
        config["treinamento"]["gradient_accumulation"] = {"ativo": True, "accum_steps": int(exp["accum_steps"])}
        config["treinamento"]["loss"] = exp["loss"]
        config["perdas"]["tversky"].update(alpha=0.7, beta=0.3)
        root = Path(pconf["root"]) / exp["name"]
        config["caminhos"]["checkpoints"] = str(root / "checkpoints")
        config["caminhos"]["logs"] = str(root / "logs")
        config["caminhos"]["resultados"] = str(root / "results")
        config["pipelines"]["unet_resnet50"]["diretorio_experimento"] = str(root / "experiment")
        config_path = root / "config.yaml"
        root.mkdir(parents=True, exist_ok=True)
        if (root / "status.json").exists():
            prior = json.loads((root / "status.json").read_text(encoding="utf-8"))
            if prior.get("status") == "completed":
                statuses.append(prior)
                continue
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (root / "status.json").write_text(json.dumps({"status": "running", "experiment": exp}, indent=2), encoding="utf-8")
        target_effective = int(exp["batch_size"]) * int(exp["accum_steps"])
        physical = int(exp["batch_size"])
        while physical >= 1:
            config["treinamento"]["batch_size"] = physical
            config["treinamento"]["gradient_accumulation"]["accum_steps"] = max(1, (target_effective + physical - 1) // physical)
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            command = [sys.executable, "train.py", "--config", str(config_path), "--pipeline", "unet_resnet50"]
            if (root / "checkpoints" / "unet_resnet50_last.pth").exists():
                command.append("--retomar")
            try:
                subprocess.run(command, check=True)
                status = {"status": "completed", "experiment": exp, "physical_batch": physical,
                          "accum_steps": config["treinamento"]["gradient_accumulation"]["accum_steps"]}
                status["test_dice"] = avaliar_experimento(root, partitions_root, physical, int(config["treinamento"]["image_size"]), config["projeto"]["dispositivo"])
                break
            except subprocess.CalledProcessError as error:
                if physical == 1:
                    status = {"status": "skipped", "reason": "training failed at physical batch 1",
                              "returncode": error.returncode, "experiment": exp}
                    break
                physical = max(1, physical // 2)
        (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        statuses.append(status)
    (Path(pconf["root"]) / "matrix_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    ranking = sorted((s for s in statuses if s.get("status") == "completed"), key=lambda s: s.get("test_dice", float("-inf")), reverse=True)
    (Path(pconf["root"]) / "ranking_test.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
