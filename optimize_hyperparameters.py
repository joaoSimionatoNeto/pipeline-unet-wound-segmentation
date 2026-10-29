"""Otimiza hiperparâmetros de segmentação com Optuna usando somente validação.

Uso: python optimize_hyperparameters.py --trials 30
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.augmentations import construir_pipeline_augmentation
from datasets.partitions import listar_pares
from datasets.wound_dataset import WoundDataset
from preprocessing.opencv_pipeline import ConfiguracaoPreProcessamento
from train import construir_modelo
from training.losses import obter_funcao_perda
from training.trainer import Trainer
from utils.seed import determinar_batch_size_disponivel, detectar_dispositivo, fixar_seed_global


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=None, help="Limite de épocas por trial; por padrão usa o YAML.")
    parser.add_argument("--study", default="wound_segmentation")
    parser.add_argument("--storage", default="sqlite:///results/optuna_study.db")
    parser.add_argument("--pipeline", default=None,
                        help="Pipeline a otimizar; por exemplo, unet_resnet50_opencv.")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    optuna_config = base.get("optuna", {})
    pipeline_name = args.pipeline or optuna_config.get("pipeline", "unet_resnet50")
    if pipeline_name not in base.get("pipelines", {}):
        raise ValueError(f"Pipeline '{pipeline_name}' não existe na configuração.")
    paths = base["caminhos"]
    train_images, train_masks = listar_pares(Path(paths["imagens_treino"]), Path(paths["mascaras_treino"]), base["dados"].get("extensao_imagem"))
    val_images, val_masks = listar_pares(Path(paths["imagens_validacao"]), Path(paths["mascaras_validacao"]), base["dados"].get("extensao_imagem"))
    if not train_images or not val_images:
        raise RuntimeError("Optuna requer pares de imagens e máscaras nos diretórios de treino e validação.")

    device = detectar_dispositivo(base["projeto"]["dispositivo"])
    def objetivo(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base)
        fixar_seed_global(int(base["projeto"]["seed"]) + trial.number)
        treino = config["treinamento"]
        augmentation = config["data_augmentation"]
        treino["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        treino["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        treino["loss"] = trial.suggest_categorical("loss", ["dice_bce", "tversky", "focal"])
        # Search loss-specific parameters only when that loss is selected.
        perdas = config.setdefault("perdas", {})
        if treino["loss"] == "focal":
            focal = perdas.setdefault("focal", {})
            focal["alpha"] = trial.suggest_float("focal_alpha", 0.1, 1.0)
            focal["gamma"] = trial.suggest_float("focal_gamma", 0.5, 5.0)
        elif treino["loss"] == "tversky":
            tversky = perdas.setdefault("tversky", {})
            tversky["alpha"] = trial.suggest_float("tversky_alpha", 0.1, 0.9)
            tversky["beta"] = trial.suggest_float("tversky_beta", 0.1, 0.9)
        treino["batch_size"] = trial.suggest_categorical("batch_size", [4, 8])
        # Otimiza apenas augmentations presentes no YAML (há esquemas antigos e novos).
        for nome, chave, param, minimo, maximo in (
            ("random_brightness_contrast", "probabilidade", "p_brightness_contrast", 0.1, 0.8),
            ("hue_saturation_value", "probabilidade", "p_hue_saturation", 0.0, 0.6),
            ("gamma", "probabilidade", "p_gamma", 0.0, 0.6),
            ("cutout", "probabilidade", "p_cutout", 0.0, 0.6),
            ("random_brightness", "probabilidade", "p_brightness", 0.0, 0.8),
            ("random_contrast", "probabilidade", "p_contrast", 0.0, 0.8),
            ("color_jitter", "probabilidade", "p_color_jitter", 0.0, 0.8),
            ("gaussian_blur", "probabilidade", "p_gaussian_blur", 0.0, 0.8),
            ("shift_scale_rotate", "probabilidade", "p_shift_scale_rotate", 0.0, 0.8),
        ):
            if nome in augmentation:
                augmentation[nome][chave] = trial.suggest_float(param, minimo, maximo)
        if "cutout" in augmentation and "num_holes_range" in augmentation["cutout"]:
            augmentation["cutout"]["num_holes_range"] = [1, trial.suggest_int("cutout_max_holes", 2, 10)]
        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        treino["epochs"] = int(args.epochs or optuna_config.get("epochs_por_trial", treino["epochs"]))
        treino["early_stopping"]["paciencia"] = min(int(treino["early_stopping"]["paciencia"]), 8)

        batch = determinar_batch_size_disponivel(int(treino["batch_size"]), device, int(treino["image_size"]))
        treino["batch_size"] = batch
        usar_opencv = bool(config["pipelines"][pipeline_name].get("usar_opencv", False))
        config_opencv = (ConfiguracaoPreProcessamento.a_partir_de_dicionario(config["pre_processamento_opencv"])
                         if usar_opencv else None)
        normalizar = not usar_opencv and config.get("modelo", {}).get("imagenet_normalizacao", False)
        dataset_train = WoundDataset(
            train_images, train_masks, treino["image_size"],
            construir_pipeline_augmentation(augmentation, int(treino["image_size"])),
            usar_opencv=usar_opencv, configuracao_opencv=config_opencv, normalizar_imagenet=normalizar,
        )
        dataset_val = WoundDataset(val_images, val_masks, treino["image_size"], None,
                                   usar_opencv=usar_opencv, configuracao_opencv=config_opencv,
                                   normalizar_imagenet=normalizar)
        workers = int(treino.get("num_workers", 0))
        loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
        loader_val = DataLoader(dataset_val, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")

        trial_root = Path(optuna_config.get("diretorio", "results/optuna")) / f"trial_{trial.number:04d}"
        config["caminhos"]["checkpoints"] = str(trial_root / "checkpoints")
        config["caminhos"]["logs"] = str(trial_root / "logs")
        model = construir_modelo(pipeline_name, config["pipelines"][pipeline_name], batch)
        trainer = Trainer(model, obter_funcao_perda(treino["loss"], config.get("perdas", {})), device, treino,
                          f"optuna_trial_{trial.number}", config["caminhos"]["checkpoints"], config["caminhos"]["logs"])
        history = trainer.treinar(loader_train, loader_val)
        if not history.dice_validacao:
            raise RuntimeError("Trial não produziu métricas de validação.")
        # Mede Dice na validação usando o limiar amostrado; treino segue monitorado
        # pelo Dice padrão, e a escolha do limiar não consulta o conjunto de teste.
        best_checkpoint = trial_root / "checkpoints" / f"optuna_trial_{trial.number}_best.pth"
        if best_checkpoint.is_file():
            checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["modelo_state_dict"])
        model.eval()
        total_dice, total_images = 0.0, 0
        with torch.no_grad():
            for images, masks in loader_val:
                logits = model(images.to(device))
                probs = torch.sigmoid(logits)
                pred = probs >= threshold
                alvo = masks.to(device) >= 0.5
                dims = (1, 2, 3)
                inter = (pred & alvo).sum(dims).float()
                denom = pred.sum(dims).float() + alvo.sum(dims).float()
                dice = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
                total_dice += float(dice.sum().item())
                total_images += int(dice.numel())
        trial.set_user_attr("best_epoch", history.dice_validacao.index(max(history.dice_validacao)) + 1)
        return total_dice / max(1, total_images)

    Path("results").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=args.study, storage=args.storage, load_if_exists=True, direction="maximize")
    study.optimize(objetivo, n_trials=args.trials)
    best = copy.deepcopy(base)
    params = study.best_params
    best["optuna"]["pipeline"] = pipeline_name
    best["treinamento"].update(learning_rate=params["learning_rate"], weight_decay=params["weight_decay"], loss=params["loss"], batch_size=params["batch_size"])
    best.setdefault("avaliacao", {})["limiar_binarizacao"] = params["threshold"]
    if params["loss"] == "focal" and "focal_alpha" in params:
        best.setdefault("perdas", {}).setdefault("focal", {}).update(
            alpha=params["focal_alpha"], gamma=params["focal_gamma"]
        )
    elif params["loss"] == "tversky" and "tversky_alpha" in params:
        best.setdefault("perdas", {}).setdefault("tversky", {}).update(
            alpha=params["tversky_alpha"], beta=params["tversky_beta"]
        )
    aug = best["data_augmentation"]
    for nome, chave, param in (("random_brightness_contrast", "probabilidade", "p_brightness_contrast"),
                               ("hue_saturation_value", "probabilidade", "p_hue_saturation"),
                               ("gamma", "probabilidade", "p_gamma"), ("cutout", "probabilidade", "p_cutout"),
                               ("random_brightness", "probabilidade", "p_brightness"),
                               ("random_contrast", "probabilidade", "p_contrast"),
                               ("color_jitter", "probabilidade", "p_color_jitter"),
                               ("gaussian_blur", "probabilidade", "p_gaussian_blur"),
                               ("shift_scale_rotate", "probabilidade", "p_shift_scale_rotate")):
        if nome in aug and param in params:
            aug[nome][chave] = params[param]
    if "cutout" in aug and "cutout_max_holes" in params:
        aug["cutout"]["num_holes_range"] = [1, params["cutout_max_holes"]]
    output = Path(optuna_config.get("config_saida", "configs/train_config_optuna.yaml"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(best, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (Path(optuna_config.get("diretorio", "results/optuna")) / "best_trial.json").write_text(
        json.dumps({"study": args.study, "best_value_val_dice": study.best_value, "best_params": params,
                    "best_epoch": study.best_trial.user_attrs.get("best_epoch"), "config": str(output)}, indent=2), encoding="utf-8")
    print(f"Melhor Dice de validação: {study.best_value:.5f}; configuração salva em {output}")


if __name__ == "__main__":
    main()
