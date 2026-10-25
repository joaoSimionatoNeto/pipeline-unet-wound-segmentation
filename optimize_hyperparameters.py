"""Otimiza hiperparâmetros de segmentação com Optuna usando somente validação.

Uso: python optimize_hyperparameters.py --trials 30
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import optuna
import yaml
from torch.utils.data import DataLoader

from datasets.augmentations import construir_pipeline_augmentation
from datasets.partitions import listar_pares
from datasets.wound_dataset import WoundDataset
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
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    optuna_config = base.get("optuna", {})
    pipeline_name = optuna_config.get("pipeline", "unet_resnet50")
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
        treino["batch_size"] = trial.suggest_categorical("batch_size", [1, 2, 4])
        treino["gradient_accumulation"]["accum_steps"] = trial.suggest_categorical("accum_steps", [1, 2, 4, 8])
        augmentation["random_brightness_contrast"]["probabilidade"] = trial.suggest_float("p_brightness_contrast", 0.1, 0.8)
        augmentation["hue_saturation_value"]["probabilidade"] = trial.suggest_float("p_hue_saturation", 0.0, 0.6)
        augmentation["gamma"]["probabilidade"] = trial.suggest_float("p_gamma", 0.0, 0.6)
        augmentation["cutout"]["probabilidade"] = trial.suggest_float("p_cutout", 0.0, 0.6)
        augmentation["cutout"]["num_holes_range"] = [1, trial.suggest_int("cutout_max_holes", 2, 10)]
        treino["epochs"] = int(args.epochs or optuna_config.get("epochs_por_trial", treino["epochs"]))
        treino["early_stopping"]["paciencia"] = min(int(treino["early_stopping"]["paciencia"]), 8)

        batch = determinar_batch_size_disponivel(int(treino["batch_size"]), device, int(treino["image_size"]))
        accumulation = max(1, (int(treino["batch_size"]) * int(treino["gradient_accumulation"]["accum_steps"]) + batch - 1) // batch)
        treino["batch_size"] = batch
        treino["gradient_accumulation"]["accum_steps"] = accumulation
        normalizar = not config["pipelines"][pipeline_name].get("usar_opencv", False) and config.get("modelo", {}).get("imagenet_normalizacao", False)
        dataset_train = WoundDataset(
            train_images, train_masks, treino["image_size"],
            construir_pipeline_augmentation(augmentation, int(treino["image_size"])),
            normalizar_imagenet=normalizar,
        )
        dataset_val = WoundDataset(val_images, val_masks, treino["image_size"], None, normalizar_imagenet=normalizar)
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
        best_dice = max(history.dice_validacao)
        trial.set_user_attr("best_epoch", history.dice_validacao.index(best_dice) + 1)
        return float(best_dice)

    Path("results").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=args.study, storage=args.storage, load_if_exists=True, direction="maximize")
    study.optimize(objetivo, n_trials=args.trials)
    best = copy.deepcopy(base)
    params = study.best_params
    best["treinamento"].update(learning_rate=params["learning_rate"], weight_decay=params["weight_decay"], loss=params["loss"], batch_size=params["batch_size"])
    best["treinamento"]["gradient_accumulation"]["accum_steps"] = params["accum_steps"]
    aug = best["data_augmentation"]
    aug["random_brightness_contrast"]["probabilidade"] = params["p_brightness_contrast"]
    aug["hue_saturation_value"]["probabilidade"] = params["p_hue_saturation"]
    aug["gamma"]["probabilidade"] = params["p_gamma"]
    aug["cutout"]["probabilidade"] = params["p_cutout"]
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
