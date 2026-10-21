"""Segmentação e medição da ferida usando a escala opcional do RulerNet."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_segmentation import (
    area_pixels_para_cm2,
    binarizar,
    calcular_metricas_mascara,
    carregar_modelo,
    listar_imagens,
    predizer_probabilidade,
)
from utils.seed import detectar_dispositivo


def _ruler_input(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, float]]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(768 / w, 768 / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((768, 768, 3), dtype=np.float32)
    top, left = (768 - nh) // 2, (768 - nw) // 2
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    canvas[top:top + nh, left:left + nw] = resized
    return np.transpose(canvas, (2, 0, 1))[None].astype(np.float32), (top, left, scale)


def _points_from_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    start, distance, ratio, direction, info = outputs[0][0], outputs[1][0][0], outputs[2][0][0], outputs[3][0], outputs[4][0]
    count = int(info[0])
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    n = np.arange(-count, count + 1)
    distances = (ratio ** n) * distance
    left = -np.expand_dims(distances[n < 0][::-1], 1) * direction
    right = np.expand_dims(distances[n >= 0], 1) * direction
    zero = np.zeros((1, 2), dtype=start.dtype)
    points = np.vstack([
        start + np.cumsum(np.vstack([zero, left]), axis=0)[::-1],
        start + np.cumsum(np.vstack([zero, right]), axis=0)[1:],
    ])
    min_x, min_y, max_x, max_y = info[1:].tolist()
    return points[
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x)
        & (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    ]


class RulerNetONNX:
    """Integra o model.onnx oficial de ymp5078/RulerNet."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Instale onnxruntime para usar RulerNet.") from error
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Modelo RulerNet inexistente: {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        tensor, (top, left, scale) = _ruler_input(image)
        outputs = self.session.run(None, {"input": tensor})
        if len(outputs) < 5:
            raise RuntimeError("RulerNet retornou menos de cinco tensores.")
        points = _points_from_outputs(outputs)
        if len(points) < 2:
            return {"pixels_per_cm": None, "confidence": 0.0, "points": []}
        px_cm = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1))) / scale
        points[:, 0] = (points[:, 0] - left) / scale
        points[:, 1] = (points[:, 1] - top) / scale
        return {"pixels_per_cm": px_cm, "confidence": 1.0, "points": points.tolist()}


def wound_measurements(mask: np.ndarray, pixels_per_cm: float | None) -> dict[str, Any]:
    ys, xs = np.where(mask)
    area = int(mask.sum())
    if len(xs) == 0:
        width = height = perimeter = 0.0
        centroid = (None, None)
    else:
        width, height = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
        centroid = (float(xs.mean()), float(ys.mean()))
    return {
        "area_pixels": area,
        "area_cm2": area_pixels_para_cm2(area, pixels_per_cm),
        "width_pixels": width,
        "height_pixels": height,
        "width_cm": width / pixels_per_cm if pixels_per_cm else None,
        "height_cm": height / pixels_per_cm if pixels_per_cm else None,
        "perimeter_pixels": perimeter,
        "perimeter_cm": perimeter / pixels_per_cm if pixels_per_cm else None,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
    }


def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = [args.image] if args.image else listar_imagens(args.images_dir)
    device = detectar_dispositivo(args.device)
    model = carregar_modelo(args.segmentation_weights, args.architecture, device)
    output = Path(args.output_dir)
    mask_dir = output / "predicted_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    ruler = RulerNetONNX(args.rulernet_model) if args.rulernet_model else None
    records: list[dict[str, Any]] = []
    logger = logging.getLogger("wound_measurement_pipeline")
    for path in paths:
        record: dict[str, Any] = {"arquivo": path.name}
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("imagem corrompida")
            probability = predizer_probabilidade(model, image, args.image_size, device)
            predicted = binarizar(probability, args.threshold)
            cv2.imwrite(str(mask_dir / path.name), predicted.astype(np.uint8) * 255)
            ruler_result = ruler.detect(image) if ruler else {"pixels_per_cm": None, "confidence": None}
            record.update(ruler_result)
            predicted_measures = wound_measurements(predicted, ruler_result["pixels_per_cm"])
            record.update({f"predicted_{key}": value for key, value in predicted_measures.items()})
            manual_path = args.manual_masks_dir / path.name
            if manual_path.exists():
                manual_image = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
                if manual_image is None or manual_image.shape != predicted.shape:
                    raise ValueError("máscara manual inválida ou com dimensões diferentes")
                record.update(calcular_metricas_mascara(predicted, manual_image > 127, path.name).como_dict())
                manual_measures = wound_measurements(manual_image > 127, ruler_result["pixels_per_cm"])
                record.update({f"manual_{key}": value for key, value in manual_measures.items()})
                record["absolute_area_difference_pixels"] = abs(
                    predicted_measures["area_pixels"] - manual_measures["area_pixels"]
                )
            else:
                record["erro"] = f"Máscara manual ausente: {manual_path}"
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("%s: %s", path.name, error)
            record["erro"] = str(error)
        records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "measurement_results.csv", "w", newline="", encoding="utf-8") as file:
        fields = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output / "measurements.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images_dir", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--manual_masks_dir", type=Path, required=True)
    parser.add_argument("--segmentation_weights", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("unet", "unet_resnet50"), default="unet_resnet50")
    parser.add_argument("--rulernet_model", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
