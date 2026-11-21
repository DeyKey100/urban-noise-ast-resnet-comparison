"""Evaluate and compare the two UrbanSound8K models.

The default protocol is fold 10, which is the test fold documented for ModelA.
Use ``--fold all`` to evaluate every file found in ``Sounds``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
LABELS = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def load_audio(path: Path, target_rate: int, duration: float | None = None) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if duration is not None:
        waveform = waveform[: int(source_rate * duration)]
    if source_rate != target_rate:
        waveform = resample_poly(waveform, target_rate, source_rate)
    return np.asarray(waveform, dtype=np.float32)


def mel_spectrogram(waveform: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    from scipy.signal import stft

    nperseg = min(2048, max(2, len(waveform)))
    noverlap = min(1536, nperseg - 1)
    frequencies, _, spectrum = stft(
        waveform, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
        nfft=2048, boundary="zeros"
    )
    power = np.abs(spectrum) ** 2
    mel_points = np.linspace(
        0, 2595 * np.log10(1 + (sample_rate / 2) / 700), 130
    )
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((2049 * hz_points) / sample_rate).astype(int)
    filters = np.zeros((128, len(frequencies)), dtype=np.float32)
    for index in range(128):
        left, center, right = bins[index:index + 3]
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)
    return filters @ power


@dataclass
class Prediction:
    label: str
    probabilities: np.ndarray


def load_manifest(csv_path: Path, sounds_dir: Path, fold: str) -> list[tuple[Path, str]]:
    """Return files that have both a CSV label and an audio file."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    audio_index = {
        path.name: path for path in sounds_dir.rglob("*.wav") if path.is_file()
    }
    selected: list[tuple[Path, str]] = []
    missing = 0
    for row in rows:
        if fold != "all" and str(row["fold"]) != fold:
            continue
        audio_path = audio_index.get(row["slice_file_name"])
        if audio_path is None:
            missing += 1
            continue
        label = row["class"]
        if label not in LABELS:
            raise ValueError(f"Unknown class {label!r} in {csv_path}")
        selected.append((audio_path, label))

    if missing:
        print(f"Warning: {missing} manifest files are absent from {sounds_dir}", file=sys.stderr)
    if not selected:
        raise FileNotFoundError(
            f"No labelled WAV files found in {sounds_dir} for fold {fold!r}"
        )
    return selected


class ModelA:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = torch.device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            str(model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()

    def predict(self, audio_path: Path) -> Prediction:
        return self.predict_many([audio_path])[0]

    def predict_many(self, audio_paths: list[Path]) -> list[Prediction]:
        waveforms = [
            load_audio(path, self.extractor.sampling_rate)
            for path in audio_paths
        ]
        inputs = self.extractor(
            waveforms,
            sampling_rate=self.extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=1).cpu().numpy()
        return [
            Prediction(LABELS[int(values.argmax())], values)
            for values in probabilities
        ]


class ModelB:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        import torch.nn as nn
        from safetensors.torch import load_file
        from torchvision.models import resnet34

        class AdaptiveConcatPool2d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.average = nn.AdaptiveAvgPool2d((1, 1))
                self.maximum = nn.AdaptiveMaxPool2d((1, 1))

            def forward(self, value: Any) -> Any:
                return torch.cat([self.maximum(value), self.average(value)], dim=1)

        base = resnet34(weights=None)
        body = nn.Sequential(*list(base.children())[:-2])
        head = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(1024, eps=1e-5, momentum=0.1),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-5, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(512, len(LABELS), bias=False),
        )
        self.torch = torch
        self.device = torch.device(device)
        self.model = nn.Sequential(body, head)
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors")))
        self.model.to(self.device).eval()

    @staticmethod
    def _audio_tensor(audio_path: Path) -> Any:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        import torchvision.transforms as transforms

        mel = mel_spectrogram(load_audio(audio_path, 22050, duration=4.0))
        mel_db = 10 * np.log10(np.maximum(mel, 1e-10) / np.max(mel))
        figure = plt.figure(figsize=[0.72, 0.72])
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", dpi=400, bbox_inches="tight",
                       pad_inches=0)
        plt.close(figure)
        image = Image.open(image_buffer).convert("RGB")
        tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )(image).unsqueeze(0)
        image.close()
        image_buffer.close()
        return tensor

    def predict(self, audio_path: Path) -> Prediction:
        return self.predict_many([audio_path])[0]

    def predict_many(self, audio_paths: list[Path]) -> list[Prediction]:
        tensor = self.torch.cat(
            [self._audio_tensor(path) for path in audio_paths]
        ).to(self.device)
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(tensor), dim=1)
        return [
            Prediction(LABELS[int(values.argmax())], values)
            for values in probabilities.cpu().numpy()
        ]


def metrics(
    truth: list[str], predictions: list[Prediction], model_name: str
) -> tuple[dict[str, Any], list[list[int]]]:
    predicted = [item.label for item in predictions]
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for actual, guess in zip(truth, predicted):
        matrix[LABELS.index(actual)][LABELS.index(guess)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    accuracy = sum(
        actual == guess for actual, guess in zip(truth, predicted)
    ) / len(truth)
    confusions = [
        {
            "true": LABELS[actual],
            "predicted": LABELS[predicted_index],
            "count": matrix[actual][predicted_index],
        }
        for actual in range(len(LABELS))
        for predicted_index in range(len(LABELS))
        if actual != predicted_index and matrix[actual][predicted_index]
    ]
    confusions.sort(key=lambda item: int(item["count"]), reverse=True)
    summary = {
        "model": model_name,
        "samples": len(truth),
        "accuracy": accuracy,
        "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])),
        "per_class": per_class,
        "top_confusions": confusions[:10],
    }
    return summary, matrix


def bootstrap_accuracy_difference(
    truth: list[str],
    first: list[Prediction],
    second: list[Prediction],
    seed: int,
    repeats: int = 2000,
) -> dict[str, float]:
    first_correct = np.array(
        [item.label == actual for item, actual in zip(first, truth)], dtype=float
    )
    second_correct = np.array(
        [item.label == actual for item, actual in zip(second, truth)], dtype=float
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(truth), size=(repeats, len(truth)))
    differences = (first_correct[samples] - second_correct[samples]).mean(axis=1)
    return {
        "model_a_minus_model_b_accuracy": float((first_correct - second_correct).mean()),
        "95_percent_ci_low": float(np.quantile(differences, 0.025)),
        "95_percent_ci_high": float(np.quantile(differences, 0.975)),
        "paired_disagreements": int(np.count_nonzero(first_correct != second_correct)),
    }


def write_outputs(
    output_dir: Path,
    manifest: list[tuple[Path, str]],
    predictions: dict[str, list[Prediction]],
    summaries: dict[str, Any],
    matrices: dict[str, list[list[int]]],
    comparison: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "labels": LABELS,
        "models": summaries,
        "comparison": comparison,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, matrix in matrices.items():
        with (output_dir / f"{name}_confusion_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_label", *LABELS])
            writer.writerows([label, *row] for label, row in zip(LABELS, matrix))
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = ["file", "true_label", "model_a", "model_a_confidence",
                      "model_b", "model_b_confidence", "models_agree"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (path, actual) in enumerate(manifest):
            a, b = predictions["model_a"][index], predictions["model_b"][index]
            writer.writerow(
                {
                    "file": path.name,
                    "true_label": actual,
                    "model_a": a.label,
                    "model_a_confidence": f"{a.probabilities.max():.8f}",
                    "model_b": b.label,
                    "model_b_confidence": f"{b.probabilities.max():.8f}",
                    "models_agree": a.label == b.label,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sounds", type=Path, default=ROOT / "Sounds")
    parser.add_argument("--csv", type=Path, default=ROOT / "UrbanSound8K.csv")
    parser.add_argument("--model-a", type=Path, default=ROOT / "ModelA")
    parser.add_argument("--model-b", type=Path, default=ROOT / "ModelB")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument("--fold", default="10", help="CSV fold number or 'all'")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N files")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("CUDA is unavailable; using CPU.", file=sys.stderr)
            args.device = "cpu"
    manifest = load_manifest(args.csv, args.sounds, args.fold)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        manifest = manifest[: args.limit]
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    models: list[tuple[str, Any]] = [
        ("model_a", ModelA(args.model_a, args.device)),
        ("model_b", ModelB(args.model_b, args.device)),
    ]
    truth = [label for _, label in manifest]
    predictions: dict[str, list[Prediction]] = {}
    for name, model in models:
        predictions[name] = []
        for start in range(0, len(manifest), args.batch_size):
            paths = [path for path, _ in manifest[start:start + args.batch_size]]
            predictions[name].extend(model.predict_many(paths))

    summaries: dict[str, Any] = {}
    matrices: dict[str, list[list[int]]] = {}
    for name in ("model_a", "model_b"):
        summaries[name], matrices[name] = metrics(truth, predictions[name], name)
    comparison = bootstrap_accuracy_difference(
        truth, predictions["model_a"], predictions["model_b"], args.seed
    )
    comparison["per_class_f1_difference_a_minus_b"] = {
        label: summaries["model_a"]["per_class"][label]["f1"]
        - summaries["model_b"]["per_class"][label]["f1"]
        for label in LABELS
    }
    comparison["agreement_rate"] = float(
        np.mean(
            [
                a.label == b.label
                for a, b in zip(predictions["model_a"], predictions["model_b"])
            ]
        )
    )
    write_outputs(
        args.output, manifest, predictions, summaries, matrices, comparison
    )
    print(json.dumps({"models": summaries, "comparison": comparison}, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
