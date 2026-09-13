"""Fine-tune Wav2Vec2 with short fixed windows and a call-level split."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "nexus" / "manifest.csv"
BASE_MODEL = "facebook/wav2vec2-base"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_OUTPUT = ROOT / "wav2vec2_finetuned_model_few_epochs"
ID_TO_LABEL = {0: "human", 1: "synthetic"}
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}


@dataclass(frozen=True)
class AudioRecord:
    anon_id: str
    path: Path
    label: int


class FixedAudioDataset(Dataset):
    def __init__(
        self,
        records: list[AudioRecord],
        processor: Wav2Vec2Processor,
        window_seconds: float,
        training: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.processor = processor
        self.target_length = int(window_seconds * TARGET_SAMPLE_RATE)
        self.training = training
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _prepare_audio(self, path: Path) -> np.ndarray:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.nan_to_num(audio[:, 0].astype(np.float32, copy=False))
        if sample_rate != TARGET_SAMPLE_RATE:
            mono = resample_poly(mono, TARGET_SAMPLE_RATE, sample_rate).astype(
                np.float32, copy=False
            )
        if len(mono) >= self.target_length:
            max_start = len(mono) - self.target_length
            start = self.random.randint(0, max_start) if self.training else max_start // 2
            return mono[start : start + self.target_length]
        return np.pad(mono, (0, self.target_length - len(mono)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        audio = self._prepare_audio(record.path)
        encoded = self.processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        item["labels"] = torch.tensor(record.label, dtype=torch.long)
        return item


def build_records(manifest_path: Path) -> dict[str, list[AudioRecord]]:
    manifest = pd.read_csv(manifest_path)
    records = {"train": [], "val": []}
    for row in manifest.itertuples(index=False):
        label = LABEL_TO_ID[str(row.label)]
        path = ROOT / "datos" / "channel_0" / str(row.label) / f"{row.anon_id}__ch0.wav"
        if not path.is_file():
            raise FileNotFoundError(f"No existe el audio etiquetado: {path}")
        records[str(row.split)].append(AudioRecord(str(row.anon_id), path, label))
    return records


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(
    model: Wav2Vec2ForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
            losses.append(float(output.loss.item()))
            labels.extend(batch["labels"].cpu().tolist())
            probabilities.extend(torch.softmax(output.logits, dim=-1)[:, 1].float().cpu().tolist())
    label_array = np.asarray(labels, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    prediction_array = (probability_array >= 0.5).astype(int)
    return {
        "loss": float(np.mean(losses)),
        "auc": float(roc_auc_score(label_array, probability_array)),
        "accuracy": float(accuracy_score(label_array, prediction_array)),
        "precision": float(precision_score(label_array, prediction_array, zero_division=0)),
        "recall": float(recall_score(label_array, prediction_array, zero_division=0)),
        "f1": float(f1_score(label_array, prediction_array, zero_division=0)),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = build_records(args.manifest)
    processor = Wav2Vec2Processor.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"
    model.freeze_feature_encoder()
    model.to(device)

    train_dataset = FixedAudioDataset(
        records["train"], processor, args.window_seconds, training=True, seed=args.seed
    )
    val_dataset = FixedAudioDataset(
        records["val"], processor, args.window_seconds, training=False, seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(output.loss.item()))

        validation = evaluate(model, val_loader, device, use_amp)
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_auc={validation['auc']:.4f} "
            f"val_accuracy={validation['accuracy']:.4f}"
        )
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                **validation,
            }

    if best_metrics is None:
        raise RuntimeError("No se obtuvo ninguna métrica de validación.")
    if best_state is None:
        raise RuntimeError("No se pudo conservar el mejor checkpoint.")
    model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    metadata = {
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_seconds": args.window_seconds,
        "seed": args.seed,
        "device": str(device),
        "train_records": len(records["train"]),
        "val_records": len(records["val"]),
        "best_metrics": best_metrics,
        "channel_1_reserved_for_external_eval": True,
    }
    (args.output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()