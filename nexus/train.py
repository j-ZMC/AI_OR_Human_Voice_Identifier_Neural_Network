from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from features import (
    ACOUSTIC_FEATURE_COLS,
    CALL_FEATURE_COLS,
    FULL_FEATURE_COLS,
    extract_features,
    extract_turn_features,
    load_turns,
)
from acoustic_models import (
    load_mini_ast_model,
    load_wav2vec_models,
    predict_acoustic_file,
    resolve_training_audio,
)
from torch_models import (
    apply_normalizer,
    build_model,
    fit_normalizer,
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
    save_normalization,
)
from random_forest import RandomForestClassifier

EPOCHS = 300
LEARNING_RATE = 0.04
TORCH_LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.001

BASE_DIR = Path(__file__).resolve().parent
MANIFEST = BASE_DIR / "manifest.csv"
TURNS_DIR = BASE_DIR / "turns"
if not TURNS_DIR.exists():
    TURNS_DIR = BASE_DIR.parent / "hackmty26" / "turns"
MODELS_DIR = BASE_DIR / "models"
MODEL_OUT = MODELS_DIR / "legacy_model" / "model.joblib"
DATASET_OUT = BASE_DIR / "dataset.csv"
ACOUSTIC_CACHE = BASE_DIR / "acoustic_predictions.csv"

MODEL_FEATURES = {
    "time_predict": CALL_FEATURE_COLS,
    "sustantive_predict": FULL_FEATURE_COLS,
}

RF_FEATURE_COLS = [
    "time_predict_probability",
    "sustantive_predict_probability",
    *ACOUSTIC_FEATURE_COLS,
]
RF_FOLDS = 5
RF_FOLD_EPOCHS = 120


def build_turn_dataset(manifest_path: str | Path = MANIFEST, turns_dir: str | Path = TURNS_DIR):
    """La base de datos: una fila por turno individual del caller (separado
    por los tiempos que habla cada uno, solo la seccion del caller). ~5100
    filas sobre 353 llamadas."""
    manifest = pd.read_csv(manifest_path)
    turns_dir = Path(turns_dir)
    rows = []
    for _, row in manifest.iterrows():
        turns_path = turns_dir / f"{row.anon_id}.json"
        turns = load_turns(turns_path)
        for turn_feats in extract_turn_features(turns):
            rows.append({
                "anon_id": row.anon_id,
                "turn_number": turn_feats["turn_number"],
                "prev_turn_end": turn_feats["prev_turn_end"],
                "turn_start": turn_feats["turn_start"],
                "turn_end": turn_feats["turn_end"],
                "reaction_time": turn_feats["reaction_time"],
                "label": 1 if row.label == "synthetic" else 0,
                "split": row.split,
            })
    return pd.DataFrame(rows)


def build_full_call_dataset(
    manifest_path: str | Path = MANIFEST,
    turns_dir: str | Path = TURNS_DIR,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    turns_dir = Path(turns_dir)
    rows = []
    for _, row in manifest.iterrows():
        turns = load_turns(turns_dir / f"{row.anon_id}.json")
        features = extract_features(turns)
        rows.append({
            **features,
            "anon_id": row.anon_id,
            "label": 1 if row.label == "synthetic" else 0,
            "split": row.split,
        })
    return pd.DataFrame(rows)


def build_acoustic_dataset(
    manifest_path: str | Path = MANIFEST,
    cache_path: str | Path = ACOUSTIC_CACHE,
    force: bool = False,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    cache_path = Path(cache_path)
    required_columns = {
        "anon_id",
        "label",
        "split",
        *ACOUSTIC_FEATURE_COLS,
    }
    if not force and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if required_columns.issubset(cached.columns) and set(manifest["anon_id"]).issubset(cached["anon_id"]):
            return cached[cached["anon_id"].isin(manifest["anon_id"])].copy()

    mini_ast_model = load_mini_ast_model()
    wav2vec_processor, wav2vec_model = load_wav2vec_models()
    rows = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        audio_path = resolve_training_audio(row.anon_id, row.label)
        probabilities = predict_acoustic_file(
            audio_path,
            mini_ast_model,
            wav2vec_processor,
            wav2vec_model,
        )
        rows.append({
            "anon_id": row.anon_id,
            "label": 1 if row.label == "synthetic" else 0,
            "split": row.split,
            "audio_path": str(audio_path),
            **probabilities,
        })
        if index == 1 or index % 25 == 0 or index == len(manifest):
            print(f"Modelos acusticos: {index}/{len(manifest)} llamadas")
    acoustic_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    acoustic_df.to_csv(cache_path, index=False)
    return acoustic_df


def _attach_acoustic_features(
    meta_frame: pd.DataFrame,
    acoustic_df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {"anon_id", *ACOUSTIC_FEATURE_COLS}
    missing_columns = sorted(required_columns - set(acoustic_df.columns))
    if missing_columns:
        raise ValueError(
            "Faltan columnas del cache acustico: " + ", ".join(missing_columns)
        )
    acoustic = acoustic_df.copy()
    acoustic["anon_id"] = acoustic["anon_id"].astype(str)
    if acoustic["anon_id"].duplicated().any():
        raise ValueError("El cache acustico tiene anon_id duplicados.")
    acoustic = acoustic.set_index("anon_id")[ACOUSTIC_FEATURE_COLS]

    ordered_ids = meta_frame["anon_id"].astype(str).reset_index(drop=True)
    acoustic_values = acoustic.reindex(ordered_ids)
    if acoustic_values.isna().any().any():
        missing_ids = ordered_ids[acoustic_values.isna().any(axis=1)].unique().tolist()
        raise ValueError(
            "Faltan predicciones acusticas para llamadas: "
            + ", ".join(missing_ids[:5])
        )
    if not np.isfinite(acoustic_values.to_numpy(dtype=float)).all():
        raise ValueError("El cache acustico contiene probabilidades no finitas.")
    return pd.concat(
        [meta_frame.reset_index(drop=True), acoustic_values.reset_index(drop=True)],
        axis=1,
    )


def build_call_dataset(turn_df):
    """Para ENTRENAR: agrega los turnos de cada llamada en 5 estadisticas
    (media/std/min/max/mediana) de reaction_time. Promediar las
    PREDICCIONES por turno da confidences pegadas a 0.5 e inestables bajo
    cross-validation (accuracy 71.7% +- 9.5%); agregar el FEATURE crudo por
    llamada antes de entrenar da 89.7% +- 2.7%, mucho mas confiable."""
    agg = turn_df.groupby("anon_id")["reaction_time"].agg(
        reaction_mean="mean", reaction_std="std", reaction_min="min",
        reaction_max="max", reaction_median="median",
    )
    agg["reaction_std"] = turn_df.groupby("anon_id")["reaction_time"].std(ddof=0).fillna(0.0)
    meta = turn_df.groupby("anon_id").agg(label=("label", "first"), split=("split", "first"))
    return agg.join(meta).reset_index()


def _evaluate_torch_model(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    loss_function: torch.nn.Module,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        loss = float(loss_function(logits, torch.from_numpy(labels)).item())
        probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities > 0.5).astype(int)
    return {
        "loss": loss,
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _train_torch_model(
    model_key: str,
    frame: pd.DataFrame,
    models_dir: Path,
    epochs: int,
    patience: int,
    seed: int,
    learning_rate: float = TORCH_LEARNING_RATE,
) -> tuple[Path, dict[str, object], dict[str, float]]:
    feature_cols = MODEL_FEATURES[model_key]
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    train_values = train[feature_cols].to_numpy(dtype=np.float32)
    val_values = val[feature_cols].to_numpy(dtype=np.float32)
    train_labels = train["label"].to_numpy(dtype=np.float32)
    val_labels = val["label"].to_numpy(dtype=np.float32)
    mean, scale = fit_normalizer(train_values)
    train_values = apply_normalizer(train_values, mean, scale)
    val_values = apply_normalizer(val_values, mean, scale)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(len(feature_cols), model_key)
    positive_count = max(1.0, float(train_labels.sum()))
    negative_count = max(1.0, float(len(train_labels) - train_labels.sum()))
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    train_tensor = torch.from_numpy(train_values)
    label_tensor = torch.from_numpy(train_labels)

    best_rank = (-float("inf"), -float("inf"), -float("inf"))
    best_state = None
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_tensor)
        train_loss = loss_function(logits, label_tensor)
        train_loss.backward()
        optimizer.step()

        train_probabilities = torch.sigmoid(logits.detach()).numpy()
        train_predictions = (train_probabilities > 0.5).astype(int)
        validation = _evaluate_torch_model(model, val_values, val_labels, loss_function)
        metrics = {
            "train_loss": float(train_loss.item()),
            "train_accuracy": float(accuracy_score(train_labels, train_predictions)),
            "val_loss": validation["loss"],
            "val_auc": validation["auc"],
            "val_accuracy": validation["accuracy"],
        }
        rank = (metrics["val_auc"], metrics["val_accuracy"], -metrics["val_loss"])
        if rank > best_rank:
            best_rank = rank
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == best_epoch:
            print(
                f"{model_key} epoca {epoch}/{epochs}: "
                f"val_auc={metrics['val_auc']:.4f}, "
                f"val_accuracy={metrics['val_accuracy']:.4f}"
            )
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"No se pudo entrenar el modelo {model_key}.")
    model.load_state_dict(best_state)
    checkpoint = make_checkpoint(
        model,
        model_key,
        feature_cols,
        mean,
        scale,
        best_epoch,
        best_metrics,
    )
    model_dir = models_dir / model_key
    path = model_dir / "model.pt"
    save_checkpoint(checkpoint, path)
    save_normalization(
        model_dir / "normalization.json",
        model_key,
        feature_cols,
        mean,
        scale,
        source_format="torch",
    )
    return path, checkpoint, best_metrics


def _fit_fold_torch_model(
    model_key: str,
    frame: pd.DataFrame,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    epochs: int,
    seed: int,
    learning_rate: float = TORCH_LEARNING_RATE,
) -> np.ndarray:
    feature_cols = MODEL_FEATURES[model_key]
    fit_frame = frame.iloc[fit_indices]
    holdout_frame = frame.iloc[holdout_indices]
    fit_values = fit_frame[feature_cols].to_numpy(dtype=np.float32)
    holdout_values = holdout_frame[feature_cols].to_numpy(dtype=np.float32)
    fit_labels = fit_frame["label"].to_numpy(dtype=np.float32)
    mean, scale = fit_normalizer(fit_values)
    fit_values = apply_normalizer(fit_values, mean, scale)
    holdout_values = apply_normalizer(holdout_values, mean, scale)

    torch.manual_seed(seed)
    model = build_model(len(feature_cols), model_key)
    positive_count = max(1.0, float(fit_labels.sum()))
    negative_count = max(1.0, float(len(fit_labels) - fit_labels.sum()))
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    fit_tensor = torch.from_numpy(fit_values)
    label_tensor = torch.from_numpy(fit_labels)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(fit_tensor), label_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(holdout_values))
        return torch.sigmoid(logits).cpu().numpy()


def _torch_checkpoint_probabilities(
    frame: pd.DataFrame,
    checkpoint: dict[str, object],
) -> np.ndarray:
    feature_cols = checkpoint["feature_cols"]
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    normalizer = checkpoint["normalizer"]
    values = apply_normalizer(values, normalizer["mean"], normalizer["scale"])
    with torch.no_grad():
        logits = checkpoint["model"](torch.from_numpy(values))
        return torch.sigmoid(logits).cpu().numpy()


def _build_stacking_dataset(
    reaction_df: pd.DataFrame,
    full_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    models_dir: Path,
    fold_epochs: int | dict[str, int] = RF_FOLD_EPOCHS,
    seed: int = 42,
    fold_learning_rate: float = TORCH_LEARNING_RATE,
) -> pd.DataFrame:
    call_ids = reaction_df["anon_id"].tolist()
    reaction_ordered = reaction_df.set_index("anon_id").loc[call_ids].reset_index()
    full_ordered = full_df.set_index("anon_id").loc[call_ids].reset_index()
    train_reaction = reaction_ordered[reaction_ordered["split"] == "train"].reset_index(drop=True)
    train_full = full_ordered[full_ordered["split"] == "train"].reset_index(drop=True)
    val_reaction = reaction_ordered[reaction_ordered["split"] == "val"].reset_index(drop=True)
    val_full = full_ordered[full_ordered["split"] == "val"].reset_index(drop=True)
    model_fold_epochs = (
        fold_epochs
        if isinstance(fold_epochs, dict)
        else {"time_predict": fold_epochs, "sustantive_predict": fold_epochs}
    )

    splitter = StratifiedKFold(n_splits=RF_FOLDS, shuffle=True, random_state=seed)
    oof_probabilities: dict[str, np.ndarray] = {}
    for model_key, frame in (
        ("time_predict", train_reaction),
        ("sustantive_predict", train_full),
    ):
        labels = frame["label"].to_numpy(dtype=int)
        probabilities = np.zeros(len(frame), dtype=np.float32)
        for fold, (fit_indices, holdout_indices) in enumerate(
            splitter.split(frame, labels)
        ):
            probabilities[holdout_indices] = _fit_fold_torch_model(
                model_key,
                frame,
                fit_indices,
                holdout_indices,
                epochs=model_fold_epochs[model_key],
                seed=seed + fold,
                learning_rate=fold_learning_rate,
            )
        oof_probabilities[model_key] = probabilities

    train_meta = _attach_acoustic_features(
        pd.DataFrame({
            "anon_id": train_reaction["anon_id"].to_numpy(),
            "label": train_reaction["label"].to_numpy(dtype=int),
            "split": "train",
            RF_FEATURE_COLS[0]: oof_probabilities["time_predict"],
            RF_FEATURE_COLS[1]: oof_probabilities["sustantive_predict"],
        }),
        acoustic_df,
    )
    checkpoints = {
        "time_predict": load_checkpoint(models_dir / "time_predict" / "model.pt"),
        "sustantive_predict": load_checkpoint(models_dir / "sustantive_predict" / "model.pt"),
    }
    val_meta = _attach_acoustic_features(
        pd.DataFrame({
            "anon_id": val_reaction["anon_id"].to_numpy(),
            "label": val_reaction["label"].to_numpy(dtype=int),
            "split": "val",
            RF_FEATURE_COLS[0]: _torch_checkpoint_probabilities(
                val_reaction, checkpoints["time_predict"]
            ),
            RF_FEATURE_COLS[1]: _torch_checkpoint_probabilities(
                val_full, checkpoints["sustantive_predict"]
            ),
        }),
        acoustic_df,
    )
    return pd.concat([train_meta, val_meta], ignore_index=True)


def train_random_forest(
    reaction_df: pd.DataFrame,
    full_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    output_dir: str | Path = BASE_DIR,
    fold_epochs: int | dict[str, int] = RF_FOLD_EPOCHS,
    seed: int = 42,
    fold_learning_rate: float = TORCH_LEARNING_RATE,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    stacking_df = _build_stacking_dataset(
        reaction_df,
        full_df,
        acoustic_df,
        models_dir=models_dir,
        fold_epochs=fold_epochs,
        seed=seed,
        fold_learning_rate=fold_learning_rate,
    )
    train = stacking_df[stacking_df["split"] == "train"]
    val = stacking_df[stacking_df["split"] == "val"]
    forest = RandomForestClassifier(
        min_samples_split=5,
        max_depth=4,
        n_trees=30,
        X_features_fraction=1.0,
        X_obs_fraction=1.0,
        random_state=seed,
    )
    forest.fit(train[RF_FEATURE_COLS], train["label"])
    val_probabilities = forest.predict_proba(val[RF_FEATURE_COLS])[:, 1]
    val_predictions = (val_probabilities >= 0.5).astype(int)
    metrics = {
        "val_auc": float(roc_auc_score(val["label"], val_probabilities)),
        "val_accuracy": float(accuracy_score(val["label"], val_predictions)),
        "n_trees": forest.n_trees,
        "n_train": len(train),
        "n_val": len(val),
        "stacking": "5-fold out-of-fold base predictions",
        "fold_epochs": fold_epochs,
        "fold_learning_rate": fold_learning_rate,
    }
    model_dir = models_dir / "random_forest"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump({
        "format": "nexus-random-forest-v1",
        "model_key": "random_forest",
        "feature_cols": RF_FEATURE_COLS,
        "forest": forest,
        "base_model_paths": {
            "time_predict": "time_predict/model.pt",
            "sustantive_predict": "sustantive_predict/model.pt",
        },
        "acoustic_model_paths": {
            "mini_ast": "../../resultados_audio/mini_ast_best.pt",
            "wav2vec": "../../wav2vec2_finetuned_model_run",
        },
        "metrics": metrics,
    }, model_path)
    save_normalization(
        model_dir / "normalization.json",
        "random_forest",
        RF_FEATURE_COLS,
        [0.0, 0.0],
        [1.0, 1.0],
        source_format="custom_random_forest",
        method="not_required",
    )

    metrics_path = output_dir / "training_metrics.json"
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary["default_model"] = "random_forest"
    summary["random_forest"] = metrics
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    registry_path = models_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["default_model"] = "random_forest"
    registry["models"] = [
        entry for entry in registry["models"] if entry["id"] != "random_forest"
    ]
    registry["models"].append({
        "id": "random_forest",
        "kind": "custom_random_forest",
        "model": "random_forest/model.joblib",
        "normalization": "random_forest/normalization.json",
        "inputs": [
            "time_predict",
            "sustantive_predict",
            "mini_ast_predict",
            "wav2vec_predict",
        ],
        "feature_cols": RF_FEATURE_COLS,
        "output": "synthetic_probability",
    })
    registry["future_nodes"] = [
        node for node in registry["future_nodes"] if node["id"] != "random_forest"
    ]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(
        f"Random forest: val_auc={metrics['val_auc']:.4f}, "
        f"val_accuracy={metrics['val_accuracy']:.4f} -> {model_path}"
    )
    return {"path": model_path, "metrics": metrics}


def _write_model_registry(
    models_dir: Path,
    results: dict[str, object],
    best_model_key: str,
) -> None:
    acoustic_metadata = [
        {
            "id": "mini_ast_predict",
            "kind": "torch_audio_classifier",
            "model": "../../resultados_audio/mini_ast_best.pt",
            "normalization": "mini_ast_predict/normalization.json",
            "feature_cols": ["mini_ast_probability"],
            "output": "synthetic_probability",
        },
        {
            "id": "wav2vec_predict",
            "kind": "wav2vec2_audio_classifier",
            "model": "../../wav2vec2_finetuned_model_run",
            "normalization": "wav2vec_predict/normalization.json",
            "feature_cols": ["wav2vec_probability"],
            "output": "synthetic_probability",
        },
    ]
    model_entries = []
    for model_key, value in results.items():
        checkpoint = value["checkpoint"]
        model_entries.append({
            "id": model_key,
            "kind": "torch_binary_classifier",
            "model": f"{model_key}/model.pt",
            "normalization": f"{model_key}/normalization.json",
            "feature_cols": checkpoint["feature_cols"],
            "output": "synthetic_probability",
        })
    model_entries.extend(acoustic_metadata)
    model_entries.extend([
        {
            "id": "best_model",
            "kind": "torch_alias",
            "model": "best_model/model.pt",
            "normalization": "best_model/normalization.json",
            "source_model": best_model_key,
            "feature_cols": results[best_model_key]["checkpoint"]["feature_cols"],
            "output": "synthetic_probability",
        },
        {
            "id": "legacy_model",
            "kind": "sklearn_pipeline",
            "model": "legacy_model/model.joblib",
            "normalization": "legacy_model/normalization.json",
            "feature_cols": CALL_FEATURE_COLS,
            "output": "synthetic_probability",
        },
    ])
    registry = {
        "version": 1,
        "models_root": "nexus/models",
        "models": model_entries,
        "future_nodes": [
            {
                "id": "decision_tree",
                "status": "planned",
                "inputs": ["time_predict", "sustantive_predict"],
            },
            {
                "id": "random_forest",
                "status": "planned",
                "inputs": [
                    "time_predict",
                    "sustantive_predict",
                    "mini_ast_predict",
                    "wav2vec_predict",
                ],
            },
        ],
    }
    (models_dir / "registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )


def train_torch_models(
    datasets: dict[str, pd.DataFrame],
    output_dir: str | Path = BASE_DIR,
    epochs: int | dict[str, int] = EPOCHS,
    patience: int = 40,
    seed: int = 42,
    learning_rate: float = TORCH_LEARNING_RATE,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    results: dict[str, object] = {}
    model_epochs = (
        epochs
        if isinstance(epochs, dict)
        else {model_key: epochs for model_key in datasets}
    )
    for model_key, frame in datasets.items():
        path, checkpoint, metrics = _train_torch_model(
            model_key,
            frame,
            models_dir,
            model_epochs[model_key],
            patience,
            seed,
            learning_rate=learning_rate,
        )
        results[model_key] = {
            "path": path,
            "checkpoint": checkpoint,
            "metrics": metrics,
        }

    best_model_key = max(
        results,
        key=lambda key: (
            results[key]["metrics"]["val_auc"],
            results[key]["metrics"]["val_accuracy"],
        ),
    )
    best_checkpoint = results[best_model_key]["checkpoint"]
    best_dir = models_dir / "best_model"
    best_path = best_dir / "model.pt"
    save_checkpoint(best_checkpoint, best_path)
    save_normalization(
        best_dir / "normalization.json",
        "best_model",
        best_checkpoint["feature_cols"],
        best_checkpoint["normalizer"]["mean"],
        best_checkpoint["normalizer"]["scale"],
        source_format="torch_alias",
    )
    summary = {
        "best_model": best_model_key,
        "epochs": model_epochs,
        "learning_rate": learning_rate,
        "models": {
            key: {
                "file": Path(value["path"]).relative_to(output_dir).as_posix(),
                "epoch": value["checkpoint"]["epoch"],
                "metrics": value["metrics"],
            }
            for key, value in results.items()
        },
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_model_registry(models_dir, results, best_model_key)
    print(f"Mejor modelo: {best_model_key} -> {best_path}")
    return {"best_model": best_model_key, "best_path": best_path, "models": results}


def _save_legacy_pipeline(call_df: pd.DataFrame, output_path: Path) -> None:
    train = call_df[call_df.split == "train"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=LEARNING_RATE,
            max_iter=EPOCHS,
            alpha=L2_REGULARIZATION,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    pipe.fit(train[CALL_FEATURE_COLS], train["label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_cols": CALL_FEATURE_COLS}, output_path)
    scaler = pipe.named_steps["scaler"]
    save_normalization(
        output_path.parent / "normalization.json",
        "legacy_model",
        CALL_FEATURE_COLS,
        scaler.mean_,
        scaler.scale_,
        source_format="sklearn_pipeline",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos PyTorch de Nexus.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--turns-dir", type=Path, default=TURNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-fold-epochs", type=int, default=RF_FOLD_EPOCHS)
    parser.add_argument("--time-epochs", type=int, default=None)
    parser.add_argument("--sustantive-epochs", type=int, default=None)
    parser.add_argument("--time-fold-epochs", type=int, default=None)
    parser.add_argument("--sustantive-fold-epochs", type=int, default=None)
    parser.add_argument(
        "--torch-learning-rate",
        type=float,
        default=TORCH_LEARNING_RATE,
    )
    args = parser.parse_args()

    model_epochs = {
        "time_predict": args.time_epochs if args.time_epochs is not None else args.epochs,
        "sustantive_predict": (
            args.sustantive_epochs
            if args.sustantive_epochs is not None
            else args.epochs
        ),
    }
    fold_epochs = {
        "time_predict": (
            args.time_fold_epochs
            if args.time_fold_epochs is not None
            else args.rf_fold_epochs
        ),
        "sustantive_predict": (
            args.sustantive_fold_epochs
            if args.sustantive_fold_epochs is not None
            else args.rf_fold_epochs
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    turn_df = build_turn_dataset(args.manifest, args.turns_dir)
    export_cols = ["reaction_time", "label"]
    dataset_path = args.output_dir / "dataset.csv"
    turn_df[export_cols].to_csv(dataset_path, index=False)
    print(f"Dataset extraido de los JSON guardado en {dataset_path} "
          f"({len(turn_df)} filas, {len(export_cols)} columnas)")

    reaction_df = build_call_dataset(turn_df)
    full_df = build_full_call_dataset(args.manifest, args.turns_dir)
    full_df = full_df[full_df["anon_id"].isin(reaction_df["anon_id"])]
    acoustic_df = build_acoustic_dataset(
        args.manifest,
        cache_path=args.output_dir / "acoustic_predictions.csv",
    )
    datasets = {
        "time_predict": reaction_df,
        "sustantive_predict": full_df,
    }
    n_calls = len(reaction_df)
    print(f"Llamadas: {n_calls}  ->  turnos de caller usados: {len(turn_df)}")
    print(f"Split (por llamada): {sum(reaction_df.split == 'train')} train / "
          f"{sum(reaction_df.split == 'val')} val")

    _save_legacy_pipeline(reaction_df, args.output_dir / "models" / "legacy_model" / "model.joblib")
    train_torch_models(
        datasets,
        output_dir=args.output_dir,
        epochs=model_epochs,
        patience=args.patience,
        seed=args.seed,
        learning_rate=args.torch_learning_rate,
    )
    train_random_forest(
        reaction_df,
        full_df,
        acoustic_df,
        output_dir=args.output_dir,
        fold_epochs=fold_epochs,
        seed=args.seed,
        fold_learning_rate=args.torch_learning_rate,
    )


if __name__ == "__main__":
    main()
