import csv
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from audio_features import extract_channel_audio
from features import extract_features, extract_turn_features, load_turns, merge_turns
from predict import predict_call

ROOT = os.path.dirname(os.path.abspath(__file__))
TURNS_DIR = os.path.join(os.path.dirname(ROOT), "hackmty26", "turns")


def test_extract_channel_audio_concatenates_only_requested_turns():
    audio = np.arange(20, dtype=np.float32).reshape(10, 2)
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 1.0, "end": 3.0},
        {"channel": 0, "start": 4.0, "end": 5.0},
    ]

    extracted = extract_channel_audio(audio, turns, sample_rate=2, channel=0)

    np.testing.assert_array_equal(
        extracted,
        np.array([4, 6, 8, 10, 16, 18], dtype=np.float32),
    )


# ---------- features.py ----------

def test_merge_turns_merges_close_same_channel_gaps():
    turns = [
        {"channel": 0, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 2.3, "end": 4.0},  # gap 0.3 < 0.6 -> se fusiona
    ]
    merged = merge_turns(turns, gap_threshold=0.6)
    assert len(merged) == 1
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 4.0


def test_merge_turns_keeps_far_same_channel_gaps_separate():
    turns = [
        {"channel": 0, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 3.0, "end": 4.0},  # gap 1.0 > 0.6 -> no se fusiona
    ]
    merged = merge_turns(turns, gap_threshold=0.6)
    assert len(merged) == 2


def test_merge_turns_never_merges_across_channels():
    turns = [
        {"channel": 0, "start": 0.0, "end": 2.0},
        {"channel": 1, "start": 2.1, "end": 4.0},  # gap chico pero canal distinto
    ]
    merged = merge_turns(turns, gap_threshold=0.6)
    assert len(merged) == 2


def test_extract_features_returns_all_expected_keys():
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 3.0, "end": 5.0},
        {"channel": 1, "start": 6.0, "end": 8.0},
    ]
    feats = extract_features(turns)
    expected_prefixes = ["reaction", "agent_reaction", "caller_turn_dur", "agent_turn_dur"]
    for prefix in expected_prefixes:
        for stat in ("mean", "std", "min", "max", "median"):
            assert f"{prefix}_{stat}" in feats
    assert set(["n_turns_caller", "n_turns_agent", "overlap_count", "overlap_ratio"]) <= feats.keys()


def test_extract_features_handles_single_speaker_only():
    # Solo el agente habla, el caller nunca responde: no debe crashear.
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 1, "start": 3.0, "end": 5.0},
    ]
    feats = extract_features(turns)
    assert feats["n_turns_caller"] == 0
    assert feats["reaction_mean"] == 0.0  # fallback de _stats para lista vacia


def test_extract_features_detects_overlap():
    turns = [
        {"channel": 1, "start": 0.0, "end": 5.0},
        {"channel": 0, "start": 4.0, "end": 6.0},  # empieza antes de que termine el agente -> overlap
    ]
    feats = extract_features(turns)
    assert feats["overlap_count"] == 1
    assert feats["overlap_ratio"] == 1.0


def test_extract_features_all_values_are_finite_numbers():
    turns = [{"channel": 1, "start": 0.0, "end": 1.0}]
    feats = extract_features(turns)
    for k, v in feats.items():
        assert isinstance(v, (int, float)), f"{k} no es numerico: {v!r}"
        assert v == v, f"{k} es NaN"  # NaN != NaN


# ---------- features.py: extract_turn_features (arquitectura por turno) ----------

def test_extract_turn_features_returns_one_row_per_caller_turn():
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 3.0, "end": 5.0},
        {"channel": 1, "start": 6.0, "end": 8.0},
        {"channel": 0, "start": 9.0, "end": 10.0},
    ]
    rows = extract_turn_features(turns)
    assert len(rows) == 2  # solo los 2 turnos del caller (channel 0)


def test_extract_turn_features_expected_keys():
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 3.0, "end": 5.0},
    ]
    rows = extract_turn_features(turns)
    expected_keys = {"turn_number", "turn_start", "turn_end", "prev_turn_end",
                      "reaction_time", "is_after_agent", "overlap",
                      "turn_duration", "call_progress"}
    assert set(rows[0].keys()) == expected_keys


def test_extract_turn_features_reaction_time_after_agent():
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 3.5, "end": 5.0},
    ]
    rows = extract_turn_features(turns)
    assert rows[0]["reaction_time"] == pytest.approx(1.5)
    assert rows[0]["is_after_agent"] == 1
    assert rows[0]["turn_duration"] == pytest.approx(1.5)


def test_extract_turn_features_first_turn_reaction_is_time_since_call_start():
    turns = [{"channel": 0, "start": 4.0, "end": 6.0}]
    rows = extract_turn_features(turns)
    assert rows[0]["reaction_time"] == pytest.approx(4.0)
    assert rows[0]["is_after_agent"] == 0


def test_extract_turn_features_detects_overlap():
    turns = [
        {"channel": 1, "start": 0.0, "end": 5.0},
        {"channel": 0, "start": 4.0, "end": 6.0},  # empieza antes de que el agente termine
    ]
    rows = extract_turn_features(turns)
    assert rows[0]["overlap"] == 1
    assert rows[0]["reaction_time"] < 0


def test_extract_turn_features_call_progress_is_between_0_and_1():
    turns = [
        {"channel": 1, "start": 0.0, "end": 2.0},
        {"channel": 0, "start": 3.0, "end": 5.0},
        {"channel": 1, "start": 6.0, "end": 20.0},
    ]
    rows = extract_turn_features(turns)
    assert 0.0 <= rows[0]["call_progress"] <= 1.0


def test_extract_turn_features_empty_when_no_caller_turns():
    turns = [{"channel": 1, "start": 0.0, "end": 2.0}]
    rows = extract_turn_features(turns)
    assert rows == []


# ---------- manifest / split integrity ----------

@pytest.fixture(scope="module")
def manifest_rows():
    with open(os.path.join(ROOT, "manifest.csv")) as f:
        return list(csv.DictReader(f))


def test_manifest_has_no_duplicate_ids(manifest_rows):
    ids = [r["anon_id"] for r in manifest_rows]
    assert len(ids) == len(set(ids))


def test_manifest_has_no_train_val_leakage(manifest_rows):
    train_ids = {r["anon_id"] for r in manifest_rows if r["split"] == "train"}
    val_ids = {r["anon_id"] for r in manifest_rows if r["split"] == "val"}
    assert train_ids.isdisjoint(val_ids)


def test_manifest_split_is_roughly_80_20(manifest_rows):
    n_train = sum(1 for r in manifest_rows if r["split"] == "train")
    ratio = n_train / len(manifest_rows)
    assert 0.75 <= ratio <= 0.85


def test_all_manifest_turns_files_exist(manifest_rows):
    for row in manifest_rows:
        path = os.path.join(TURNS_DIR, f"{row['anon_id']}.json")
        assert os.path.exists(path), f"falta {path}"


def test_manifest_labels_are_only_human_or_synthetic(manifest_rows):
    labels = {r["label"] for r in manifest_rows}
    assert labels <= {"human", "synthetic"}


def test_turn_level_dataset_has_no_call_leakage_across_split(manifest_rows):
    # Cada llamada genera ~14 filas (una por turno); si el split se hiciera
    # por turno en vez de por llamada, una misma llamada podria terminar
    # con turnos en train Y en val. Esto verifica que NO pasa.
    train_ids, val_ids = set(), set()
    for row in manifest_rows:
        turns = load_turns(os.path.join(TURNS_DIR, f"{row['anon_id']}.json"))
        if not extract_turn_features(turns):
            continue
        (train_ids if row["split"] == "train" else val_ids).add(row["anon_id"])
    assert train_ids.isdisjoint(val_ids)


# ---------- predict.py / legacy model ----------

@pytest.fixture(scope="module")
def model_bundle():
    return joblib.load(os.path.join(ROOT, "models", "legacy_model", "model.joblib"))


def test_predict_call_confidence_is_a_valid_probability(model_bundle):
    result = predict_call(os.path.join(TURNS_DIR, "call_0181ce113ebe.json"), model_bundle=model_bundle)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_call_label_matches_confidence_threshold(model_bundle):
    result = predict_call(os.path.join(TURNS_DIR, "call_0181ce113ebe.json"), model_bundle=model_bundle)
    expected_label = "synthetic" if result["confidence"] > 0.5 else "human"
    assert result["label"] == expected_label


def test_predict_call_uses_saved_feature_order(model_bundle):
    # El modelo espera 5 stats agregadas por llamada (reaction_mean, std,
    # min, max, median), no las columnas crudas por turno.
    expected = {"reaction_mean", "reaction_std", "reaction_min",
                "reaction_max", "reaction_median"}
    assert set(model_bundle["feature_cols"]) == expected


def test_predict_call_matches_manual_aggregation(model_bundle):
    import numpy as np

    turns_path = os.path.join(TURNS_DIR, "call_0181ce113ebe.json")
    turns = load_turns(turns_path)
    reaction_times = np.array([r["reaction_time"] for r in extract_turn_features(turns)])

    pipe, feature_cols = model_bundle["pipeline"], model_bundle["feature_cols"]
    manual_feats = pd.DataFrame([{
        "reaction_mean": reaction_times.mean(),
        "reaction_std": reaction_times.std(),
        "reaction_min": reaction_times.min(),
        "reaction_max": reaction_times.max(),
        "reaction_median": float(np.median(reaction_times)),
    }])[feature_cols]
    expected_confidence = float(pipe.predict_proba(manual_feats)[0, 1])

    result = predict_call(turns_path, model_bundle=model_bundle)
    assert result["confidence"] == pytest.approx(expected_confidence)
    assert result["n_turns"] == len(reaction_times)


def test_batch_accuracy_on_val_set_matches_expected_range(model_bundle):
    with open(os.path.join(ROOT, "manifest.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]

    correct = 0
    for row in rows:
        result = predict_call(
            os.path.join(TURNS_DIR, f"{row['anon_id']}.json"), model_bundle=model_bundle
        )
        correct += result["label"] == row["label"]

    accuracy = correct / len(rows)
    # Regression test: si un cambio futuro rompe el pipeline y la accuracy
    # se desploma, este test debe fallar.
    assert accuracy >= 0.80, f"accuracy inesperadamente baja: {accuracy:.3f}"
