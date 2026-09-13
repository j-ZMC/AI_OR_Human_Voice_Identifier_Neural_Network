"""Prueba si el modelo tiene un sesgo sistematico hacia una clase (human o
synthetic), corriendo cross-validation repetida y comparando:
  - recall/precision por clase (por que un accuracy global alto puede
    esconder que el modelo falla mucho mas en una clase que en otra)
  - la proporcion de predicciones vs. la proporcion real de cada clase
    (por si el modelo tiende a "jalar" hacia la clase mayoritaria)

Umbral usado para decidir "hay sesgo": una diferencia de recall entre
clases mayor a 10 puntos porcentuales se marca como sesgo notable. Es un
umbral arbitrario pero razonable para reportar, no una ley matematica.
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import recall_score, precision_score

from train import (build_turn_dataset, build_call_dataset, CALL_FEATURE_COLS,
                    EPOCHS, LEARNING_RATE, L2_REGULARIZATION)

RECALL_GAP_THRESHOLD = 0.10
N_SPLITS = 5
N_REPEATS = 10


def make_pipeline():
    return Pipeline([
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


def main():
    turn_df = build_turn_dataset()
    call_df = build_call_dataset(turn_df)

    real_synthetic_rate = call_df["label"].mean()
    print(f"Distribucion real: {(1 - real_synthetic_rate):.1%} human / "
          f"{real_synthetic_rate:.1%} synthetic\n")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    recall_human, recall_synth = [], []
    precision_human, precision_synth = [], []
    pred_synth_rate = []

    for tr_idx, va_idx in cv.split(call_df, call_df["label"]):
        train, val = call_df.iloc[tr_idx], call_df.iloc[va_idx]
        pipe = make_pipeline()
        pipe.fit(train[CALL_FEATURE_COLS], train["label"])
        preds = (pipe.predict_proba(val[CALL_FEATURE_COLS])[:, 1] > 0.5).astype(int)

        recall_human.append(recall_score(val["label"], preds, pos_label=0))
        recall_synth.append(recall_score(val["label"], preds, pos_label=1))
        precision_human.append(precision_score(val["label"], preds, pos_label=0, zero_division=0))
        precision_synth.append(precision_score(val["label"], preds, pos_label=1, zero_division=0))
        pred_synth_rate.append(preds.mean())

    n_folds = N_SPLITS * N_REPEATS
    print(f"{n_folds} particiones evaluadas ({N_SPLITS}-fold x {N_REPEATS} repeticiones)\n")

    print(f"{'Metrica':<22}{'human':<12}{'synthetic':<12}{'gap':<10}")
    recall_gap = abs(np.mean(recall_human) - np.mean(recall_synth))
    print(f"{'Recall':<22}{np.mean(recall_human):<12.3f}{np.mean(recall_synth):<12.3f}{recall_gap:<10.3f}")
    print(f"{'Precision':<22}{np.mean(precision_human):<12.3f}{np.mean(precision_synth):<12.3f}"
          f"{abs(np.mean(precision_human) - np.mean(precision_synth)):<10.3f}")

    print(f"\n% predicho como synthetic: {np.mean(pred_synth_rate):.1%} "
          f"(la proporcion real es {real_synthetic_rate:.1%}, "
          f"diferencia de {abs(np.mean(pred_synth_rate) - real_synthetic_rate):.1%})")

    print("\n=== Veredicto ===")
    if recall_gap > RECALL_GAP_THRESHOLD:
        peor = "human" if np.mean(recall_human) < np.mean(recall_synth) else "synthetic"
        print(f"SESGO NOTABLE: el gap de recall entre clases es {recall_gap:.1%} "
              f"(> {RECALL_GAP_THRESHOLD:.0%}). El modelo detecta peor la clase '{peor}'.")
    else:
        print(f"Sin sesgo notable: el gap de recall entre clases es {recall_gap:.1%} "
              f"(<= {RECALL_GAP_THRESHOLD:.0%}).")


if __name__ == "__main__":
    main()
