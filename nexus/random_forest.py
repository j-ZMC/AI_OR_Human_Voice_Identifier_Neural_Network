from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd


class RandomForestTree:
    def __init__(
        self,
        Y,
        X: pd.DataFrame,
        min_samples_split: int = 20,
        max_depth: int = 5,
        depth: int = 0,
        X_features_fraction: float = 1.0,
        node_type: str = "root",
        rule: str = "",
        rng: random.Random | None = None,
    ) -> None:
        self.Y = np.asarray(Y, dtype=int)
        self.X = X.reset_index(drop=True).astype(float).copy()
        self.min_samples_split = max(2, int(min_samples_split))
        self.max_depth = max(1, int(max_depth))
        self.depth = depth
        self.features = list(self.X.columns)
        self.node_type = node_type
        self.rule = rule
        self.n_features = len(self.features)
        self.X_features_fraction = float(X_features_fraction)
        self.rng = rng or random.Random()
        self.n = len(self.Y)
        self.counts = Counter(self.Y.tolist())
        self.gini_impurity = self.get_gini()
        self.yhat = int(self.Y.mean() >= 0.5) if self.n else 0
        self.probability = float(self.Y.mean()) if self.n else 0.5
        self.left: RandomForestTree | None = None
        self.right: RandomForestTree | None = None
        self.best_feature: str | None = None
        self.best_value: float | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=2).astype(float)
        probabilities = counts / len(y)
        return float(1.0 - np.sum(probabilities ** 2))

    def get_gini(self) -> float:
        return self._gini(self.Y)

    def _candidate_features(self) -> list[str]:
        n_selected = int(round(self.n_features * self.X_features_fraction))
        n_selected = max(1, min(self.n_features, n_selected))
        return self.rng.sample(self.features, n_selected)

    def best_split(self) -> tuple[str | None, float | None]:
        if self.n < 2 or self.gini_impurity == 0.0:
            return None, None
        base_gini = self.gini_impurity
        best_gain = 0.0
        best_feature = None
        best_value = None
        for feature in self._candidate_features():
            values = self.X[feature].to_numpy(dtype=float)
            unique_values = np.unique(values[np.isfinite(values)])
            if len(unique_values) < 2:
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_size = int(left_mask.sum())
                right_size = int(right_mask.sum())
                weighted_gini = (
                    left_size * self._gini(self.Y[left_mask])
                    + right_size * self._gini(self.Y[right_mask])
                ) / self.n
                gain = base_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = float(threshold)
        return best_feature, best_value

    def grow_tree(self) -> None:
        if self.depth >= self.max_depth or self.n < self.min_samples_split:
            return
        best_feature, best_value = self.best_split()
        if best_feature is None or best_value is None:
            return
        values = self.X[best_feature].to_numpy(dtype=float)
        left_mask = values <= best_value
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            return
        self.best_feature = best_feature
        self.best_value = best_value
        self.left = RandomForestTree(
            self.Y[left_mask],
            self.X.loc[left_mask],
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            X_features_fraction=self.X_features_fraction,
            node_type="left_node",
            rule=f"{best_feature} <= {best_value:.6g}",
            rng=self.rng,
        )
        self.right = RandomForestTree(
            self.Y[right_mask],
            self.X.loc[right_mask],
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            X_features_fraction=self.X_features_fraction,
            node_type="right_node",
            rule=f"{best_feature} > {best_value:.6g}",
            rng=self.rng,
        )
        self.left.grow_tree()
        self.right.grow_tree()

    def predict_obs(self, values: dict[str, float]) -> float:
        node = self
        while node.best_feature is not None and node.best_value is not None:
            next_node = node.left if values[node.best_feature] <= node.best_value else node.right
            if next_node is None:
                break
            node = next_node
        return node.probability

    def predict(self, X: pd.DataFrame) -> list[int]:
        return [int(self.predict_obs(row.to_dict()) >= 0.5) for _, row in X.iterrows()]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = [self.predict_obs(row.to_dict()) for _, row in X.iterrows()]
        probabilities = np.asarray(probabilities, dtype=float)
        return np.column_stack((1.0 - probabilities, probabilities))

    def print_info(self, width: int = 4) -> None:
        indentation = "-" * int(self.depth * width ** 1.5)
        if self.node_type == "root":
            print("Root")
        else:
            print(f"|{indentation} Split rule: {self.rule}")
        print(f"{' ' * len(indentation)}   | GINI impurity: {self.gini_impurity:.3f}")
        print(f"{' ' * len(indentation)}   | Class distribution: {dict(self.counts)}")
        print(f"{' ' * len(indentation)}   | Predicted class: {self.yhat}")

    def print_tree(self) -> None:
        self.print_info()
        if self.left is not None:
            self.left.print_tree()
        if self.right is not None:
            self.right.print_tree()


class RandomForestClassifier:
    def __init__(
        self,
        Y=None,
        X: pd.DataFrame | None = None,
        min_samples_split: int = 20,
        max_depth: int = 5,
        n_trees: int = 30,
        X_features_fraction: float = 1.0,
        X_obs_fraction: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.Y = None if Y is None else np.asarray(Y, dtype=int)
        self.X = None if X is None else X.reset_index(drop=True).astype(float).copy()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.X_features_fraction = X_features_fraction
        self.X_obs_fraction = X_obs_fraction
        self.random_state = random_state
        self.random_forest: list[RandomForestTree] = []
        self.features: list[str] = [] if X is None else list(X.columns)

    def fit(self, X: pd.DataFrame, Y) -> RandomForestClassifier:
        self.X = X.reset_index(drop=True).astype(float).copy()
        self.Y = np.asarray(Y, dtype=int)
        self.features = list(self.X.columns)
        self.grow_random_forest()
        return self

    def bootstrap_sample(self) -> tuple[pd.DataFrame, np.ndarray]:
        if self.X is None or self.Y is None:
            raise RuntimeError("El bosque necesita datos antes de entrenar.")
        sample_size = max(1, int(round(len(self.X) * self.X_obs_fraction)))
        indexes = [self._rng.randrange(len(self.X)) for _ in range(sample_size)]
        return self.X.iloc[indexes].reset_index(drop=True), self.Y[indexes]

    def grow_random_forest(self) -> None:
        if self.X is None or self.Y is None:
            raise RuntimeError("El bosque necesita datos antes de entrenar.")
        if len(self.X) != len(self.Y) or len(self.X) == 0:
            raise ValueError("X e Y deben tener la misma cantidad de filas no vacia.")
        self._rng = random.Random(self.random_state)
        self.random_forest = []
        for _ in range(self.n_trees):
            bootstrap_x, bootstrap_y = self.bootstrap_sample()
            tree_rng = random.Random(self._rng.randrange(2**32))
            tree = RandomForestTree(
                bootstrap_y,
                bootstrap_x,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                X_features_fraction=self.X_features_fraction,
                rng=tree_rng,
            )
            tree.grow_tree()
            self.random_forest.append(tree)

    def tree_predictions(self, X: pd.DataFrame) -> list[list[int]]:
        return [tree.predict(X[self.features]) for tree in self.random_forest]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.random_forest:
            raise RuntimeError("El bosque no ha sido entrenado.")
        probabilities = [tree.predict_proba(X[self.features])[:, 1] for tree in self.random_forest]
        synthetic_probability = np.mean(np.asarray(probabilities), axis=0)
        return np.column_stack((1.0 - synthetic_probability, synthetic_probability))

    def predict(self, X: pd.DataFrame) -> list[int]:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int).tolist()

    def print_trees(self) -> None:
        for index, tree in enumerate(self.random_forest, start=1):
            print(f"------ Tree {index} ------")
            tree.print_tree()