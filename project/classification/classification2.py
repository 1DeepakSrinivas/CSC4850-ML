"""
Author: Deepak Govindarajan
Date: 2025-11-08
CSC 4850 Machine Learning
Project: Classification 
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Detects available Apple Silicon GPU support.
import numpy as np
import pandas as pd
import torch

MISSING_VALUE = 1.0e99


# Select Apple GPU when possible, else CPU.
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Read feature and label tensors from disk.
def load_data(data_path: str, label_path: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    frame = pd.read_csv(data_path, header=None, sep=r"\s+").astype(np.float32)
    features = torch.from_numpy(frame.to_numpy(copy=True))

    labels = None
    if label_path and os.path.exists(label_path):
        label_series = pd.read_csv(label_path, header=None).squeeze()
        if label_series.ndim == 0:
            label_series = pd.Series([label_series])
        labels = torch.from_numpy(label_series.to_numpy(copy=True)).long()
    return features, labels


# Replace dataset sentinel values with column means.
def handle_missing_values(
    data: torch.Tensor, stats: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    clean = data.clone()
    missing = torch.isclose(clean, torch.tensor(MISSING_VALUE, dtype=clean.dtype), atol=1e-8)
    clean[missing] = torch.nan

    if stats is None:
        means = torch.nanmean(clean, dim=0)
        means = torch.nan_to_num(means, nan=0.0)
    else:
        means = stats

    nan_mask = torch.isnan(clean)
    if nan_mask.any():
        clean[nan_mask] = means.repeat(clean.size(0), 1)[nan_mask]
    return clean, means


# Apply standard score normalization to features.
def standardize(
    train: torch.Tensor, test: Optional[torch.Tensor] = None, stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    if stats is None:
        mean = train.mean(dim=0)
        std = train.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
    else:
        mean, std = stats

    train_norm = (train - mean) / std
    if test is None:
        return train_norm, None, (mean, std)
    test_norm = (test - mean) / std
    return train_norm, test_norm, (mean, std)


@dataclass
class TreeNode:
    prediction: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


# Single decision tree used inside the forest ensemble.
class DecisionTree:
    # Store tree hyperparameters.
    def __init__(self, max_depth: int, min_samples_split: int, min_samples_leaf: int, max_features: int, n_classes: int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_classes = n_classes
        self.root: Optional[TreeNode] = None
        self.features: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None

    # Train the tree on bootstrap samples.
    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels
        indices = torch.arange(features.size(0))
        self.root = self._build(indices, depth=0)

    # Predict class ids for a batch of rows.
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        preds = torch.empty(features.size(0), dtype=torch.long)
        for idx in range(features.size(0)):
            preds[idx] = self._predict_one(features[idx])
        return preds

    # Walk the tree to classify a single row.
    def _predict_one(self, row: torch.Tensor) -> int:
        node = self.root
        while node and node.feature is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction if node else 0

    # Recursively grow the tree from the current split.
    def _build(self, indices: torch.Tensor, depth: int) -> TreeNode:
        labels = self.labels[indices]
        majority = int(torch.mode(labels).values.item())

        if (
            depth >= self.max_depth
            or indices.numel() < self.min_samples_split
            or torch.unique(labels).numel() == 1
        ):
            return TreeNode(prediction=majority)

        best_feature, best_threshold, gain = self._best_split(indices)
        if best_feature is None or gain <= 0.0:
            return TreeNode(prediction=majority)

        values = self.features[indices, best_feature]
        left_mask = values <= best_threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return TreeNode(prediction=majority)

        left_child = self._build(indices[left_mask], depth + 1)
        right_child = self._build(indices[right_mask], depth + 1)
        return TreeNode(
            prediction=majority,
            feature=best_feature,
            threshold=float(best_threshold),
            left=left_child,
            right=right_child,
        )

    # Evaluate Gini gain for feature thresholds.
    def _best_split(self, indices: torch.Tensor) -> Tuple[Optional[int], Optional[float], float]:
        best_feature = None
        best_threshold = None
        best_gain = 0.0

        current_gini = self._gini(self.labels[indices])
        feature_count = self.features.size(1)
        candidates = torch.randperm(feature_count)[: self.max_features]

        for feature in candidates:
            values = self.features[indices, feature]
            unique_vals = torch.unique(values)
            if unique_vals.numel() <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                left_size = left_mask.sum()
                right_size = right_mask.sum()
                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue

                left_gini = self._gini(self.labels[indices[left_mask]])
                right_gini = self._gini(self.labels[indices[right_mask]])
                weighted = (left_size * left_gini + right_size * right_gini) / indices.numel()
                gain = current_gini - weighted

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    # Compute Gini impurity for a label set.
    def _gini(self, labels: torch.Tensor) -> float:
        counts = torch.bincount(labels, minlength=self.n_classes).float()
        probs = counts / labels.numel()
        return float(1.0 - torch.sum(probs ** 2))


# Ensemble of decision trees trained with bagging.
class RandomForestTorch:
    # Capture forest-wide hyperparameters.
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: str,
        random_state: Optional[int],
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_setting = max_features
        self.random_state = random_state
        self.trees: list[DecisionTree] = []
        self.classes_: Optional[torch.Tensor] = None

    # Fit each tree on a bootstrap sample.
    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        unique_classes, encoded = torch.unique(labels, sorted=True, return_inverse=True)
        self.classes_ = unique_classes
        n_classes = unique_classes.numel()
        feature_count = features.size(1)
        max_features = self._resolve_max_features(feature_count)

        self.trees = []
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)

        for tree_idx in range(self.n_estimators):
            bootstrap = torch.randint(0, features.size(0), (features.size(0),), generator=generator)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                n_classes=n_classes,
            )
            tree.fit(features[bootstrap], encoded[bootstrap])
            self.trees.append(tree)

    # Aggregate tree predictions by majority vote.
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        votes = torch.stack([tree.predict(features) for tree in self.trees])
        majority, _ = torch.mode(votes, dim=0)
        return self.classes_[majority]

    # Determine the number of features to sample per split.
    def _resolve_max_features(self, feature_count: int) -> int:
        if isinstance(self.max_features_setting, int):
            return max(1, min(self.max_features_setting, feature_count))
        if isinstance(self.max_features_setting, float):
            return max(1, min(int(self.max_features_setting * feature_count), feature_count))
        if self.max_features_setting == "sqrt":
            return max(1, int(torch.sqrt(torch.tensor(float(feature_count))).item()))
        if self.max_features_setting == "log2":
            return max(1, int(torch.log2(torch.tensor(float(feature_count))).item()))
        return feature_count


# Load data, train the forest, and save predictions.
def train_classifier(dataset_num: int) -> torch.Tensor:
    device = get_device()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "dataset")

    train_data_path = os.path.join(base_path, f"TrainData{dataset_num}.txt")
    train_label_path = os.path.join(base_path, f"TrainLabel{dataset_num}.txt")
    test_data_path = os.path.join(base_path, f"TestData{dataset_num}.txt")

    print(f"\nProcessing Dataset {dataset_num}...")

    X_train_raw, y_train = load_data(train_data_path, train_label_path)
    X_test_raw, _ = load_data(test_data_path)

    X_train_raw = X_train_raw.to(device)
    X_test_raw = X_test_raw.to(device)
    y_train = y_train.to(device)

    print(f"  Training samples: {X_train_raw.shape[0]}, Features: {X_train_raw.shape[1]}")
    print(f"  Test samples: {X_test_raw.shape[0]}")
    print(f"  Number of classes: {int(torch.unique(y_train).numel())}")
    print(f"  Using device: {device}")

    X_train_clean, imputer_stats = handle_missing_values(X_train_raw)
    X_test_clean, _ = handle_missing_values(X_test_raw, imputer_stats)

    X_train_scaled, X_test_scaled, _ = standardize(X_train_clean, X_test_clean)

    # Tree building uses CPU tensors for easier indexing.
    X_train_cpu = X_train_scaled.cpu()
    X_test_cpu = X_test_scaled.cpu()
    y_train_cpu = y_train.cpu()

    n_estimators = min(100, max(25, X_train_cpu.shape[0] // 2))
    forest = RandomForestTorch(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
    )

    forest.fit(X_train_cpu, y_train_cpu)
    predictions = forest.predict(X_test_cpu)

    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"test_result{dataset_num}.txt")
    np.savetxt(output_path, predictions.numpy(), fmt="%d")
    print(f"  Predictions saved to {output_path}")

    return predictions


if __name__ == "__main__":
    for dataset_index in range(1, 5):
        try:
            train_classifier(dataset_index)
        except Exception as err:
            print(f"Error processing dataset {dataset_index}: {err}")
            import traceback

            traceback.print_exc()

