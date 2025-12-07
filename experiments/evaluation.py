import random
from collections.abc import Sequence
from typing import Hashable, Sequence as SeqType

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

def ari_score(true_labels, pred_labels) -> float:
    """Compute Adjusted Rand Index (ARI) between true and predicted labels."""
    return float(adjusted_rand_score(true_labels, pred_labels))

def ami_score(true_labels, pred_labels) -> float:
    """Compute Adjusted Mutual Information (AMI) between true and predicted labels."""
    return float(adjusted_mutual_info_score(true_labels, pred_labels))

def evaluate_clustering(true_labels, pred_labels) -> dict[str, float]:
    if true_labels is None:
        raise ValueError("True labels are required for clustering evaluation.")
    if pred_labels is None:
        raise ValueError("Predicted labels are required for clustering evaluation.")
    return {
        "ari": ari_score(true_labels, pred_labels),
        "ami": ami_score(true_labels, pred_labels)
    }
    
    
    
def add_uniform_label_noise(labels: SeqType[Hashable] | np.ndarray, noise_level: float, random_state: int | None = None
                            ) -> np.ndarray:
    labels_arr = np.array(labels, dtype=object).copy()
    N = len(labels_arr)
    if N == 0:
        raise ValueError("Labels array is empty.")
    if not (0.0 <= noise_level <= 1.0):
        raise ValueError("noise_level must be between 0.0 and 1.0.")
    
    unique_labels = np.unique(labels_arr)
    if len(unique_labels) < 2 or noise_level == 0.0:
        return labels_arr
    
    rng = np.random.default_rng(random_state)
    py_rng = random.Random(random_state)
    
    indices = rng.permutation(N)
    split_point = int(N * noise_level)
    flip_indices = indices[:split_point]
    
    for idx in flip_indices:
        new_label = py_rng.choice(list(unique_labels))
        labels_arr[idx] = new_label
    
    return labels_arr


def evaluate_label_noise_curve(true_labels: SeqType[Hashable], noise_levels: SeqType[float], random_state: int | None = None
                            ) -> dict[str, list[float]]:
    true_labels_arr = np.array(true_labels)
    if true_labels_arr.size == 0:
        raise ValueError("true_labels is empty.")
    if noise_levels is None:
        noise_levels = np.arange(0.0, 0.9, 0.1)
    
    results: dict[str, list[float]] = {
        "p": [],
        "ari": [],
        "ami": []
    }
    
    for p in noise_levels:
        noisy = add_uniform_label_noise(true_labels_arr, noise_level=float(p), random_state=random_state)  
        scores = evaluate_clustering(true_labels_arr, noisy)
        
        results["p"].append(float(p))
        results["ari"].append(scores["ari"])
        results["ami"].append(scores["ami"])

    return results