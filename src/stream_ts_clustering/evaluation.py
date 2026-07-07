from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterQuality:
    windows: int
    cluster_count: int
    noise_count: int
    noise_ratio: float
    largest_cluster_ratio: float
    label_switch_ratio: float
    visual_run_count: int
    mean_visual_run_length: float
    short_visual_run_ratio: float
    transitions_per_100_samples: float
    entropy: float
    purity: float
    score: float
    label_distribution: dict[int, int]
    contingency_table: dict[int, dict[int, int]]


def cluster_quality_metrics(
    labels: list[int],
    source_classes: list[int] | None = None,
    sample_labels: np.ndarray | None = None,
) -> ClusterQuality:
    labels_array = np.asarray(labels, dtype=int)
    windows = int(labels_array.size)
    if windows == 0:
        return ClusterQuality(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

    counts = Counter(int(label) for label in labels_array)
    cluster_labels = sorted(label for label in counts if label != -1)
    non_noise_count = windows - counts.get(-1, 0)
    largest_cluster = max((counts[label] for label in cluster_labels), default=0)
    label_switches = int(np.sum(labels_array[1:] != labels_array[:-1])) if windows > 1 else 0

    entropy = _normalized_entropy([counts[label] for label in cluster_labels])
    purity = _cluster_purity(labels, source_classes or [])
    noise_ratio = counts.get(-1, 0) / windows
    largest_cluster_ratio = largest_cluster / max(1, non_noise_count)
    label_switch_ratio = label_switches / max(1, windows - 1)
    visual_stats = _visual_run_stats(
        np.asarray(sample_labels, dtype=int) if sample_labels is not None else labels_array
    )
    score = _quality_score(
        cluster_count=len(cluster_labels),
        noise_ratio=noise_ratio,
        largest_cluster_ratio=largest_cluster_ratio,
        label_switch_ratio=label_switch_ratio,
        short_visual_run_ratio=visual_stats[2],
        entropy=entropy,
        purity=purity,
    )

    return ClusterQuality(
        windows=windows,
        cluster_count=len(cluster_labels),
        noise_count=counts.get(-1, 0),
        noise_ratio=noise_ratio,
        largest_cluster_ratio=largest_cluster_ratio,
        label_switch_ratio=label_switch_ratio,
        visual_run_count=visual_stats[0],
        mean_visual_run_length=visual_stats[1],
        short_visual_run_ratio=visual_stats[2],
        transitions_per_100_samples=visual_stats[3],
        entropy=entropy,
        purity=purity,
        score=score,
        label_distribution=dict(sorted(counts.items())),
        contingency_table=_contingency_table(labels, source_classes or []),
    )


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probabilities = np.asarray(counts, dtype=float) / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / float(np.log(len(counts)))


def _cluster_purity(labels: list[int], source_classes: list[int]) -> float:
    if not source_classes:
        return 0.0

    usable = [
        (label, source_class)
        for label, source_class in zip(labels, source_classes)
        if label != -1
    ]
    if not usable:
        return 0.0

    grouped: dict[int, Counter[int]] = {}
    for label, source_class in usable:
        grouped.setdefault(int(label), Counter())[int(source_class)] += 1
    correct = sum(max(class_counts.values()) for class_counts in grouped.values())
    return correct / len(usable)


def _contingency_table(labels: list[int], source_classes: list[int]) -> dict[int, dict[int, int]]:
    if not source_classes:
        return {}

    table: dict[int, Counter[int]] = {}
    for label, source_class in zip(labels, source_classes):
        table.setdefault(int(source_class), Counter())[int(label)] += 1
    return {
        source_class: dict(sorted(label_counts.items()))
        for source_class, label_counts in sorted(table.items())
    }


def _visual_run_stats(labels: np.ndarray) -> tuple[int, float, float, float]:
    if len(labels) == 0:
        return 0, 0.0, 0.0, 0.0

    run_lengths: list[int] = []
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[index - 1]:
            run_lengths.append(index - start)
            start = index
    run_lengths.append(len(labels) - start)

    run_count = len(run_lengths)
    mean_run_length = float(np.mean(run_lengths))
    short_threshold = max(3, len(labels) // 120)
    short_ratio = float(sum(length < short_threshold for length in run_lengths) / run_count)
    transitions_per_100 = (run_count - 1) / max(1, len(labels)) * 100.0
    return run_count, mean_run_length, short_ratio, transitions_per_100


def _quality_score(
    cluster_count: int,
    noise_ratio: float,
    largest_cluster_ratio: float,
    label_switch_ratio: float,
    short_visual_run_ratio: float,
    entropy: float,
    purity: float,
) -> float:
    # Compact diagnostic score. It no longer assumes that five clusters or a
    # 30% label-switch rate are desirable for every stream. Stable regimes with
    # low noise, low dominance and few short visual runs receive higher scores.
    if cluster_count <= 0:
        cluster_fit = 0.0
    elif cluster_count <= 8:
        cluster_fit = 1.0
    else:
        cluster_fit = max(0.0, 1.0 - (cluster_count - 8) / 8)

    noise_fit = max(0.0, 1.0 - noise_ratio / 0.30)
    dominance_fit = max(0.0, 1.0 - max(0.0, largest_cluster_ratio - 0.70) / 0.30)
    switch_fit = max(0.0, 1.0 - label_switch_ratio / 0.50)

    score = (
        0.18 * cluster_fit
        + 0.24 * noise_fit
        + 0.18 * dominance_fit
        + 0.18 * switch_fit
        - 0.10 * short_visual_run_ratio
        + 0.07 * entropy
        + 0.15 * purity
    )
    return float(np.clip(score, 0.0, 1.0))

