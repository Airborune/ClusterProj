from __future__ import annotations

from collections import Counter

import numpy as np

from .preprocessing import WindowSegment


def window_labels_to_sample_labels(
    signal_length: int,
    segments: list[WindowSegment],
    labels: list[int],
) -> np.ndarray:
    """Преобразовать оконные метки в пометку каждого сэмпла без изменения идентичности меток.

    Сэмпл получает метку сегмента, который его покрывает; непокрытые хвостовые
    сэмплы наследуют только последнюю известную метку, чтобы можно было отрисовать
    весь сигнал.
    """
    votes: list[Counter[int]] = [Counter() for _ in range(signal_length)]
    for segment, label in zip(segments, labels):
        start = max(0, segment.start)
        end = min(signal_length, segment.end)
        for index in range(start, end):
            votes[index][int(label)] += 1

    sample_labels = np.full(signal_length, -1, dtype=int)
    previous_label = -1
    for index, counter in enumerate(votes):
        if not counter:
            sample_labels[index] = previous_label
            continue
        non_noise = {label: count for label, count in counter.items() if label != -1}
        selected_counter = non_noise or dict(counter)
        sample_labels[index] = max(
            selected_counter,
            key=lambda label: (selected_counter[label], label == previous_label),
        )
        previous_label = int(sample_labels[index])

    return sample_labels


def sample_label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    return _runs(labels)


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []

    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for index in range(1, len(labels)):
        label = int(labels[index])
        if label != current:
            runs.append((start, index, current))
            start = index
            current = label
    runs.append((start, len(labels), current))
    return runs
