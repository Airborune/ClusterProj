from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterProbe:
    """Сигнал совместимости без побочных эффектов для открытого расширяющегося окна."""

    label: int
    compatible: bool
    score: float
    distance: float
    radius: float
    is_novel: bool = False
    is_noise: bool = False


class StreamClusterer(ABC):
    name: str

    @abstractmethod
    def fit_predict_one(self, x: np.ndarray, timestamp: int) -> int:
        """Обновить модель по одному зафиксированному вектору признаков и вернуть метку."""

    def probe(self, x: np.ndarray, timestamp: int = 0) -> ClusterProbe:
        """Оценить принадлежность кластера для открытого кандидатного окна без обновления состояния."""
        label = self.predict_label_without_update(x)
        return ClusterProbe(
            label=label,
            compatible=label != -1,
            score=0.0 if label != -1 else float("inf"),
            distance=0.0,
            radius=1.0,
            is_novel=label == -1,
            is_noise=label == -1,
        )


    def commit_from_probe(self, x: np.ndarray, timestamp: int, probe: ClusterProbe) -> int:
        """Зафиксировать ровно тот кандидат, который породил ``probe``.

        Это связывает решение о границе и назначение кластера: конкретные
        алгоритмы могут обновить кластер, выбранный ``probe``, вместо повторной
        кластеризации того же окна с нуля после его закрытия.
        """
        return self.fit_predict_one(x, timestamp)

    def predict_label_without_update(self, x: np.ndarray) -> int:
        return -1

    @abstractmethod
    def reset(self) -> None:
        """Сбросить накопленное состояние модели."""
