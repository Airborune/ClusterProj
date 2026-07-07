from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ClusterProbe, StreamClusterer


@dataclass
class MicroCluster:
    cluster_id: int
    n: int
    linear_sum: np.ndarray
    squared_sum: np.ndarray
    last_update: int

    @property
    def center(self) -> np.ndarray:
        return self.linear_sum / self.n

    @property
    def radius(self) -> float:
        if self.n <= 1:
            return 0.0
        variance = self.squared_sum / self.n - self.center**2
        return float(np.sqrt(np.maximum(variance, 0.0).sum()))

    def add(self, x: np.ndarray, timestamp: int) -> None:
        self.n += 1
        self.linear_sum += x
        self.squared_sum += x**2
        self.last_update = timestamp


@dataclass(frozen=True)
class MacroCluster:
    cluster_id: int
    center: np.ndarray
    weight: float


class CluStream(StreamClusterer):
    name = "CluStream"

    def __init__(
        self,
        max_micro_clusters: int = 32,
        radius_factor: float = 1.35,
        min_radius: float = 1.10,
        stale_after: int = 120,
        macro_rebuild_interval: int = 2,
        tentative_distance: float = 2.50,
        emit_micro_labels: bool = False,
    ) -> None:
        self.max_micro_clusters = max_micro_clusters
        self.radius_factor = radius_factor
        self.min_radius = min_radius
        self.stale_after = stale_after
        self.macro_rebuild_interval = max(1, macro_rebuild_interval)
        self.tentative_distance = float(tentative_distance)
        self.emit_micro_labels = bool(emit_micro_labels)
        self.microclusters: list[MicroCluster] = []
        self._next_id = 0
        self._next_macro_id = 0
        self._macro_cache: list[MacroCluster] = []
        self._macro_cache_signature: tuple[tuple[int, int, int], ...] | None = None

    def reset(self) -> None:
        self.microclusters = []
        self._next_id = 0
        self._next_macro_id = 0
        self._macro_cache = []
        self._macro_cache_signature = None


    def probe(self, x: np.ndarray, timestamp: int = 0) -> ClusterProbe:
        """Вернуть совместимость CluStream для открытого кандидатного окна.

        Метод намеренно не имеет побочных эффектов. Он позволяет контроллеру
        реального времени решить, всё ли ещё подходит текущее расширяющееся окно
        модели, прежде чем сегмент будет зафиксирован.
        """
        x = np.asarray(x, dtype=float)
        if not self.microclusters:
            return ClusterProbe(
                label=0,
                compatible=True,
                score=0.0,
                distance=0.0,
                radius=max(self.min_radius, 1e-6),
                is_novel=False,
                is_noise=False,
            )
        nearest = self._nearest_cluster(x)
        distance = float(np.linalg.norm(x - nearest.center))
        threshold = max(self._absorption_threshold(nearest), 1e-6)
        score = distance / threshold
        if x.shape[0] >= 11:
            scale_idx = np.array([1, 3, 4, 6, 10], dtype=int)
            scale_distance = float(np.linalg.norm(x[scale_idx] - nearest.center[scale_idx]))
            score = max(score, scale_distance / max(threshold * 0.62, 1e-6))
            level_idx = np.array([0, 2], dtype=int)
            level_distance = float(np.linalg.norm(x[level_idx] - nearest.center[level_idx]))
            score = max(score, level_distance / max(threshold * 0.48, 1e-6))
        macro_label = int(nearest.cluster_id) if self.emit_micro_labels else self.predict_visible_label(x)
        soft_limit = 1.12 if len(self.microclusters) <= 2 else 1.05
        compatible = score <= soft_limit
        return ClusterProbe(
            label=int(macro_label),
            compatible=bool(compatible),
            score=float(score),
            distance=distance,
            radius=threshold,
            is_novel=bool(score > 1.35),
            is_noise=False,
        )

    def predict_label_without_update(self, x: np.ndarray) -> int:
        return self.predict_visible_label(np.asarray(x, dtype=float))

    def commit_from_probe(self, x: np.ndarray, timestamp: int, probe: ClusterProbe) -> int:
        """Обновить структуру CluStream, выбранную ``probe``.

        Зафиксированное окно не классифицируется повторно после того, как
        контроллер его закрыл. Если проба без побочных эффектов отнесла окно к
        существующему видимому кластеру, обновляется ближайший совместимый
        микрокластер и возвращается та же метка. Если probe пометил окно как
        новое/несовместимое, CluStream создаёт новый микрокластер как
        алгоритмически управляемое решение о переходе.
        """
        x = np.asarray(x, dtype=float)
        if not self.microclusters:
            created = self._create_cluster(x, timestamp)
            self._invalidate_macro_cache()
            if self.emit_micro_labels:
                return int(created.cluster_id)
            return self.predict_visible_label(x)

        if probe.compatible and probe.label != -1:
            nearest = self._nearest_cluster(x)
            nearest.add(x, timestamp)
            self._invalidate_macro_cache()
            return int(probe.label)

        nearest = self._nearest_cluster(x)
        distance = float(np.linalg.norm(x - nearest.center))
        threshold = self._absorption_threshold(nearest)
        if distance <= threshold:
            nearest.add(x, timestamp)
            self._invalidate_macro_cache()
            return int(nearest.cluster_id) if self.emit_micro_labels else self.predict_visible_label(x)

        level_distance = 0.0
        scale_distance = distance
        if x.shape[0] >= 11:
            level_idx = np.array([0, 2], dtype=int)
            scale_idx = np.array([1, 3, 4, 6, 10], dtype=int)
            level_distance = float(np.linalg.norm(x[level_idx] - nearest.center[level_idx]))
            scale_distance = float(np.linalg.norm(x[scale_idx] - nearest.center[scale_idx]))

        created = self._create_cluster(x, timestamp)
        if len(self.microclusters) > self.max_micro_clusters:
            self._reduce_microclusters(timestamp)
        self._invalidate_macro_cache()

        # Сильный скачок по уровню/масштабу сразу выдаётся как новый онлайн-кластер.
        # Более слабая новизна всё равно сохраняется как микрокластер, но видимая
        # метка может прийти с макроуровня, чтобы повторяющиеся высокодисперсные
        # куски одного режима не становились новой меткой каждый раз.
        if self.emit_micro_labels or level_distance >= 1.35 or scale_distance >= 3.10:
            return int(created.cluster_id)
        return self.predict_visible_label(x)

    def fit_predict_one(self, x: np.ndarray, timestamp: int) -> int:
        x = np.asarray(x, dtype=float)
        if not self.microclusters:
            created = self._create_cluster(x, timestamp)
            self._invalidate_macro_cache()
            return int(created.cluster_id) if self.emit_micro_labels else self.predict_visible_label(x)

        nearest = self._nearest_cluster(x)
        distance = float(np.linalg.norm(x - nearest.center))
        threshold = self._absorption_threshold(nearest)

        if distance <= threshold:
            nearest.add(x, timestamp)
            self._invalidate_macro_cache()
            return int(nearest.cluster_id) if self.emit_micro_labels else self.predict_visible_label(x)

        previous_macroclusters = self.macrocluster_centers()
        delayed_label = self._tentative_existing_label(x, previous_macroclusters, distance)
        created = self._create_cluster(x, timestamp)
        if len(self.microclusters) > self.max_micro_clusters:
            self._reduce_microclusters(timestamp)
        self._invalidate_macro_cache()
        if self.emit_micro_labels:
            return int(created.cluster_id)
        if delayed_label is not None:
            return delayed_label
        return self.predict_visible_label(x)


    def _tentative_existing_label(
        self,
        x: np.ndarray,
        previous_macroclusters: list[MacroCluster],
        distance_to_nearest_micro: float,
    ) -> int | None:
        """Отложить слабую новизну, не меняя прошлые метки.

        Один расширяющийся сегмент может быть переходным фрагментом. Если новый
        микрокластер лишь умеренно далеко от существующей модели, текущая точка
        выдаётся как ближайший существующий макрокластер, а микрокластер всё ещё
        хранится внутри. Если позже новые точки подтвердят эту область, он станет
        обычным видимым кластером онлайн.
        """
        if not previous_macroclusters:
            return None
        nearest_macro = min(
            previous_macroclusters,
            key=lambda cluster: float(np.linalg.norm(x - cluster.center)),
        )
        nearest_distance = float(np.linalg.norm(x - nearest_macro.center))
        if nearest_distance >= self.tentative_distance:
            return None
        if len(previous_macroclusters) >= 2:
            macro_centers = np.array([cluster.center for cluster in previous_macroclusters], dtype=float)
            pairwise = np.linalg.norm(macro_centers[:, None, :] - macro_centers[None, :, :], axis=2)
            nonzero = pairwise[pairwise > 1e-12]
            typical_macro_gap = float(np.median(nonzero)) if nonzero.size else 0.0
            if typical_macro_gap > 0.0 and nearest_distance >= typical_macro_gap * 0.52:
                return None
        if distance_to_nearest_micro >= self.tentative_distance * 0.95 and len(previous_macroclusters) >= 2:
            return None
        return int(nearest_macro.cluster_id)

    def predict_visible_label(self, x: np.ndarray) -> int:
        macroclusters = self.macrocluster_centers()
        if not macroclusters:
            return -1
        nearest = min(
            macroclusters,
            key=lambda cluster: float(np.linalg.norm(x - cluster.center)),
        )
        return int(nearest.cluster_id)

    def macrocluster_centers(self) -> list[MacroCluster]:
        """Свернуть микрокластеры в макрокластеры без фиксированного верхнего предела.

        Число макрокластеров выбирается по текущей геометрии микрокластеров.
        Метки сопоставляются с предыдущей макрораскладкой, чтобы избежать
        визуального переключения меток.
        """
        if not self.microclusters:
            self._macro_cache = []
            self._macro_cache_signature = None
            return []

        signature = tuple(
            (cluster.cluster_id, cluster.n, cluster.last_update)
            for cluster in self.microclusters
        )
        if self._macro_cache_signature == signature:
            return list(self._macro_cache)

        centers = np.array([cluster.center for cluster in self.microclusters], dtype=float)
        weights = np.array([cluster.n for cluster in self.microclusters], dtype=float)
        max_k = len(self.microclusters)

        if max_k == 1:
            raw_macroclusters = [
                MacroCluster(
                    cluster_id=-1,
                    center=np.average(centers, axis=0, weights=weights),
                    weight=float(weights.sum()),
                )
            ]
        else:
            best_result: tuple[float, int, np.ndarray, np.ndarray] | None = None
            for k in range(1, max_k + 1):
                centroids, assignments, inertia = _weighted_kmeans(centers, weights, k)
                score = _macro_partition_score(centers, weights, assignments, inertia, k)
                if best_result is None or score > best_result[0]:
                    best_result = (score, k, centroids, assignments)

            assert best_result is not None
            _, k, centroids, assignments = best_result
            raw_macroclusters = []
            for cluster_index in range(k):
                mask = assignments == cluster_index
                if not np.any(mask):
                    continue
                raw_macroclusters.append(
                    MacroCluster(
                        cluster_id=-1,
                        center=centroids[cluster_index].copy(),
                        weight=float(weights[mask].sum()),
                    )
                )

        stable_macroclusters = self._assign_stable_macro_ids(raw_macroclusters)
        self._macro_cache = stable_macroclusters
        self._macro_cache_signature = signature
        return list(stable_macroclusters)

    def _assign_stable_macro_ids(self, raw_macroclusters: list[MacroCluster]) -> list[MacroCluster]:
        if not raw_macroclusters:
            return []
        if not self._macro_cache:
            assigned: list[MacroCluster] = []
            for raw in sorted(raw_macroclusters, key=lambda cluster: -cluster.weight):
                assigned.append(
                    MacroCluster(
                        cluster_id=self._next_macro_id,
                        center=raw.center,
                        weight=raw.weight,
                    )
                )
                self._next_macro_id += 1
            return sorted(assigned, key=lambda cluster: cluster.cluster_id)

        # Сопоставляем новые макроцентры со старыми по наименьшему геометрическому
        # смещению. Это предотвращает "кражу" идентификаторов в разреженных
        # потоках: когда появляется третий макрокластер, старый стабильный
        # макроцентр должен сохранить свой идентификатор, если сырой центр всё
        # ещё находится на нём, а действительно новый центр получает новый
        # идентификатор. Это часть онлайн-стабильности меток, а не постфактум
        # исправление уже выданных меток.
        reuse_limit = max(self.min_radius * 1.35, 1.5)
        raw_count = len(raw_macroclusters)
        old_clusters = list(self._macro_cache)
        pair_candidates: list[tuple[float, int, int]] = []
        for raw_index, raw in enumerate(raw_macroclusters):
            for old_index, old in enumerate(old_clusters):
                distance = float(np.linalg.norm(raw.center - old.center))
                if distance <= reuse_limit:
                    pair_candidates.append((distance, raw_index, old_index))
        pair_candidates.sort(key=lambda item: item[0])

        raw_to_old: dict[int, MacroCluster] = {}
        used_raw: set[int] = set()
        used_old: set[int] = set()
        for _distance, raw_index, old_index in pair_candidates:
            if raw_index in used_raw or old_index in used_old:
                continue
            raw_to_old[raw_index] = old_clusters[old_index]
            used_raw.add(raw_index)
            used_old.add(old_index)

        assigned: list[MacroCluster] = []
        for raw_index, raw in enumerate(raw_macroclusters):
            old = raw_to_old.get(raw_index)
            if old is None:
                cluster_id = self._next_macro_id
                self._next_macro_id += 1
            else:
                cluster_id = old.cluster_id
            assigned.append(MacroCluster(cluster_id=cluster_id, center=raw.center, weight=raw.weight))
        return sorted(assigned, key=lambda cluster: cluster.cluster_id)

    def _invalidate_macro_cache(self) -> None:
        self._macro_cache_signature = None

    def _create_cluster(self, x: np.ndarray, timestamp: int) -> MicroCluster:
        cluster = MicroCluster(
            cluster_id=self._next_id,
            n=1,
            linear_sum=x.copy(),
            squared_sum=x**2,
            last_update=timestamp,
        )
        self._next_id += 1
        self.microclusters.append(cluster)
        return cluster

    def _nearest_cluster(self, x: np.ndarray) -> MicroCluster:
        return min(self.microclusters, key=lambda mc: float(np.linalg.norm(x - mc.center)))

    def _absorption_threshold(self, cluster: MicroCluster) -> float:
        if cluster.n > 1:
            return min(
                max(self.min_radius, self.radius_factor * cluster.radius),
                self.min_radius * 1.45,
            )
        return max(self.min_radius * 0.95, 0.45)

    def _reduce_microclusters(self, timestamp: int) -> None:
        stale = [
            (index, cluster)
            for index, cluster in enumerate(self.microclusters)
            if timestamp - cluster.last_update >= self.stale_after
        ]
        if stale:
            stale.sort(key=lambda item: (item[1].n, item[1].last_update))
            self.microclusters.pop(stale[0][0])
            return
        if len(self.microclusters) < 2:
            return

        best_pair = (0, 1)
        best_distance = float("inf")
        for i in range(len(self.microclusters)):
            for j in range(i + 1, len(self.microclusters)):
                distance = float(np.linalg.norm(self.microclusters[i].center - self.microclusters[j].center))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        merge_limit = max(self.min_radius * 2.5, 0.75)
        if best_distance <= merge_limit:
            self._merge_pair(best_pair)
            return

        # Защищаем совсем свежие слабые кластеры: они могут соответствовать только что появившемуся режиму.
        fresh_horizon = max(2, self.stale_after // 4)
        candidates = [
            idx for idx, cluster in enumerate(self.microclusters)
            if timestamp - cluster.last_update >= fresh_horizon
        ]
        if not candidates:
            candidates = list(range(len(self.microclusters)))
        removable_index = min(
            candidates,
            key=lambda idx: (
                self.microclusters[idx].n,
                self.microclusters[idx].last_update,
            ),
        )
        self.microclusters.pop(removable_index)

    def _merge_pair(self, pair: tuple[int, int]) -> None:
        i, j = pair
        first = self.microclusters[i]
        second = self.microclusters[j]
        merged = MicroCluster(
            cluster_id=min(first.cluster_id, second.cluster_id),
            n=first.n + second.n,
            linear_sum=first.linear_sum + second.linear_sum,
            squared_sum=first.squared_sum + second.squared_sum,
            last_update=max(first.last_update, second.last_update),
        )
        for index in sorted(pair, reverse=True):
            self.microclusters.pop(index)
        self.microclusters.append(merged)


def _macro_partition_score(
    points: np.ndarray,
    weights: np.ndarray,
    assignments: np.ndarray,
    inertia: float,
    k: int,
) -> float:
    if k <= 1:
        spread = float(np.sum(weights * np.linalg.norm(points - np.average(points, axis=0, weights=weights), axis=1)))
        return -0.05 * np.log1p(spread)

    if k == len(points) and len(points) <= 3:
        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        nonzero = pairwise[pairwise > 1e-12]
        if nonzero.size == 0:
            return -0.25 * (k - 1)
        nearest = float(np.min(nonzero))
        median_distance = float(np.median(nonzero))
        if nearest < 0.42:
            return -0.30 * (k - 1)
        separation_balance = nearest / max(median_distance, 1e-6)
        # У потоков с расширяющимся окном может быть всего два или три закрытых
        # сегмента. В таком разреженном режиме расстояние около 0.5-0.7 в
        # нормализованном пространстве признаков уже имеет смысл; иначе реальные
        # двухрежимные случаи скрываются как один макрокластер. Это выбор макроуровня,
        # а не постобработка меток.
        absolute_separation = np.tanh(nearest / 0.70)
        entropy = _cluster_weight_entropy(weights, assignments)
        return float(0.72 * absolute_separation + 0.18 * separation_balance + 0.10 * entropy - 0.04 * (k - 1))

    silhouette = _weighted_silhouette_score(points, weights, assignments)
    entropy = _cluster_weight_entropy(weights, assignments)
    singleton_penalty = _separation_aware_singleton_penalty(points, assignments)
    inertia_penalty = 0.01 * np.log1p(max(inertia, 0.0))
    return float(
        silhouette
        + 0.04 * entropy
        - 0.30 * (k - 1)
        - 1.20 * singleton_penalty
        - 1.0 * inertia_penalty
    )


def _separation_aware_singleton_penalty(points: np.ndarray, assignments: np.ndarray) -> float:
    cluster_ids = sorted(set(int(label) for label in assignments))
    if len(cluster_ids) <= 1 or len(points) <= 2:
        return 0.0
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    nonzero = pairwise[pairwise > 1e-12]
    typical = float(np.median(nonzero)) if nonzero.size else 0.0
    penalty = 0.0
    for cluster_id in cluster_ids:
        mask = assignments == cluster_id
        if np.count_nonzero(mask) != 1:
            continue
        index = int(np.where(mask)[0][0])
        other_mask = ~mask
        if not np.any(other_mask):
            continue
        nearest = float(np.min(pairwise[index, other_mask]))
        if typical > 0.0 and nearest >= typical * 0.85:
            continue
        penalty += 0.65
    return penalty


def _weighted_farthest_initialization(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> np.ndarray:
    first_index = int(np.argmax(weights))
    centroids = [points[first_index].copy()]
    while len(centroids) < k:
        distances = np.min(
            np.linalg.norm(points[:, None, :] - np.array(centroids)[None, :, :], axis=2),
            axis=1,
        )
        score = distances * np.sqrt(weights)
        next_index = int(np.argmax(score))
        centroids.append(points[next_index].copy())
    return np.array(centroids, dtype=float)


def _weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if k <= 1:
        centroid = np.average(points, axis=0, weights=weights)
        distances = np.linalg.norm(points - centroid, axis=1)
        inertia = float(np.sum(weights * distances**2))
        return np.array([centroid], dtype=float), np.zeros(len(points), dtype=int), inertia

    centroids = _weighted_farthest_initialization(points, weights, k)
    assignments = np.zeros(len(points), dtype=int)
    for _ in range(12):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_index in range(k):
            mask = assignments == cluster_index
            if np.any(mask):
                centroids[cluster_index] = np.average(points[mask], axis=0, weights=weights[mask])

    distances = np.linalg.norm(points - centroids[assignments], axis=1)
    inertia = float(np.sum(weights * distances**2))
    return centroids, assignments, inertia


def _weighted_silhouette_score(
    points: np.ndarray,
    weights: np.ndarray,
    assignments: np.ndarray,
) -> float:
    cluster_ids = sorted(set(int(label) for label in assignments))
    if len(cluster_ids) <= 1:
        return 0.0

    total_weight = float(weights.sum())
    if total_weight <= 1e-12:
        return 0.0

    pairwise_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    silhouette_sum = 0.0
    for index, cluster_id in enumerate(assignments):
        same_mask = assignments == cluster_id
        same_mask[index] = False
        if np.count_nonzero(same_mask) == 0:
            continue
        same_weights = weights[same_mask]
        same_distances = pairwise_distances[index, same_mask]
        a = float(np.average(same_distances, weights=same_weights))

        b = float("inf")
        for other_cluster in cluster_ids:
            if other_cluster == cluster_id:
                continue
            other_mask = assignments == other_cluster
            if np.count_nonzero(other_mask) == 0:
                continue
            candidate = float(np.average(pairwise_distances[index, other_mask], weights=weights[other_mask]))
            if candidate < b:
                b = candidate

        if not np.isfinite(b):
            continue
        denom = max(a, b)
        silhouette = 0.0 if denom <= 1e-12 else (b - a) / denom
        silhouette_sum += silhouette * float(weights[index])

    return float(silhouette_sum / total_weight)


def _cluster_weight_entropy(weights: np.ndarray, assignments: np.ndarray) -> float:
    cluster_weights = np.array(
        [float(weights[assignments == cluster_id].sum()) for cluster_id in sorted(set(int(label) for label in assignments))],
        dtype=float,
    )
    total_weight = float(cluster_weights.sum())
    if total_weight <= 1e-12 or len(cluster_weights) <= 1:
        return 0.0
    probabilities = cluster_weights / total_weight
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(entropy / np.log(len(cluster_weights)))
