from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ClusterProbe, StreamClusterer


@dataclass
class DensityMicroCluster:
    cluster_id: int
    weight: float
    linear_sum: np.ndarray
    squared_sum: np.ndarray
    creation_time: int
    last_update: int

    @property
    def center(self) -> np.ndarray:
        return self.linear_sum / max(self.weight, 1e-12)

    @property
    def radius(self) -> float:
        variance = self.squared_sum / max(self.weight, 1e-12) - self.center**2
        return float(np.sqrt(np.maximum(variance, 0.0).sum()))

    def decay_to(self, timestamp: int, lambd: float) -> None:
        if timestamp <= self.last_update:
            return
        factor = 2.0 ** (-lambd * (timestamp - self.last_update))
        self.weight *= factor
        self.linear_sum *= factor
        self.squared_sum *= factor
        self.last_update = timestamp

    def radius_after_add(self, x: np.ndarray, timestamp: int, lambd: float) -> float:
        factor = 2.0 ** (-lambd * max(0, timestamp - self.last_update))
        weight = self.weight * factor + 1.0
        linear_sum = self.linear_sum * factor + x
        squared_sum = self.squared_sum * factor + x**2
        center = linear_sum / max(weight, 1e-12)
        variance = squared_sum / max(weight, 1e-12) - center**2
        return float(np.sqrt(np.maximum(variance, 0.0).sum()))

    def add(self, x: np.ndarray, timestamp: int, lambd: float) -> None:
        self.decay_to(timestamp, lambd)
        self.weight += 1.0
        self.linear_sum += x
        self.squared_sum += x**2
        self.last_update = timestamp


class DenStream(StreamClusterer):

    name = "DenStream"

    def __init__(
        self,
        epsilon: float = 2.0,
        beta: float = 0.35,
        mu: float = 2.0,
        lambd: float = 0.002,
        prune_interval: int = 10,
        radius_tolerance: float = 1.15,
        emit_noise: bool = True,
        grace_period: int = 2,
        outlier_radius_tolerance: float = 1.20,
        outlier_promote_weight: float = 2.0,
        outlier_promote_age: int = 1,
        min_points_before_noise: int = 4,
        noise_score_threshold: float = 2.45,
    ) -> None:
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.lambd = lambd
        self.prune_interval = prune_interval
        self.radius_tolerance = radius_tolerance
        self.emit_noise = emit_noise
        self.grace_period = max(0, grace_period)
        self.outlier_radius_tolerance = max(1.0, outlier_radius_tolerance)
        self.outlier_promote_weight = max(0.0, outlier_promote_weight)
        self.outlier_promote_age = max(0, outlier_promote_age)
        self.min_points_before_noise = max(self.grace_period, int(min_points_before_noise))
        self.noise_score_threshold = max(1.0, float(noise_score_threshold))
        self.p_microclusters: list[DensityMicroCluster] = []
        self.o_microclusters: list[DensityMicroCluster] = []
        self._next_id = 0
        self._seen_points = 0

    def reset(self) -> None:
        self.p_microclusters = []
        self.o_microclusters = []
        self._next_id = 0
        self._seen_points = 0


    def _outlier_would_promote_after_add(self, cluster: DensityMicroCluster, timestamp: int) -> bool:
        factor = 2.0 ** (-self.lambd * max(0, timestamp - cluster.last_update))
        projected_weight = cluster.weight * factor + 1.0
        if projected_weight < max(self.beta * self.mu, self.outlier_promote_weight):
            return False
        if timestamp - cluster.creation_time < self.outlier_promote_age:
            return False
        return True

    def probe(self, x: np.ndarray, timestamp: int = 0) -> ClusterProbe:

        """Return DenStream density compatibility without updating clusters."""
        x = np.asarray(x, dtype=float)
        if not self.p_microclusters and not self.o_microclusters:
            # Empty DenStream has no confirmed density region yet. The first
            # committed object becomes an outlier candidate, so the side-effect
            # free probe should report novelty rather than a confirmed compatible
            # p-microcluster.
            return ClusterProbe(
                label=0,
                compatible=False,
                score=0.0,
                distance=0.0,
                radius=max(self.epsilon * self.radius_tolerance, 1e-6),
                is_novel=True,
                is_noise=False,
            )

        p_tolerance = max(self.epsilon * self.radius_tolerance, 1e-6)
        o_tolerance = max(self.epsilon * self.outlier_radius_tolerance, 1e-6)

        nearest_p = self._nearest(self.p_microclusters, x)
        if nearest_p is not None:
            p_distance = float(np.linalg.norm(x - nearest_p.center))
            p_radius_after = nearest_p.radius_after_add(x, timestamp, self.lambd)
            p_score = max(p_distance / p_tolerance, p_radius_after / p_tolerance)
            if x.shape[0] >= 11:
                scale_idx = np.array([1, 3, 4, 6, 10], dtype=int)
                p_scale_distance = float(np.linalg.norm(x[scale_idx] - nearest_p.center[scale_idx]))
                p_score = max(p_score, p_scale_distance / max(p_tolerance * 0.65, 1e-6))
                level_idx = np.array([0, 2], dtype=int)
                p_level_distance = float(np.linalg.norm(x[level_idx] - nearest_p.center[level_idx]))
                p_score = max(p_score, p_level_distance / max(p_tolerance * 0.50, 1e-6))
            if p_score <= 1.08:
                return ClusterProbe(
                    label=int(nearest_p.cluster_id),
                    compatible=True,
                    score=float(p_score),
                    distance=p_distance,
                    radius=p_tolerance,
                    is_novel=False,
                    is_noise=False,
                )
        else:
            p_distance = float("inf")
            p_score = float("inf")

        nearest_o = self._nearest(self.o_microclusters, x)
        if nearest_o is not None:
            o_distance = float(np.linalg.norm(x - nearest_o.center))
            o_radius_after = nearest_o.radius_after_add(x, timestamp, self.lambd)
            o_score = max(o_distance / o_tolerance, o_radius_after / o_tolerance)
            if x.shape[0] >= 11:
                scale_idx = np.array([1, 3, 4, 6, 10], dtype=int)
                o_scale_distance = float(np.linalg.norm(x[scale_idx] - nearest_o.center[scale_idx]))
                o_score = max(o_score, o_scale_distance / max(o_tolerance * 0.70, 1e-6))
                level_idx = np.array([0, 2], dtype=int)
                o_level_distance = float(np.linalg.norm(x[level_idx] - nearest_o.center[level_idx]))
                o_score = max(o_score, o_level_distance / max(o_tolerance * 0.55, 1e-6))
            # An outlier microcluster is allowed to keep accumulating evidence,
            # but it is not treated as a confirmed compatible region until the
            # next point would promote it to a potential microcluster.
            if o_score <= 1.05 and self._outlier_would_promote_after_add(nearest_o, timestamp):
                return ClusterProbe(
                    label=int(nearest_o.cluster_id),
                    compatible=True,
                    score=float(o_score),
                    distance=o_distance,
                    radius=o_tolerance,
                    is_novel=False,
                    is_noise=False,
                )
        else:
            o_distance = float("inf")
            o_score = float("inf")

        if nearest_p is not None and p_distance <= o_distance:
            label = int(nearest_p.cluster_id)
            distance = p_distance
            score = p_score
            radius = p_tolerance
        elif nearest_o is not None:
            label = int(nearest_o.cluster_id)
            distance = o_distance
            score = o_score
            radius = o_tolerance
        else:
            label = -1
            distance = float("inf")
            score = float("inf")
            radius = p_tolerance

        is_noise = (
            self.emit_noise
            and self._seen_points >= self.min_points_before_noise
            and np.isfinite(score)
            and float(score) >= self.noise_score_threshold
        )
        return ClusterProbe(
            label=label if not is_noise else -1,
            compatible=False,
            score=float(score),
            distance=float(distance),
            radius=float(radius),
            is_novel=True,
            is_noise=bool(is_noise),
        )

    def predict_label_without_update(self, x: np.ndarray) -> int:
        return self.predict_visible_label(np.asarray(x, dtype=float), include_outliers=True)

    def commit_from_probe(self, x: np.ndarray, timestamp: int, probe: ClusterProbe) -> int:
        """Commit the density object identified by ``probe``.

        The realtime controller closes a window using DenStream's density probe.
        This method updates that same density object instead of doing a second,
        independent label decision after the window has already been closed.
        """
        x = np.asarray(x, dtype=float)
        self._seen_points += 1
        self._decay_all(timestamp)

        if not self.p_microclusters and not self.o_microclusters:
            created = DensityMicroCluster(
                cluster_id=self._next_id,
                weight=1.0,
                linear_sum=x.copy(),
                squared_sum=x**2,
                creation_time=timestamp,
                last_update=timestamp,
            )
            self._next_id += 1
            # Keep the first object as an outlier candidate. It becomes a potential
            # microcluster only after enough supporting evidence arrives, matching
            # DenStream's density-confirmation idea while still emitting a bootstrap
            # label during the grace period.
            self.o_microclusters.append(created)
            return self._bootstrap_label(x)

        if probe.compatible and probe.label != -1:
            target = self._cluster_by_id(probe.label)
            if target is not None:
                target_was_outlier = target in self.o_microclusters
                target.add(x, timestamp, self.lambd)
                if target_was_outlier and self._should_promote_outlier(target, timestamp):
                    self.o_microclusters.remove(target)
                    self.p_microclusters.append(target)
                self._periodic_prune(timestamp)
                return int(probe.label)

        nearest_p = self._nearest_mergeable(self.p_microclusters, x, timestamp)
        if nearest_p is not None:
            nearest_p.add(x, timestamp, self.lambd)
            self._periodic_prune(timestamp)
            return int(nearest_p.cluster_id)

        nearest_o = self._nearest_mergeable(
            self.o_microclusters,
            x,
            timestamp,
            radius_tolerance=self.outlier_radius_tolerance,
        )
        if nearest_o is not None:
            nearest_o.add(x, timestamp, self.lambd)
            promoted = False
            if self._should_promote_outlier(nearest_o, timestamp):
                self.o_microclusters.remove(nearest_o)
                self.p_microclusters.append(nearest_o)
                promoted = True
            self._periodic_prune(timestamp)
            return self._outlier_output_label(nearest_o, x, promoted)

        created = DensityMicroCluster(
            cluster_id=self._next_id,
            weight=1.0,
            linear_sum=x.copy(),
            squared_sum=x**2,
            creation_time=timestamp,
            last_update=timestamp,
        )
        self.o_microclusters.append(created)
        self._next_id += 1
        self._periodic_prune(timestamp)
        return self._new_outlier_output_label(created, x, probe_is_noise=probe.is_noise, probe_score=probe.score)

    def fit_predict_one(self, x: np.ndarray, timestamp: int) -> int:
        x = np.asarray(x, dtype=float)
        self._seen_points += 1
        self._decay_all(timestamp)

        nearest_p = self._nearest_mergeable(self.p_microclusters, x, timestamp)
        if nearest_p is not None:
            nearest_p.add(x, timestamp, self.lambd)
            self._periodic_prune(timestamp)
            return int(nearest_p.cluster_id)

        nearest_o = self._nearest_mergeable(
            self.o_microclusters,
            x,
            timestamp,
            radius_tolerance=self.outlier_radius_tolerance,
        )
        if nearest_o is not None:
            nearest_o.add(x, timestamp, self.lambd)
            promoted = False
            if self._should_promote_outlier(nearest_o, timestamp):
                self.o_microclusters.remove(nearest_o)
                self.p_microclusters.append(nearest_o)
                promoted = True
            self._periodic_prune(timestamp)
            if self._seen_points <= self.grace_period:
                return self._bootstrap_label(x)
            if promoted or self._seen_points <= self.min_points_before_noise:
                return int(nearest_o.cluster_id)
            if self.emit_noise:
                return -1
            return int(nearest_o.cluster_id)

        created = DensityMicroCluster(
            cluster_id=self._next_id,
            weight=1.0,
            linear_sum=x.copy(),
            squared_sum=x**2,
            creation_time=timestamp,
            last_update=timestamp,
        )
        self.o_microclusters.append(created)
        self._next_id += 1
        self._periodic_prune(timestamp)
        return self._new_outlier_output_label(created, x)

    def _new_outlier_output_label(
        self,
        cluster: DensityMicroCluster,
        x: np.ndarray,
        probe_is_noise: bool = False,
        probe_score: float | None = None,
    ) -> int:
        if self._seen_points <= self.grace_period:
            return self._bootstrap_label(x)
        if self._seen_points <= self.min_points_before_noise:
            return int(cluster.cluster_id)
        # A brand-new outlier is not automatically hidden as noise. For
        # time-series segmentation it can be the first segment of a new regime,
        # so the temporary outlier label is emitted unless the side-effect-free
        # probe explicitly marked the object as a strong finite-distance noise
        # case. The outlier microcluster is still stored internally and can be
        # promoted later if similar segments arrive.
        if (
            self.emit_noise
            and probe_is_noise
            and probe_score is not None
            and np.isfinite(probe_score)
            and float(probe_score) >= self.noise_score_threshold
        ):
            return -1
        return int(cluster.cluster_id)

    def _outlier_output_label(
        self,
        cluster: DensityMicroCluster,
        x: np.ndarray,
        promoted: bool,
    ) -> int:
        if self._seen_points <= self.grace_period:
            return self._bootstrap_label(x)
        # If an object was mergeable with an existing outlier microcluster, it is
        # not isolated noise anymore: the outlier candidate has repeated support.
        # Emit its temporary label even before formal promotion; promotion still
        # controls when it becomes a p-microcluster. Only brand-new isolated
        # outliers after warm-up are emitted as -1. This prevents DenStream from
        # swallowing whole noisy-but-consistent regimes as noise.
        return int(cluster.cluster_id)

    def predict_visible_label(self, x: np.ndarray, include_outliers: bool = False) -> int:
        clusters = list(self.p_microclusters)
        if include_outliers:
            clusters.extend(self.o_microclusters)
        if not clusters:
            return -1
        nearest = min(
            clusters,
            key=lambda cluster: float(np.linalg.norm(x - cluster.center)),
        )
        distance = float(np.linalg.norm(x - nearest.center))
        tolerance = self.outlier_radius_tolerance if nearest in self.o_microclusters else self.radius_tolerance
        if self.emit_noise and distance > self.epsilon * tolerance:
            return -1
        return int(nearest.cluster_id)

    def _cluster_by_id(self, cluster_id: int) -> DensityMicroCluster | None:
        for cluster in self.p_microclusters + self.o_microclusters:
            if int(cluster.cluster_id) == int(cluster_id):
                return cluster
        return None

    def _nearest(
        self, clusters: list[DensityMicroCluster], x: np.ndarray
    ) -> DensityMicroCluster | None:
        if not clusters:
            return None
        return min(clusters, key=lambda mc: float(np.linalg.norm(x - mc.center)))

    def _nearest_mergeable(
        self,
        clusters: list[DensityMicroCluster],
        x: np.ndarray,
        timestamp: int,
        radius_tolerance: float | None = None,
    ) -> DensityMicroCluster | None:
        ordered = sorted(clusters, key=lambda mc: float(np.linalg.norm(x - mc.center)))
        tolerance = self.epsilon * (self.radius_tolerance if radius_tolerance is None else radius_tolerance)
        for cluster in ordered:
            center_distance = float(np.linalg.norm(x - cluster.center))
            # In sparse expanding-window streams, the classic radius-after-add test can
            # merge two distant singleton windows because the new two-point radius is
            # only half their center distance. The additional center-distance gate keeps
            # DenStream density-based while preventing regime shifts from being swallowed
            # by the first cluster.
            if center_distance > tolerance:
                continue
            if cluster.radius_after_add(x, timestamp, self.lambd) <= tolerance:
                return cluster
        return None

    def _decay_all(self, timestamp: int) -> None:
        for cluster in self.p_microclusters + self.o_microclusters:
            cluster.decay_to(timestamp, self.lambd)

    def _periodic_prune(self, timestamp: int) -> None:
        if self.prune_interval <= 0 or self._seen_points % self.prune_interval != 0:
            return
        self.o_microclusters = [
            cluster
            for cluster in self.o_microclusters
            if cluster.weight >= self._outlier_weight_threshold(timestamp, cluster.creation_time)
        ]
        kept_p = [
            cluster for cluster in self.p_microclusters if cluster.weight >= self.beta * self.mu
        ]
        # Avoid wiping the entire confirmed model at once; a single stale
        # potential microcluster is still a better anchor than immediately
        # returning to an all-outlier state.
        if kept_p or len(self.p_microclusters) <= 1:
            self.p_microclusters = kept_p or self.p_microclusters
        else:
            self.p_microclusters = kept_p

    def _outlier_weight_threshold(self, timestamp: int, creation_time: int) -> float:
        if self.lambd <= 0:
            return self.beta * self.mu
        tp = max(1, self.prune_interval)
        denominator = 2.0 ** (-self.lambd * tp) - 1.0
        if abs(denominator) < 1e-12:
            return self.beta * self.mu
        age = max(0, timestamp - creation_time + tp)
        numerator = 2.0 ** (-self.lambd * age) - 1.0
        return float(min(self.beta * self.mu, numerator / denominator))

    def _bootstrap_label(self, x: np.ndarray) -> int:
        if self.p_microclusters:
            return self.predict_visible_label(x, include_outliers=False)
        if self.o_microclusters:
            nearest = min(
                self.o_microclusters,
                key=lambda cluster: float(np.linalg.norm(x - cluster.center)),
            )
            return int(nearest.cluster_id)
        return 0

    def _should_promote_outlier(self, cluster: DensityMicroCluster, timestamp: int) -> bool:
        if cluster.weight < max(self.beta * self.mu, self.outlier_promote_weight):
            return False
        if timestamp - cluster.creation_time < self.outlier_promote_age:
            return False
        return True

