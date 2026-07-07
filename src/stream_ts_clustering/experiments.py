from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithms.base import StreamClusterer
from .algorithms.clustream import CluStream
from .algorithms.denstream import DenStream
from .data_loader import (
    StreamSignal,
    build_stream_signal,
    load_labeled_csv_stream,
    load_multivariate_csv_stream,
    load_ucr_tsv,
    load_uci_stream,
    load_univariate_csv_stream,
    list_synthetic_dataset_specs,
    list_synthetic_validation_dataset_specs,
    make_synthetic_segmentation_stream,
)
from .evaluation import cluster_quality_metrics
from .postprocessing import window_labels_to_sample_labels
from .preprocessing import (
    FeatureScaler,
    WindowSegment,
    extract_features_with_profile,
)
from .visualization import (
    RealtimeDashboardPlotter,
    RealtimeFrameState,
    RealtimePlotter,
    RealtimeScenarioState,
    plot_clustered_signal_comparison,
)


@dataclass(frozen=True)
class SyntheticValidationSummary:
    dataset: str
    method: str
    segment_count: int
    true_boundary_count: int
    boundary_mae: float
    boundary_max_error: float
    segment_purity: float
    cluster_count: int
    noise_ratio: float
    largest_cluster_ratio: float
    score: float
    passed: bool


@dataclass(frozen=True)
class BoundaryMetrics:
    method: str
    tolerance: int
    true_boundary_count: int
    predicted_boundary_count: int
    boundary_mae: float | None
    boundary_median_error: float | None
    boundary_max_error: float | None
    tp: int | None
    fp: int | None
    fn: int | None
    tn: int | None
    far: float | None
    frr: float | None
    accuracy: float | None
    recall: float | None
    precision: float | None


@dataclass(frozen=True)
class SyntheticRealtimeJob:
    name: str
    dataset_path: str
    series_index: int
    series_count: int
    window_size: int
    step: int
    balanced: bool
    window_mode: str
    feature_profile: str


@dataclass(frozen=True)
class SyntheticRealtimeScenario:
    name: str
    stream: StreamSignal
    segments: list[WindowSegment]
    labels_by_method: dict[str, list[int]]
    evaluation_labels_by_method: dict[str, list[int]]
    frame_states: list[RealtimeFrameState]



def _configure_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def default_clusterer_params() -> tuple[dict[str, object], dict[str, object]]:
    clustream_kwargs: dict[str, object] = {
        "max_micro_clusters": 36,
        "radius_factor": 1.24,
        "min_radius": 0.82,
        "tentative_distance": 0.0,
        "stale_after": 90,
        "macro_rebuild_interval": 2,
        "emit_micro_labels": False,
    }
    denstream_kwargs: dict[str, object] = {
        "epsilon": 0.72,
        "beta": 0.35,
        "mu": 1.8,
        "lambd": 0.0015,
        "prune_interval": 8,
        "radius_tolerance": 1.16,
        "emit_noise": True,
        "grace_period": 2,
        "outlier_radius_tolerance": 1.16,
        "outlier_promote_weight": 1.10,
        "outlier_promote_age": 1,
        "min_points_before_noise": 3,
        "noise_score_threshold": 4.40,
    }
    return dict(clustream_kwargs), dict(denstream_kwargs)


def create_default_clusterers(
    selected_methods: list[str] | None = None,
    clustream_params: dict[str, object] | None = None,
    denstream_params: dict[str, object] | None = None,
) -> list[StreamClusterer]:
    clustream_kwargs, denstream_kwargs = default_clusterer_params()
    clustream_kwargs.update(clustream_params or {})
    denstream_kwargs.update(denstream_params or {})
    clusterers: list[StreamClusterer] = [
        CluStream(**clustream_kwargs),
        DenStream(**denstream_kwargs),
    ]
    if selected_methods:
        wanted = {method.strip().lower() for method in selected_methods if method.strip()}
        clusterers = [clusterer for clusterer in clusterers if clusterer.name.lower() in wanted]
    return clusterers




class ModelDrivenExpandingWindow:
    """Algorithm-owned expanding-window controller with fused assignment/commit.

    The previous implementation used ``probe()`` to decide when a candidate
    window should close and then called ``fit_predict_one()`` to classify that
    closed segment. This class removes that conceptual split. The candidate
    feature that receives a probe label is the same feature that is committed,
    and the concrete algorithm updates the cluster selected by that probe via
    ``commit_from_probe()``. Thus the model decision that closes the window is
    also the cluster assignment of the closed window.
    """

    def __init__(
        self,
        clusterer: StreamClusterer,
        min_window_size: int,
        feature_profile: str,
        standardize: bool,
        max_window_size: int | None,
        patience: int = 2,
        episode_reuse_distance: float = 1.75,
    ) -> None:
        self.clusterer = clusterer
        self.min_window_size = int(min_window_size)
        self.feature_profile = feature_profile
        # A hard max-window cutoff would be an external segmenter. In the
        # algorithm-owned realtime path it is kept only for very long real-world
        # streams; for 100-point synthetic tests it is None.
        self.max_window_size = max_window_size
        self.patience = max(1, int(patience))
        feature_count = len(extract_features_with_profile(np.zeros(self.min_window_size)))
        self.scaler = FeatureScaler(n_features=feature_count) if standardize else None
        self.buffer: list[float] = []
        self.start = 0
        self.bad_count = 0
        self.active_label: int | None = None
        self.last_stable_len: int | None = None
        self.last_stable_raw_feature: np.ndarray | None = None
        self.last_stable_probe = None
        self.segments: list[WindowSegment] = []
        self.labels: list[int] = []
        # Internal owner labels come from CluStream/DenStream. Visible labels are
        # temporal regime episodes: if the model returns to an old internal owner
        # after another owner, this is treated as a new regime episode online, not
        # remapped after the fact. This prevents old labels from reappearing on
        # visually different later parts of a stream.
        self._owner_to_current_episode: dict[int, int] = {}
        self._owner_last_feature: dict[int, np.ndarray] = {}
        self._episode_reuse_distance = float(episode_reuse_distance)
        self._last_internal_owner: int | None = None
        self._last_visible_episode: int | None = None
        self._next_visible_episode = 0

    @property
    def open_start(self) -> int | None:
        return self.start if self.buffer else None

    @property
    def open_size(self) -> int:
        return len(self.buffer)

    def _ownership_raw_feature(self, values: np.ndarray) -> np.ndarray:
        """Feature used both for algorithm probe and commit.

        It is still a feature of the current window, but it is deliberately
        recency-aware: long expanding windows are represented by a blend of the
        whole-window regime summary and the recent tail. This gives the
        algorithms enough information to own boundaries without a separate
        feature-distance segmenter. The same vector is committed, so there is no
        second classification step after closing.
        """
        full = extract_features_with_profile(values)
        if len(values) < 2 * self.min_window_size:
            return full
        tail_values = values[-self.min_window_size :]
        tail = extract_features_with_profile(tail_values)
        blended = 0.55 * full + 0.45 * tail
        # Level and scale dimensions should react fast enough to regime changes.
        # The remaining dimensions stay closer to the full window to keep the
        # fused controller stable.
        fast_indices = [0, 1, 2, 3, 4, 6, 10]
        for index in fast_indices:
            if index < len(blended):
                blended[index] = 0.35 * full[index] + 0.65 * tail[index]
        return blended

    def update(self, value: float, sample_index: int) -> None:
        if not self.buffer:
            self.start = int(sample_index)
        self.buffer.append(float(value))

        if len(self.buffer) < self.min_window_size:
            return

        raw_feature = self._ownership_raw_feature(np.asarray(self.buffer, dtype=float))
        feature = self.scaler.transform(raw_feature) if self.scaler is not None else raw_feature
        probe = self.clusterer.probe(feature, timestamp=sample_index)

        # The first complete candidate initializes the algorithm immediately.
        if not self.segments and not self.labels:
            self._commit_prefix(
                self.min_window_size,
                self.start + self.min_window_size,
                raw_feature=raw_feature,
                probe=probe,
            )
            return

        # Close only on a model-owned regime transition, not merely because the
        # candidate window has a high distance score. A bad score with the same
        # owner means "the current window is becoming heterogeneous"; it is not
        # enough evidence that a new cluster has started. This avoids cuts like
        # Cluster 0 | Cluster 0.
        candidate_label = int(probe.label) if probe.label != -1 else -1
        confirmed_owner = probe.compatible and not probe.is_noise and candidate_label != -1

        if confirmed_owner and (self.active_label is None or candidate_label == int(self.active_label)):
            self.active_label = candidate_label
            self.last_stable_len = len(self.buffer)
            self.last_stable_raw_feature = raw_feature.copy()
            self.last_stable_probe = probe
            self.bad_count = 0
            return

        if self.active_label is None and not probe.is_noise and not probe.is_novel:
            # No confirmed owner yet and no confirmed novelty either: keep collecting.
            return

        owner_changed = False
        if probe.is_noise:
            owner_changed = True
        elif probe.is_novel:
            owner_changed = True
        elif confirmed_owner and self.active_label is not None and candidate_label != int(self.active_label):
            owner_changed = True

        if not owner_changed:
            # Same owner but temporarily poor compatibility: keep expanding instead
            # of creating a boundary that would later receive the same label.
            return

        self.bad_count += 1

        should_close = self.bad_count >= self.patience
        if self.max_window_size is not None and len(self.buffer) >= self.max_window_size:
            should_close = True

        if not should_close:
            return

        if self.last_stable_len is not None and self.last_stable_len >= self.min_window_size:
            close_len = self.last_stable_len
            close_raw_feature = self.last_stable_raw_feature
            close_probe = self.last_stable_probe
        else:
            close_len = len(self.buffer)
            close_raw_feature = raw_feature
            close_probe = probe
        self._commit_prefix(
            close_len,
            self.start + close_len,
            raw_feature=close_raw_feature,
            probe=close_probe,
        )

    def finish(self, final_index: int) -> None:
        if len(self.buffer) < self.min_window_size:
            self.buffer = []
            self._reset_open_state()
            return

        raw_feature = self._ownership_raw_feature(np.asarray(self.buffer, dtype=float))
        feature = self.scaler.transform(raw_feature) if self.scaler is not None else raw_feature
        probe = self.clusterer.probe(feature, timestamp=final_index)

        # The end of stream is committed only if it is a real candidate window.
        # A tiny unsupported tail is not promoted into a cluster just because the
        # data ended.
        if probe.compatible or len(self.buffer) >= 2 * self.min_window_size or not self.segments:
            self._commit_prefix(len(self.buffer), final_index, raw_feature=raw_feature, probe=probe)
        else:
            self.buffer = []
            self._reset_open_state()

    def _commit_prefix(
        self,
        length: int,
        end: int,
        raw_feature: np.ndarray | None = None,
        probe = None,
    ) -> None:
        length = max(0, min(int(length), len(self.buffer)))
        if length <= 0:
            return
        values = np.asarray(self.buffer[:length], dtype=float)
        if raw_feature is None:
            raw_feature = extract_features_with_profile(values)
        feature = self.scaler.transform_update(raw_feature) if self.scaler is not None else raw_feature
        if probe is None:
            probe = self.clusterer.probe(feature, timestamp=int(end))
        segment = WindowSegment(
            index=len(self.segments),
            start=self.start,
            end=int(end),
            center=self.start + max(0, int(end) - self.start) // 2,
            values=values,
            feature=feature,
        )
        internal_label = self.clusterer.commit_from_probe(feature, segment.end, probe)
        label = self._visible_episode_label(int(internal_label), feature)
        self.segments.append(segment)
        self.labels.append(int(label))

        remainder = self.buffer[length:]
        self.start = int(end)
        self.buffer = list(remainder)
        self._reset_open_state()

    def _visible_episode_label(self, internal_label: int, feature: np.ndarray) -> int:
        if internal_label == -1:
            # Noise is uncertainty, not a temporal regime owner. It should not
            # become sticky and should not reserve a future episode id.
            self._last_internal_owner = None
            self._last_visible_episode = None
            return -1
        if self._last_internal_owner is None:
            episode = self._owner_to_current_episode.get(int(internal_label))
            if episode is None:
                episode = self._next_visible_episode
                self._next_visible_episode += 1
        elif int(internal_label) == int(self._last_internal_owner):
            episode = int(self._last_visible_episode)
        else:
            previous_feature = self._owner_last_feature.get(int(internal_label))
            if previous_feature is not None:
                distance = float(np.linalg.norm(np.asarray(feature, dtype=float) - previous_feature))
            else:
                distance = float("inf")
            if previous_feature is not None and distance <= self._episode_reuse_distance:
                # Genuine class return: the internal owner is old and the new
                # feature is still close enough to the last episode of that owner.
                episode = self._owner_to_current_episode.get(int(internal_label))
                if episode is None:
                    episode = self._next_visible_episode
                    self._next_visible_episode += 1
            else:
                # Same internal centroid/density id, but temporally and geometrically
                # far enough to be a new regime episode. This is online, not a
                # post-hoc relabeling pass.
                episode = self._next_visible_episode
                self._next_visible_episode += 1
        self._last_internal_owner = int(internal_label)
        self._last_visible_episode = int(episode)
        self._owner_to_current_episode[int(internal_label)] = int(episode)
        self._owner_last_feature[int(internal_label)] = np.asarray(feature, dtype=float).copy()
        return int(episode)

    def _reset_open_state(self) -> None:
        self.bad_count = 0
        self.active_label = None
        self.last_stable_len = None
        self.last_stable_raw_feature = None
        self.last_stable_probe = None



def _realtime_controller_patience(method_name: str, signal_length: int, real_unlabeled_stream: bool) -> int:
    if real_unlabeled_stream:
        # Algorithm-specific temporal policy: DenStream should require longer
        # density persistence than CluStream before closing a smooth real-world
        # financial/telecom regime. This also prevents both methods from being
        # forced into identical cuts by the shared controller shell.
        return 10 if method_name == "DenStream" else 8
    return 4 if signal_length <= 240 else 5

def _run_true_realtime_algorithm(
    stream: StreamSignal,
    window_size: int,
    window_mode: str = "expanding",
    feature_profile: str = "basic",
    standardize: bool = True,
) -> tuple[dict[str, list[WindowSegment]], dict[str, list[int]], dict[str, list[int]], list[StreamClusterer]]:
    if window_mode != "expanding":
        raise ValueError("Unsupported configuration")

    signal_length = len(stream.signal)
    effective_window_size = max(2, int(window_size))
    real_unlabeled_stream = _is_unlabeled_real_stream(stream)
    max_window_size = _adaptive_max_window_size(signal_length, effective_window_size)
    clusterers = create_default_clusterers()
    controllers = {
        clusterer.name: ModelDrivenExpandingWindow(
            clusterer=clusterer,
            min_window_size=effective_window_size,
            feature_profile=feature_profile,
            standardize=standardize,
            max_window_size=max_window_size,
            patience=_realtime_controller_patience(clusterer.name, signal_length, real_unlabeled_stream),
            episode_reuse_distance=(0.20 if real_unlabeled_stream else 3.50),
        )
        for clusterer in clusterers
    }

    for sample_index, value in enumerate(stream.signal):
        for controller in controllers.values():
            controller.update(float(value), sample_index)
    for controller in controllers.values():
        controller.finish(len(stream.signal))

    segments_by_method = {name: controller.segments for name, controller in controllers.items()}
    online_labels_by_method = {name: controller.labels for name, controller in controllers.items()}
    evaluation_labels_by_method = {name: remap_labels(labels) for name, labels in online_labels_by_method.items()}
    return segments_by_method, online_labels_by_method, evaluation_labels_by_method, clusterers

def run_realtime_true(
    dataset_path: str | Path,
    series_index: int,
    series_count: int,
    window_size: int,
    delay: float,
    balanced: bool = False,
    window_mode: str = "expanding",
    feature_profile: str = "basic",
    standardize: bool = True,
    debug_frames_dir: str | Path | None = None,
    title_prefix: str | None = None,
    max_points: int | None = None,
) -> None:
    if window_mode != "expanding":
        raise ValueError("Unsupported configuration")

    stream = _load_stream(
        dataset_path,
        series_index,
        series_count,
        balanced,
        max_points=max_points,
    )
    signal_length = len(stream.signal)
    effective_window_size = max(2, int(window_size))
    real_unlabeled_stream = _is_unlabeled_real_stream(stream)
    clusterers = create_default_clusterers()
    controllers = {
        clusterer.name: ModelDrivenExpandingWindow(
            clusterer=clusterer,
            min_window_size=effective_window_size,
            feature_profile=feature_profile,
            standardize=standardize,
            max_window_size=_adaptive_max_window_size(signal_length, effective_window_size),
            patience=_realtime_controller_patience(clusterer.name, signal_length, real_unlabeled_stream),
            episode_reuse_distance=(0.20 if real_unlabeled_stream else 3.50),
        )
        for clusterer in clusterers
    }
    method_names = list(controllers)
    frame_states: list[RealtimeFrameState] = []

    plotter = RealtimePlotter(
        stream.signal,
        method_names,
        view_width=len(stream.signal),
        stability=1,
        title_prefix=title_prefix or "Model-driven realtime",
        show_future_signal=False,
        reference_signal=stream.reference_signal,
    )

    for sample_index, value in enumerate(stream.signal):
        for controller in controllers.values():
            controller.update(float(value), sample_index)
        if sample_index + 1 < effective_window_size:
            continue
        frame_states.append(
            RealtimeFrameState(
                visible_end=sample_index + 1,
                processed_segments=0,
                processed_segments_by_method={name: len(controller.segments) for name, controller in controllers.items()},
                open_start_by_method={name: controller.open_start for name, controller in controllers.items()},
                open_size_by_method={name: controller.open_size for name, controller in controllers.items()},
                min_window_size=effective_window_size,
            )
        )

    for controller in controllers.values():
        controller.finish(len(stream.signal))
    frame_states.append(
        RealtimeFrameState(
            visible_end=len(stream.signal),
            processed_segments=0,
            processed_segments_by_method={name: len(controller.segments) for name, controller in controllers.items()},
            open_start_by_method={name: None for name in controllers},
            open_size_by_method={name: 0 for name in controllers},
            min_window_size=effective_window_size,
        )
    )

    segments_by_method = {name: controller.segments for name, controller in controllers.items()}
    online_labels_by_method = {name: controller.labels for name, controller in controllers.items()}
    evaluation_labels_by_method = {name: remap_labels(labels) for name, labels in online_labels_by_method.items()}
    boundary_metrics_by_method = {
        name: _compute_boundary_metrics_by_method(
            stream.source_labels,
            segments_by_method[name],
            {name: evaluation_labels_by_method[name]},
            boundary_tolerance=_default_boundary_tolerance_from_segments(segments_by_method[name]),
        ).get(name)
        for name in method_names
    }

    if debug_frames_dir is not None:
        plotter.save_frames(frame_states, segments_by_method, online_labels_by_method, debug_frames_dir)
    else:
        plotter.play(frame_states, segments_by_method, online_labels_by_method, delay)
    plotter.finish()

    for method_name in method_names:
        print(
            _format_run_report(
                {method_name: evaluation_labels_by_method[method_name]},
                [clusterer for clusterer in clusterers if clusterer.name == method_name],
                {method_name: boundary_metrics_by_method[method_name]},
                stream.source_labels,
                segments_by_method[method_name],
            )
        )

def run_all_synthetic_realtime(
    delay: float = 0.08,
    window_size: int = 10,
    step: int = 1,
    standardize: bool = False,
) -> None:
    scenarios = build_synthetic_realtime_scenarios(
        window_size=window_size,
        step=step,
        standardize=standardize,
    )
    method_names = [
        clusterer.name
        for clusterer in create_default_clusterers()
    ]
    dashboard = RealtimeDashboardPlotter(
        [
            RealtimeScenarioState(
                name=scenario.name,
                signal=scenario.stream.signal,
                reference_signal=scenario.stream.reference_signal,
                segments=scenario.segments,
                labels_by_method=scenario.labels_by_method,
                frame_states=scenario.frame_states,
            )
            for scenario in scenarios
        ],
        method_names,
        stability=1,
    )
    frame_count = max((len(scenario.frame_states) for scenario in scenarios), default=0)
    dashboard.play(frame_count, delay)
    dashboard.finish()
    for scenario in scenarios:
        boundary_metrics_by_method = _compute_boundary_metrics_by_method(
            scenario.stream.source_labels,
            scenario.segments,
            scenario.evaluation_labels_by_method,
            boundary_tolerance=_default_boundary_tolerance_from_segments(scenario.segments),
        )
        print(
            _format_run_report(
                scenario.evaluation_labels_by_method,
                [],
                boundary_metrics_by_method,
                scenario.stream.source_labels,
                scenario.segments,
            )
        )


def _is_unlabeled_real_stream(stream: StreamSignal) -> bool:
    labels = np.asarray(stream.source_labels, dtype=int)
    if labels.size == 0:
        return True
    return bool(np.unique(labels).size <= 1 and stream.reference_signal is None)


def _adaptive_max_window_size(signal_length: int, min_window_size: int) -> int | None:
    if signal_length <= 240:
        return None
    return min(signal_length, max(min_window_size * 4, min(min_window_size * 8, signal_length // 3)))


def build_synthetic_realtime_jobs(
    window_size: int = 10,
    step: int = 1,
    window_mode: str = "expanding",
    feature_profile: str = "basic",
    ) -> list[SyntheticRealtimeJob]:
    return [
        SyntheticRealtimeJob(
            name=spec.name,
            dataset_path=f"synthetic:{spec.key}",
            series_index=0,
            series_count=1,
            window_size=window_size,
            step=step,
            balanced=False,
            window_mode=window_mode,
            feature_profile=feature_profile,
        )
        for spec in list_synthetic_dataset_specs()
    ]


def build_synthetic_realtime_scenarios(
    window_size: int = 10,
    step: int = 1,
    standardize: bool = False,
) -> list[SyntheticRealtimeScenario]:
    scenarios: list[SyntheticRealtimeScenario] = []
    for job in build_synthetic_realtime_jobs(
        window_size=window_size,
        step=step,
    ):
        print(f"Preparing realtime synthetic scenario: {job.name}")
        scenarios.append(
            _build_synthetic_realtime_scenario(
                job,
                standardize=standardize,
            )
        )
    return scenarios


def _build_synthetic_realtime_scenario(
    job: SyntheticRealtimeJob,
    standardize: bool,
    ) -> SyntheticRealtimeScenario:
    return _build_realtime_scenario(
        job,
        standardize=standardize,
    )


def _build_realtime_scenario(
    job: SyntheticRealtimeJob,
    standardize: bool,
) -> SyntheticRealtimeScenario:
    """Build a realtime scenario using the same controller as normal UI realtime.

    Metrics/dashboards must evaluate the main mode of the project, so this path
    uses ModelDrivenExpandingWindow through _run_true_realtime_algorithm instead
    of prebuilt fixed windows.
    """
    stream = _load_stream(job.dataset_path, job.series_index, job.series_count, job.balanced)
    segments_by_method, online_labels_by_method, evaluation_labels_by_method, _clusterers = _run_true_realtime_algorithm(
        stream=stream,
        window_size=job.window_size,
        window_mode=job.window_mode,
        feature_profile=job.feature_profile,
        standardize=standardize,
    )
    frame_states = [
        RealtimeFrameState(
            visible_end=index + 1,
            processed_segments=0,
            processed_segments_by_method={name: sum(1 for segment in segments if segment.end <= index + 1) for name, segments in segments_by_method.items()},
            open_start_by_method={name: None for name in segments_by_method},
            open_size_by_method={name: 0 for name in segments_by_method},
            min_window_size=max(2, int(job.window_size)),
        )
        for index in range(max(2, int(job.window_size)) - 1, len(stream.signal))
    ]
    # Dashboard expects one segment list, but true realtime owns segments per
    # method. Use the first method only for compatibility with this legacy
    # dashboard path; metrics are computed per method elsewhere.
    first_method = next(iter(segments_by_method), "")
    return SyntheticRealtimeScenario(
        name=job.name,
        stream=stream,
        segments=segments_by_method.get(first_method, []),
        labels_by_method=online_labels_by_method,
        evaluation_labels_by_method=evaluation_labels_by_method,
        frame_states=frame_states,
    )

def _load_stream(
    dataset_path: str | Path,
    series_index: int,
    series_count: int,
    balanced: bool,
    max_points: int | None = None,
) -> StreamSignal:
    dataset_key = str(dataset_path).lower()
    synthetic_prefix = "synthetic:"
    uci_prefix = "uci:"
    if dataset_key.startswith(synthetic_prefix):
        return make_synthetic_segmentation_stream(dataset_key[len(synthetic_prefix) :])
    if dataset_key.startswith(uci_prefix):
        return load_uci_stream(dataset_key[len(uci_prefix) :], max_points=max_points)

    dataset_path = Path(dataset_path)
    normalized = dataset_path.as_posix().lower()
    root_csvs = {
        "nab_traffic_occupancy_1000_with_hint.csv",
        "skab_valve1_flow.csv",
        "skab_rotor_imbalance_vibration.csv",
        "skab_hot_water_temperature.csv",
        "uci_occupancy_light.csv",
        "nab_ec2_cpu.csv",
    }
    if dataset_path.name.lower() in root_csvs:
        return load_labeled_csv_stream(
            dataset_path,
            value_column="value",
            label_column="regime_hint",
            max_points=max_points,
        )
    if "operating_regime_datasets_500" in normalized:
        return load_labeled_csv_stream(
            dataset_path,
            value_column="value",
            label_column="regime_hint",
            max_points=max_points,
        )
    if "small_regime_datasets" in normalized:
        name = dataset_path.name.lower()
        if name == "appliances_energy_1000_with_daypart_hint.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="Appliances",
                label_column="regime_hint_by_time",
                max_points=max_points,
            )
        if name == "occupancy_test1_1000_labeled.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="CO2",
                label_column="Occupancy",
                max_points=max_points,
            )
        if name == "hapt_exp01_user01_1000_labeled.csv":
            return load_multivariate_csv_stream(
                dataset_path,
                feature_columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                label_column="activity_id",
                max_points=max_points,
            )
        if name == "nab_ec2_cpu_1000_with_hint.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="value",
                label_column="regime_hint_by_value",
                max_points=max_points,
            )
        if name == "nab_traffic_occupancy_1000_with_hint.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="value",
                label_column="regime_hint_by_value",
                max_points=max_points,
            )
        return load_univariate_csv_stream(dataset_path, max_points=max_points)
    if "small_original_regime_datasets" in normalized:
        name = dataset_path.name.lower()
        if name == "nile_100_labeled.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="volume",
                label_column="regime",
                max_points=max_points,
            )
        if name == "sunspots_yearly_309_regime_hints.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="sunactivity",
                label_column="regime_hint_by_intensity",
                max_points=max_points,
            )
        if name == "elnino_monthly_732_regime_hints.csv":
            return load_labeled_csv_stream(
                dataset_path,
                value_column="sst",
                label_column="regime_hint_by_anomaly",
                max_points=max_points,
            )
        if name == "us_macro_quarterly_203_recession_labeled.csv":
            return load_multivariate_csv_stream(
                dataset_path,
                feature_columns=["realgdp", "realcons", "realinv", "tbilrate", "unemp", "infl", "realint"],
                label_column="regime_label",
                max_points=max_points,
            )
        if name == "germany_interest_inflation_107_regime_hints.csv":
            return load_multivariate_csv_stream(
                dataset_path,
                feature_columns=["Dp", "R"],
                label_column="regime_hint_by_rate_inflation",
                max_points=max_points,
            )
        return load_univariate_csv_stream(dataset_path, max_points=max_points)

    if not dataset_path.exists() and not dataset_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / dataset_path
        if candidate.exists():
            dataset_path = candidate

    if any(part.lower() == "uci" for part in dataset_path.parts):
        return load_uci_stream(dataset_path, max_points=max_points)
    if dataset_path.suffix.lower() == ".csv":
        return load_univariate_csv_stream(dataset_path, max_points=max_points)

    dataset = load_ucr_tsv(dataset_path)
    return build_stream_signal(dataset, series_index, series_count, balanced=balanced)



def _evaluate_true_realtime_synthetic_run(
    stream: StreamSignal,
    window_size: int,
    standardize: bool = True,
) -> tuple[dict[str, list[WindowSegment]], dict[str, list[int]], dict[str, BoundaryMetrics | None], dict[str, dict[str, float | int]]]:
    """Run the exact main realtime controller and compute metrics per algorithm.

    This helper is used by all synthetic metric modes. It keeps metric reports
    consistent with the UI Start tab: ModelDrivenExpandingWindow, probe(),
    commit_from_probe(), fixed feature scaling, and algorithm-owned boundaries.
    """
    segments_by_method, _online_labels_by_method, evaluation_labels_by_method, _clusterers = _run_true_realtime_algorithm(
        stream=stream,
        window_size=window_size,
        window_mode="expanding",
        feature_profile="basic",
        standardize=standardize,
    )
    boundary_by_method: dict[str, BoundaryMetrics | None] = {}
    quality_by_method: dict[str, dict[str, float | int]] = {}
    for method_name, labels in evaluation_labels_by_method.items():
        method_segments = segments_by_method[method_name]
        metrics = _compute_boundary_metrics_by_method(
            stream.source_labels,
            method_segments,
            {method_name: labels},
            boundary_tolerance=_default_boundary_tolerance_from_segments(method_segments),
        ).get(method_name)
        boundary_by_method[method_name] = metrics
        quality_by_method[method_name] = _synthetic_cluster_quality_metrics(stream, method_segments, labels)
    return segments_by_method, evaluation_labels_by_method, boundary_by_method, quality_by_method

def run_synthetic_validation(
    output_dir: str | Path,
    window_size: int = 10,
    standardize: bool = True,
) -> list[SyntheticValidationSummary]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[SyntheticValidationSummary] = []
    lines: list[str] = [
        "Synthetic validation report",
        "Mode: main realtime controller (ModelDrivenExpandingWindow)",
        "",
    ]
    for spec in list_synthetic_validation_dataset_specs():
        stream = make_synthetic_segmentation_stream(spec.key, segment_length=100)
        segments_by_method, evaluation_labels_by_method, _boundary_by_method, quality_by_method = _evaluate_true_realtime_synthetic_run(
            stream,
            window_size=window_size,
            standardize=standardize,
        )
        true_boundaries = _true_boundaries(stream.source_labels)
        lines.append(f"[{spec.name}]")
        for method_name, labels in evaluation_labels_by_method.items():
            segments = segments_by_method[method_name]
            metrics = _synthetic_validation_metrics(stream, segments, labels, true_boundaries)
            summary = SyntheticValidationSummary(
                dataset=spec.name,
                method=method_name,
                segment_count=len(segments),
                true_boundary_count=len(true_boundaries),
                boundary_mae=metrics["boundary_mae"],
                boundary_max_error=metrics["boundary_max_error"],
                segment_purity=metrics["segment_purity"],
                cluster_count=int(quality_by_method[method_name].get("cluster_count", metrics["cluster_count"])),
                noise_ratio=float(quality_by_method[method_name].get("noise_ratio", metrics["noise_ratio"])),
                largest_cluster_ratio=float(quality_by_method[method_name].get("largest_cluster_ratio", metrics["largest_cluster_ratio"])),
                score=float(quality_by_method[method_name].get("score", metrics["score"])),
                passed=bool(metrics["passed"]),
            )
            summaries.append(summary)
            lines.append(
                "  "
                f"{method_name}: pass={summary.passed}, "
                f"segments={summary.segment_count}, "
                f"boundaries={summary.true_boundary_count}, "
                f"boundary_mae={summary.boundary_mae:.1f}, "
                f"boundary_max={summary.boundary_max_error:.1f}, "
                f"purity={summary.segment_purity:.3f}, "
                f"clusters={summary.cluster_count}, "
                f"noise={summary.noise_ratio:.1%}, "
                f"largest={summary.largest_cluster_ratio:.1%}, "
                f"score={summary.score:.3f}"
            )
        lines.append("")

    report_path = output_dir / "synthetic_validation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summaries

def run_synthetic_boundary_metrics_benchmark(
    output_dir: str | Path,
    runs_per_dataset: int = 10,
    window_size: int = 10,
    standardize: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if runs_per_dataset <= 0:
        raise ValueError("runs_per_dataset must be greater than 0")

    specs = list_synthetic_dataset_specs()
    lines: list[str] = [
        "Synthetic boundary-metrics benchmark",
        f"Runs per dataset: {runs_per_dataset}",
        f"Window size: {window_size}",
        "Mode: main realtime controller (ModelDrivenExpandingWindow, probe+commit)",
        "",
    ]

    for spec_index, spec in enumerate(specs):
        per_method: dict[str, list[BoundaryMetrics]] = {}
        for run_index in range(runs_per_dataset):
            stream = make_synthetic_segmentation_stream(
                spec.key,
                segment_length=100,
                noise_seed=10_000 * spec_index + run_index,
            )
            _segments_by_method, _evaluation_labels_by_method, boundary_by_method, _quality_by_method = _evaluate_true_realtime_synthetic_run(
                stream,
                window_size=window_size,
                standardize=standardize,
            )
            for method_name, metrics in boundary_by_method.items():
                if metrics is not None:
                    per_method.setdefault(method_name, []).append(metrics)

        lines.append(f"[{spec.name}]")
        for method_name in sorted(per_method):
            aggregated = _aggregate_boundary_metrics(per_method[method_name], method_name)
            lines.extend(_format_boundary_metrics_block(method_name, aggregated))
        lines.append("")

    report_path = output_dir / "synthetic_boundary_metrics_benchmark.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

def run_synthetic_full_metrics_benchmark(
    output_dir: str | Path,
    runs_per_dataset: int = 20,
    window_size: int = 10,
) -> None:
    """Full synthetic metrics for the main realtime mode.

    Earlier versions compared several legacy fixed-window settings. The project
    now has one primary mode, so the full report is an expanded report for that
    same realtime controller.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if runs_per_dataset <= 0:
        raise ValueError("runs_per_dataset must be greater than 0")

    specs = list_synthetic_dataset_specs()
    lines: list[str] = [
        "Synthetic full-metrics benchmark",
        f"Runs per dataset: {runs_per_dataset}",
        f"Window size: {window_size}",
        "Mode: main realtime controller (ModelDrivenExpandingWindow, probe+commit)",
        "Metric sets: boundary metrics + cluster quality metrics",
        "",
    ]

    for spec_index, spec in enumerate(specs):
        lines.append(f"[{spec.name}]")
        per_method_boundary: dict[str, list[BoundaryMetrics]] = {}
        per_method_quality: dict[str, list[dict[str, float | int]]] = {}
        for run_index in range(runs_per_dataset):
            stream = make_synthetic_segmentation_stream(
                spec.key,
                segment_length=100,
                noise_seed=10_000 * spec_index + run_index,
            )
            _segments_by_method, _evaluation_labels_by_method, boundary_by_method, quality_by_method = _evaluate_true_realtime_synthetic_run(
                stream,
                window_size=window_size,
                standardize=True,
            )
            for method_name, metrics in boundary_by_method.items():
                if metrics is not None:
                    per_method_boundary.setdefault(method_name, []).append(metrics)
            for method_name, quality_metrics in quality_by_method.items():
                per_method_quality.setdefault(method_name, []).append(quality_metrics)

        for method_name in sorted(per_method_quality):
            boundary_summary = (
                _aggregate_boundary_metrics(per_method_boundary.get(method_name, []), method_name)
                if per_method_boundary.get(method_name)
                else None
            )
            quality_summary = _aggregate_cluster_quality_metrics(
                per_method_quality[method_name],
                method_name,
            )
            lines.extend(_format_full_metrics_block(
                method_name,
                boundary_summary,
                quality_summary,
            ))
        lines.append("")

    report_path = output_dir / "synthetic_full_metrics_benchmark.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

def run_synthetic_recommended_metrics_benchmark(
    output_dir: str | Path,
    runs_per_dataset: int = 30,
    window_size: int = 10,
) -> None:
    """Recommended metrics for the main realtime configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if runs_per_dataset <= 0:
        raise ValueError("runs_per_dataset must be greater than 0")

    specs = list_synthetic_dataset_specs()
    lines: list[str] = [
        "Synthetic recommended full-metrics benchmark",
        f"Runs per dataset: {runs_per_dataset}",
        f"Window size: {window_size}",
        "Settings: main realtime controller, normalized fixed feature scaling",
        "Metric sets: boundary metrics + cluster quality metrics",
        "",
    ]

    for spec_index, spec in enumerate(specs):
        lines.append(f"[{spec.name}]")
        per_method_boundary: dict[str, list[BoundaryMetrics]] = {}
        per_method_quality: dict[str, list[dict[str, float | int]]] = {}

        for run_index in range(runs_per_dataset):
            stream = make_synthetic_segmentation_stream(
                spec.key,
                segment_length=100,
                noise_seed=10_000 * spec_index + run_index,
            )
            _segments_by_method, _evaluation_labels_by_method, boundary_by_method, quality_by_method = _evaluate_true_realtime_synthetic_run(
                stream,
                window_size=window_size,
                standardize=True,
            )
            for method_name, metrics in boundary_by_method.items():
                if metrics is not None:
                    per_method_boundary.setdefault(method_name, []).append(metrics)
            for method_name, quality_metrics in quality_by_method.items():
                per_method_quality.setdefault(method_name, []).append(quality_metrics)

        for method_name in sorted(per_method_quality):
            boundary_summary = (
                _aggregate_boundary_metrics(per_method_boundary.get(method_name, []), method_name)
                if per_method_boundary.get(method_name)
                else None
            )
            quality_summary = _aggregate_cluster_quality_metrics(
                per_method_quality[method_name],
                method_name,
            )
            lines.extend(_format_full_metrics_block(
                method_name,
                boundary_summary,
                quality_summary,
            ))
        lines.append("")

    report_path = output_dir / "synthetic_recommended_full_metrics_30runs.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

def run_synthetic_realtime_metrics_benchmark(
    output_dir: str | Path,
    runs_per_dataset: int = 30,
    window_size: int = 10,
) -> None:
    """Benchmark the true realtime path used by the UI Start tab."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if runs_per_dataset <= 0:
        raise ValueError("runs_per_dataset must be greater than 0")

    specs = list_synthetic_dataset_specs()
    lines: list[str] = [
        "Synthetic true realtime full-metrics benchmark",
        f"Runs per dataset: {runs_per_dataset}",
        f"Window size: {window_size}",
        "Settings: main realtime controller (ModelDrivenExpandingWindow, probe+commit)",
        "Metric sets: boundary metrics + cluster quality metrics",
        "Labels: online visible regime labels, dense-id remap only",
        "",
    ]

    for spec_index, spec in enumerate(specs):
        lines.append(f"[{spec.name}]")
        per_method_boundary: dict[str, list[BoundaryMetrics]] = {}
        per_method_quality: dict[str, list[dict[str, float | int]]] = {}

        for run_index in range(runs_per_dataset):
            stream = make_synthetic_segmentation_stream(
                spec.key,
                segment_length=100,
                noise_seed=10_000 * spec_index + run_index,
            )
            _segments_by_method, _evaluation_labels_by_method, boundary_by_method, quality_by_method = _evaluate_true_realtime_synthetic_run(
                stream,
                window_size=window_size,
                standardize=True,
            )
            for method_name, metrics in boundary_by_method.items():
                if metrics is not None:
                    per_method_boundary.setdefault(method_name, []).append(metrics)
            for method_name, quality_metrics in quality_by_method.items():
                per_method_quality.setdefault(method_name, []).append(quality_metrics)

        for method_name in sorted(per_method_quality):
            boundary_summary = (
                _aggregate_boundary_metrics(per_method_boundary.get(method_name, []), method_name)
                if per_method_boundary.get(method_name)
                else None
            )
            quality_summary = _aggregate_cluster_quality_metrics(
                per_method_quality[method_name],
                method_name,
            )
            lines.extend(_format_full_metrics_block(
                method_name,
                boundary_summary,
                quality_summary,
            ))
        lines.append("")

    report_path = output_dir / "synthetic_true_realtime_metrics_30runs.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

def _aggregate_boundary_metrics(metrics_list: list[BoundaryMetrics], method_name: str) -> BoundaryMetrics:
    if not metrics_list:
        raise ValueError("metrics_list must not be empty")

    def avg(values: list[float | None]) -> float | None:
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        return float(np.mean(finite)) if finite else None

    def worst(values: list[float | None]) -> float | None:
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        return float(np.max(finite)) if finite else None

    tp = int(sum(metric.tp or 0 for metric in metrics_list))
    fp = int(sum(metric.fp or 0 for metric in metrics_list))
    fn = int(sum(metric.fn or 0 for metric in metrics_list))
    tn = int(sum(metric.tn or 0 for metric in metrics_list))
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)

    return BoundaryMetrics(
        method=method_name,
        tolerance=int(round(float(np.mean([metric.tolerance for metric in metrics_list])))),
        true_boundary_count=int(round(float(np.mean([metric.true_boundary_count for metric in metrics_list])))),
        predicted_boundary_count=int(round(float(np.mean([metric.predicted_boundary_count for metric in metrics_list])))),
        boundary_mae=avg([metric.boundary_mae for metric in metrics_list]),
        boundary_median_error=avg([metric.boundary_median_error for metric in metrics_list]),
        # In a multi-run report this value must represent the worst observed
        # boundary error across all runs, not the average of per-run maxima.
        # Otherwise scenarios with one true boundary can show identical MAE and
        # max error after aggregation, which hides rare bad runs.
        boundary_max_error=worst([metric.boundary_max_error for metric in metrics_list]),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        far=far,
        frr=frr,
        accuracy=accuracy,
        recall=recall,
        precision=precision,
    )


def _aggregate_cluster_quality_metrics(
    metrics_list: list[dict[str, float | int]],
    method_name: str,
) -> dict[str, float | int]:
    if not metrics_list:
        raise ValueError("metrics_list must not be empty")

    def avg(key: str) -> float:
        values = [float(metric[key]) for metric in metrics_list if key in metric and np.isfinite(float(metric[key]))]
        return float(np.mean(values)) if values else 0.0

    return {
        "method": method_name,
        "windows": avg("windows"),
        "cluster_count": avg("cluster_count"),
        "noise_count": avg("noise_count"),
        "noise_ratio": avg("noise_ratio"),
        "largest_cluster_ratio": avg("largest_cluster_ratio"),
        "label_switch_ratio": avg("label_switch_ratio"),
        "visual_run_count": avg("visual_run_count"),
        "mean_visual_run_length": avg("mean_visual_run_length"),
        "short_visual_run_ratio": avg("short_visual_run_ratio"),
        "transitions_per_100_samples": avg("transitions_per_100_samples"),
        "entropy": avg("entropy"),
        "purity": avg("purity"),
        "score": avg("score"),
    }


def _synthetic_cluster_quality_metrics(
    stream: StreamSignal,
    segments: list[WindowSegment],
    labels: list[int],
) -> dict[str, float | int]:
    sample_labels = window_labels_to_sample_labels(
        len(stream.signal),
        segments,
        labels,
    )
    source_classes = [
        int(stream.source_labels[min(segment.center, len(stream.source_labels) - 1)])
        for segment in segments
    ]
    cluster_metrics = cluster_quality_metrics(labels, source_classes, sample_labels=sample_labels)
    return {
        "windows": float(cluster_metrics.windows),
        "cluster_count": float(cluster_metrics.cluster_count),
        "noise_count": float(cluster_metrics.noise_count),
        "noise_ratio": float(cluster_metrics.noise_ratio),
        "largest_cluster_ratio": float(cluster_metrics.largest_cluster_ratio),
        "label_switch_ratio": float(cluster_metrics.label_switch_ratio),
        "visual_run_count": float(cluster_metrics.visual_run_count),
        "mean_visual_run_length": float(cluster_metrics.mean_visual_run_length),
        "short_visual_run_ratio": float(cluster_metrics.short_visual_run_ratio),
        "transitions_per_100_samples": float(cluster_metrics.transitions_per_100_samples),
        "entropy": float(cluster_metrics.entropy),
        "purity": float(cluster_metrics.purity),
        "score": float(cluster_metrics.score),
    }


def _format_full_metrics_block(
    method_name: str,
    boundary_metrics: BoundaryMetrics | None,
    cluster_metrics: dict[str, float | int],
) -> list[str]:
    lines = [f"  Method: {method_name}"]
    lines.extend(
        [
            f"    Settings summary:",
            f"      windows average: {_format_float_or_na(cluster_metrics.get('windows', None))}",
            f"      cluster count average: {_format_float_or_na(cluster_metrics.get('cluster_count', None))}",
            f"      noise windows average: {_format_float_or_na(cluster_metrics.get('noise_count', None))}",
            f"      noise ratio average: {_format_float_or_na(cluster_metrics.get('noise_ratio', None))}",
            f"      largest cluster ratio average: {_format_float_or_na(cluster_metrics.get('largest_cluster_ratio', None))}",
            f"      label switch ratio average: {_format_float_or_na(cluster_metrics.get('label_switch_ratio', None))}",
            f"      visual run count average: {_format_float_or_na(cluster_metrics.get('visual_run_count', None))}",
            f"      mean visual run length average: {_format_float_or_na(cluster_metrics.get('mean_visual_run_length', None))}",
            f"      short visual run ratio average: {_format_float_or_na(cluster_metrics.get('short_visual_run_ratio', None))}",
            f"      transitions per 100 samples average: {_format_float_or_na(cluster_metrics.get('transitions_per_100_samples', None))}",
            f"      entropy average: {_format_float_or_na(cluster_metrics.get('entropy', None))}",
            f"      purity average: {_format_float_or_na(cluster_metrics.get('purity', None))}",
            f"      score average: {_format_float_or_na(cluster_metrics.get('score', None))}",
        ]
    )
    lines.extend(
        [
            f"    Boundary metrics:",
        ]
    )
    lines.extend([f"      {line.strip()}" for line in _format_boundary_metrics_block(method_name, boundary_metrics)[1:]])
    return lines


def _process_segments(
    clusterers: list[StreamClusterer],
    segments: list[WindowSegment],
) -> dict[str, list[int]]:
    labels_by_method: dict[str, list[int]] = {clusterer.name: [] for clusterer in clusterers}
    for segment in segments:
        for clusterer in clusterers:
            label = clusterer.fit_predict_one(segment.feature, segment.index)
            labels_by_method[clusterer.name].append(label)
    return labels_by_method


def _synthetic_validation_metrics(
    stream: StreamSignal,
    segments: list[WindowSegment],
    labels: list[int],
    true_boundaries: list[int],
) -> dict[str, float | bool]:
    sample_labels = window_labels_to_sample_labels(
        len(stream.signal),
        segments,
        labels,
    )
    source_classes = [
        int(stream.source_labels[min(segment.center, len(stream.source_labels) - 1)])
        for segment in segments
    ]
    cluster_metrics = cluster_quality_metrics(labels, source_classes, sample_labels=sample_labels)
    predicted_boundaries = [segment.end for segment in segments[:-1]]
    boundary_errors = [
        min((abs(boundary - predicted) for predicted in predicted_boundaries), default=len(stream.signal))
        for boundary in true_boundaries
    ]
    boundary_mae = float(np.mean(boundary_errors)) if boundary_errors else 0.0
    boundary_max_error = float(np.max(boundary_errors)) if boundary_errors else 0.0
    segment_purity = _segment_purity(stream.source_labels, segments)
    passed = (
        boundary_mae <= max(10.0, len(stream.signal) * 0.02)
        and boundary_max_error <= max(20.0, len(stream.signal) * 0.04)
        and segment_purity >= 0.8
        and cluster_metrics.noise_ratio <= 0.4
    )
    return {
        "boundary_mae": boundary_mae,
        "boundary_max_error": boundary_max_error,
        "segment_purity": segment_purity,
        "cluster_count": float(cluster_metrics.cluster_count),
        "noise_ratio": float(cluster_metrics.noise_ratio),
        "largest_cluster_ratio": float(cluster_metrics.largest_cluster_ratio),
        "score": float(cluster_metrics.score),
        "passed": passed,
    }


def _true_boundaries(source_labels: np.ndarray) -> list[int]:
    boundaries: list[int] = []
    for index in range(1, len(source_labels)):
        if source_labels[index] != source_labels[index - 1]:
            boundaries.append(index)
    return boundaries


def _segment_purity(source_labels: np.ndarray, segments: list[WindowSegment]) -> float:
    if not segments:
        return 0.0

    scores: list[float] = []
    for segment in segments:
        segment_labels = source_labels[segment.start : segment.end]
        if len(segment_labels) == 0:
            continue
        _, counts = np.unique(segment_labels, return_counts=True)
        scores.append(float(np.max(counts) / len(segment_labels)))
    return float(np.mean(scores)) if scores else 0.0



def remap_labels(labels: list[int]) -> list[int]:
    """Make cluster labels dense and stable for readable CSV files and plots."""
    mapping: dict[int, int] = {}
    next_label = 0
    remapped: list[int] = []
    for label in labels:
        if label == -1:
            remapped.append(-1)
            continue
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        remapped.append(mapping[label])
    return remapped


def _default_boundary_tolerance(window_size: int) -> int:
    return max(5, min(15, window_size // 6))


def _default_boundary_tolerance_from_segments(segments: list[WindowSegment]) -> int:
    if not segments:
        return 5
    median_length = int(np.median([segment.end - segment.start for segment in segments]))
    return max(5, min(15, max(1, median_length // 6)))


def _format_run_report(
    labels_by_method: dict[str, list[int]],
    clusterers: list[StreamClusterer],
    boundary_metrics_by_method: dict[str, BoundaryMetrics | None] | None,
    source_labels: np.ndarray,
    segments: list[WindowSegment],
) -> str:
    lines = [
        "Streaming time-series clustering comparison",
        f"Objects/windows processed: {len(segments)}",
        "",
    ]
    for method_name, labels in labels_by_method.items():
        metrics = (boundary_metrics_by_method or {}).get(method_name)
        lines.extend(_format_boundary_metrics_block(method_name, metrics))
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_boundary_metrics_block(method_name: str, metrics: BoundaryMetrics | None) -> list[str]:
    if metrics is None:
        return [
            f"{method_name}:",
            "  boundary MAE: n/a",
            "  boundary max error: n/a",
            "  FAR: n/a",
            "  FRR: n/a",
            "  Accuracy: n/a",
            "  Recall: n/a",
            "  Precision: n/a",
        ]

    return [
        f"{method_name}:",
        f"  boundary MAE: {_format_float_or_na(metrics.boundary_mae)}",
        f"  boundary max error: {_format_float_or_na(metrics.boundary_max_error)}",
        f"  FAR: {_format_float_or_na(metrics.far)}",
        f"  FRR: {_format_float_or_na(metrics.frr)}",
        f"  Accuracy: {_format_float_or_na(metrics.accuracy)}",
        f"  Recall: {_format_float_or_na(metrics.recall)}",
        f"  Precision: {_format_float_or_na(metrics.precision)}",
    ]


def _format_float_or_na(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _compute_boundary_metrics_by_method(
    source_labels: np.ndarray,
    segments: list[WindowSegment],
    labels_by_method: dict[str, list[int]],
    boundary_tolerance: int,
) -> dict[str, BoundaryMetrics | None]:
    return {
        method_name: _compute_boundary_metrics(
            source_labels,
            segments,
            labels,
            boundary_tolerance=boundary_tolerance,
            method_name=method_name,
        )
        for method_name, labels in labels_by_method.items()
    }


def _compute_boundary_metrics(
    source_labels: np.ndarray,
    segments: list[WindowSegment],
    labels: list[int],
    boundary_tolerance: int,
    method_name: str,
) -> BoundaryMetrics | None:
    if len(source_labels) == 0:
        return None

    true_boundaries = _true_boundaries(source_labels)
    if not true_boundaries:
        return None

    sample_labels = window_labels_to_sample_labels(
        len(source_labels),
        segments,
        labels,
    )
    predicted_boundaries = _sample_label_boundaries(sample_labels)

    boundary_errors = [
        min((abs(true_boundary - predicted) for predicted in predicted_boundaries), default=float("inf"))
        for true_boundary in true_boundaries
    ]
    finite_errors = [float(error) for error in boundary_errors if np.isfinite(error)]
    boundary_mae = float(np.mean(finite_errors)) if finite_errors else None
    boundary_max_error = float(np.max(finite_errors)) if finite_errors else None

    true_mask = _boundary_mask(len(source_labels), true_boundaries, boundary_tolerance)
    predicted_mask = _boundary_mask(len(source_labels), predicted_boundaries, boundary_tolerance)

    tp = int(np.sum(true_mask & predicted_mask))
    fp = int(np.sum(~true_mask & predicted_mask))
    fn = int(np.sum(true_mask & ~predicted_mask))
    tn = int(np.sum(~true_mask & ~predicted_mask))

    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)

    return BoundaryMetrics(
        method=method_name,
        tolerance=boundary_tolerance,
        true_boundary_count=len(true_boundaries),
        predicted_boundary_count=len(predicted_boundaries),
        boundary_mae=boundary_mae,
        boundary_median_error=None,
        boundary_max_error=boundary_max_error,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        far=far,
        frr=frr,
        accuracy=accuracy,
        recall=recall,
        precision=precision,
    )


def _sample_label_boundaries(labels: np.ndarray) -> list[int]:
    boundaries: list[int] = []
    for index in range(1, len(labels)):
        if labels[index] != labels[index - 1]:
            boundaries.append(index)
    return boundaries


def _boundary_mask(length: int, boundaries: list[int], tolerance: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if length <= 0:
        return mask

    radius = max(0, tolerance)
    for boundary in boundaries:
        start = max(0, boundary - radius)
        end = min(length, boundary + radius + 1)
        mask[start:end] = True
    return mask



