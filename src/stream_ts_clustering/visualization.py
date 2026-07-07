from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from .postprocessing import sample_label_runs, window_labels_to_sample_labels
from .preprocessing import WindowSegment


@dataclass(frozen=True)
class RealtimeFrameState:
    visible_end: int
    processed_segments: int
    open_start: int | None = None
    open_size: int = 0
    min_window_size: int = 0
    processed_segments_by_method: dict[str, int] | None = None
    open_start_by_method: dict[str, int | None] | None = None
    open_size_by_method: dict[str, int] | None = None


@dataclass(frozen=True)
class RealtimeScenarioState:
    name: str
    signal: np.ndarray
    reference_signal: np.ndarray | None
    segments: list[WindowSegment]
    labels_by_method: dict[str, list[int]]
    frame_states: list[RealtimeFrameState]


def _label_color(label: int) -> str:
    if label == -1:
        return "#222222"
    vivid_palette = (
        "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
        "#f781bf", "#a65628", "#ffff33", "#00bcd4", "#d95f02",
        "#1b9e77", "#d01c8b", "#66a61e", "#e6ab02", "#7570b3",
        "#fb9a99", "#fdbf6f", "#cab2d6", "#b2df8a", "#ff1493",
    )
    return vivid_palette[label % len(vivid_palette)]


def plot_clustered_signal_comparison(
    signal: np.ndarray,
    segments: list[WindowSegment],
    labels_by_method: dict[str, list[int]],
    title: str,
    output_path: str | Path | None = None,
    show: bool = True,
    stability: int = 3,
    smooth_labels: bool = False,
    annotations_by_method: dict[str, list[str]] | None = None,
    reference_signal: np.ndarray | None = None,
) -> None:
    method_names = list(labels_by_method)
    fig, axes = plt.subplots(
        max(1, len(method_names)),
        1,
        figsize=(16, max(5.2, 4.4 * max(1, len(method_names)))),
        dpi=140,
        sharex=True,
    )
    if len(method_names) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for ax, method_name in zip(axes, method_names):
        _draw_clustered_signal(
            ax,
            signal,
            segments,
            labels_by_method[method_name],
            f"{title} - {method_name}",
            stability=stability,
            smooth_labels=smooth_labels,
            reference_signal=reference_signal,
        )
        if annotations_by_method and method_name in annotations_by_method:
            _annotate_box(ax, annotations_by_method[method_name])

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if show and _can_show_figures():
        plt.show()
    plt.close(fig)


def _draw_clustered_signal(
    ax: Axes,
    signal: np.ndarray,
    segments: list[WindowSegment],
    labels: list[int],
    title: str,
    view_start: int = 0,
    view_end: int | None = None,
    stability: int = 3,
    smooth_labels: bool = False,
    reference_signal: np.ndarray | None = None,
    show_points: bool | None = None,
) -> None:
    if view_end is None:
        view_end = len(signal)
    view_start = max(0, min(view_start, len(signal)))
    view_end = max(view_start, min(view_end, len(signal)))

    display_labels = _remap_labels_for_display(labels)
    view_signal = signal[view_start:view_end]
    x_axis = np.arange(view_start, view_end)
    if show_points is None:
        show_points = len(signal) <= 200

    if not segments or not display_labels:
        ax.plot(x_axis, view_signal, color="#6b7280", linewidth=1.4, alpha=0.90, label="Unclustered / warm-up", zorder=2)
        if show_points and len(x_axis):
            ax.scatter(
                x_axis,
                view_signal,
                s=12,
                facecolors="white",
                edgecolors="#374151",
                linewidths=0.45,
                alpha=0.90,
                zorder=3,
            )
        if reference_signal is not None:
            reference_view = np.asarray(reference_signal, dtype=float)[view_start:view_end]
            ax.plot(
                x_axis,
                reference_view,
                color="#94a3b8",
                linewidth=1.4,
                alpha=0.45,
                linestyle="-",
                label="Reference signal",
                zorder=1,
            )
        _set_view_limits(ax, view_signal, view_start, view_end)
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncols=2, fontsize=8)
        return

    sample_labels = window_labels_to_sample_labels(
        len(signal),
        segments,
        display_labels,
    )
    view_labels = sample_labels[view_start:view_end]

    seen_labels: set[int] = set()
    for start, end, label in sample_label_runs(view_labels):
        absolute_start = view_start + start
        absolute_end = view_start + end
        color = _label_color(label)
        label_text = "Noise" if label == -1 else f"Cluster {label}"
        legend_label = label_text if label not in seen_labels else None
        seen_labels.add(label)
        ax.plot(
            x_axis[start:end],
            view_signal[start:end],
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=legend_label,
            zorder=3,
        )
        ax.axvspan(absolute_start, absolute_end, color=color, alpha=0.055, linewidth=0, zorder=1)

    ax.plot(x_axis, view_signal, color="#111827", linewidth=0.65, alpha=0.45, zorder=2)
    if show_points and len(x_axis):
        ax.scatter(
            x_axis,
            view_signal,
            s=11,
            facecolors="white",
            edgecolors="#111827",
            linewidths=0.45,
            alpha=0.88,
            zorder=4,
        )
    if reference_signal is not None:
        reference_view = np.asarray(reference_signal, dtype=float)[view_start:view_end]
        ax.plot(
            x_axis,
            reference_view,
            color="#94a3b8",
            linewidth=1.4,
            alpha=0.45,
            linestyle="-",
                label="Reference signal",
            zorder=1,
        )
    _set_view_limits(ax, view_signal, view_start, view_end)

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=2, fontsize=8)


def _remap_labels_for_display(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    next_label = 0
    remapped: list[int] = []
    for label in labels:
        label = int(label)
        if label == -1:
            remapped.append(-1)
            continue
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        remapped.append(mapping[label])
    return remapped


def _set_view_limits(ax: Axes, signal: np.ndarray, view_start: int, view_end: int) -> None:
    ax.set_xlim(view_start, max(view_start + 1, view_end - 1))
    if len(signal) == 0:
        return
    y_min = float(np.min(signal))
    y_max = float(np.max(signal))
    if abs(y_max - y_min) < 1e-9:
        margin = 1.0
    else:
        margin = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - margin, y_max + margin)


def _annotate_box(ax: Axes, lines: list[str]) -> None:
    if not lines:
        return
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="#cbd5e1"),
    )


def _draw_open_window(ax: Axes, frame_state: RealtimeFrameState) -> None:
    if frame_state.open_start is None or frame_state.open_size <= 0:
        return
    open_end = frame_state.visible_end
    ax.axvspan(
        frame_state.open_start,
        max(frame_state.open_start + 1, open_end),
        color="#f59e0b",
        alpha=0.08,
        linewidth=0,
        zorder=0.8,
    )
    if (
        frame_state.open_start == 0
        and frame_state.min_window_size > 0
        and frame_state.open_size < frame_state.min_window_size
    ):
        ax.text(
            0.02,
            0.06,
            f"collecting initial window: {frame_state.open_size}/{frame_state.min_window_size}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.78, edgecolor="#f59e0b"),
        )


class RealtimePlotter:
    def __init__(
        self,
        signal: np.ndarray,
        method_names: list[str],
        view_width: int | None = None,
        stability: int = 3,
        title_prefix: str | None = None,
        show_future_signal: bool = True,
        reference_signal: np.ndarray | None = None,
    ) -> None:
        self.can_show = _can_show_figures()
        self.signal = signal
        self.reference_signal = reference_signal
        self.fig, axes = plt.subplots(len(method_names), 1, figsize=(14, 7), dpi=120, sharex=True)
        if len(method_names) == 1:
            axes = [axes]
        self.axes: list[Axes] = list(axes)
        self.method_names = method_names
        self.view_width = view_width or min(len(signal), 1200)
        self.stability = stability
        self.title_prefix = title_prefix.strip() if title_prefix else ""
        self.show_future_signal = show_future_signal
        self._animation: FuncAnimation | None = None

    def render_frame(
        self,
        frame_state: RealtimeFrameState,
        segments: list[WindowSegment] | dict[str, list[WindowSegment]],
        labels_by_method: dict[str, list[int]],
    ) -> None:
        processed_until = frame_state.visible_end
        for ax, method_name in zip(self.axes, self.method_names):
            ax.clear()
            visible_signal = self.signal[:processed_until]
            if self.show_future_signal:
                full_x_axis = np.arange(len(self.signal))
                ax.plot(
                    full_x_axis,
                    self.signal,
                    color="#9ca3af",
                    linewidth=0.7,
                    alpha=0.35,
                    zorder=0,
                )
            method_segments = segments[method_name] if isinstance(segments, dict) else segments
            processed_segments = (
                frame_state.processed_segments_by_method.get(method_name, frame_state.processed_segments)
                if frame_state.processed_segments_by_method is not None
                else frame_state.processed_segments
            )
            visible_segments = method_segments[:processed_segments]
            labels = labels_by_method[method_name][:processed_segments]
            view_end = min(processed_until, len(self.signal))
            view_start = 0
            _draw_clustered_signal(
                ax,
                visible_signal,
                visible_segments,
                labels,
                self._format_title(method_name),
                view_start=view_start,
                view_end=view_end,
                stability=self.stability,
                smooth_labels=False,
                reference_signal=self.reference_signal,
                show_points=len(self.signal) <= 200,
            )
            method_frame_state = frame_state
            if frame_state.open_start_by_method is not None or frame_state.open_size_by_method is not None:
                method_frame_state = RealtimeFrameState(
                    visible_end=frame_state.visible_end,
                    processed_segments=processed_segments,
                    open_start=(frame_state.open_start_by_method or {}).get(method_name),
                    open_size=(frame_state.open_size_by_method or {}).get(method_name, 0),
                    min_window_size=frame_state.min_window_size,
                )
            _draw_open_window(ax, method_frame_state)
            _set_view_limits(ax, self.signal, 0, len(self.signal))

        self.fig.tight_layout()

    def play(
        self,
        frame_states: list[RealtimeFrameState],
        segments: list[WindowSegment] | dict[str, list[WindowSegment]],
        labels_by_method: dict[str, list[int]],
        delay: float,
    ) -> None:
        if not frame_states:
            return
        if not self.can_show:
            self.render_frame(frame_states[-1], segments, labels_by_method)
            return

        interval_ms = max(1, int(delay * 1000))

        def _update(frame_state: RealtimeFrameState) -> None:
            self.render_frame(frame_state, segments, labels_by_method)

        self._animation = FuncAnimation(
            self.fig,
            _update,
            frames=frame_states,
            interval=interval_ms,
            repeat=False,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=True)

    def save_frames(
        self,
        frame_states: list[RealtimeFrameState],
        segments: list[WindowSegment] | dict[str, list[WindowSegment]],
        labels_by_method: dict[str, list[int]],
        output_dir: str | Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, frame_state in enumerate(frame_states):
            self.render_frame(frame_state, segments, labels_by_method)
            self.fig.savefig(output_dir / f"frame_{index:04d}.png")

    def finish(self) -> None:
        plt.close(self.fig)

    def _format_title(self, method_name: str) -> str:
        if self.title_prefix:
            return f"{self.title_prefix} - {method_name}"
        return method_name


class RealtimeDashboardPlotter:
    def __init__(
        self,
        scenarios: list[RealtimeScenarioState],
        method_names: list[str],
        stability: int = 3,
    ) -> None:
        self.can_show = _can_show_figures()
        self.scenarios = scenarios
        self.method_names = method_names
        self.stability = stability
        rows = max(1, len(scenarios))
        cols = max(1, len(method_names))
        self.fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(16, max(6.0, 2.8 * rows)),
            dpi=120,
            sharex=False,
        )
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])
        self.axes = np.asarray(axes, dtype=object)
        self._animation: FuncAnimation | None = None

    def render_frame(self, frame_index: int) -> None:
        for row_index, scenario in enumerate(self.scenarios):
            if not scenario.frame_states:
                continue
            frame_state = scenario.frame_states[min(frame_index, len(scenario.frame_states) - 1)]
            for col_index, method_name in enumerate(self.method_names):
                ax = self.axes[row_index, col_index]
                ax.clear()
                ax.plot(
                    np.arange(len(scenario.signal)),
                    scenario.signal,
                    color="#9ca3af",
                    linewidth=0.7,
                    alpha=0.35,
                    zorder=0,
                )
                if scenario.reference_signal is not None and len(scenario.reference_signal) == len(scenario.signal):
                    ax.plot(
                        np.arange(len(scenario.reference_signal)),
                        scenario.reference_signal,
                        color="#2563eb",
                        linewidth=1.2,
                        alpha=0.72,
                        zorder=1,
                    )
                visible_signal = scenario.signal[: frame_state.visible_end]
                visible_segments = scenario.segments[: frame_state.processed_segments]
                labels = scenario.labels_by_method[method_name][: frame_state.processed_segments]
                _draw_clustered_signal(
                    ax,
                    visible_signal,
                    visible_segments,
                    labels,
                    f"{scenario.name} - {method_name}",
                    view_start=0,
                    view_end=min(frame_state.visible_end, len(scenario.signal)),
                    stability=self.stability,
                    smooth_labels=False,
                    show_points=len(scenario.signal) <= 200,
                )
                _draw_open_window(ax, frame_state)
                _set_view_limits(ax, scenario.signal, 0, len(scenario.signal))
                ax.tick_params(labelsize=8)

        self.fig.tight_layout()

    def play(self, frame_count: int, delay: float) -> None:
        if frame_count <= 0:
            return
        if not self.can_show:
            self.render_frame(frame_count - 1)
            return

        interval_ms = max(1, int(delay * 1000))

        def _update(frame_index: int) -> None:
            self.render_frame(frame_index)

        self._animation = FuncAnimation(
            self.fig,
            _update,
            frames=range(frame_count),
            interval=interval_ms,
            repeat=False,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=True)

    def finish(self) -> None:
        plt.close(self.fig)


def _can_show_figures() -> bool:
    return plt.get_backend().lower() != "agg"


