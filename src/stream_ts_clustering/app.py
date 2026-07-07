from __future__ import annotations

import contextlib
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from tkinter import END, BOTH, LEFT, RIGHT, X, Y, TOP, BOTTOM, Canvas, Scrollbar, StringVar, BooleanVar, IntVar, DoubleVar, Tk, TclError
from tkinter import messagebox, scrolledtext, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .data_loader import (
    StreamSignal,
    list_synthetic_dataset_specs,
)
from .evaluation import cluster_quality_metrics
from .experiments import (
    BoundaryMetrics,
    ModelDrivenExpandingWindow,
    RealtimeFrameState,
    _adaptive_max_window_size,
    _aggregate_boundary_metrics,
    _aggregate_cluster_quality_metrics,
    _compute_boundary_metrics_by_method,
    _format_run_report,
    _default_boundary_tolerance_from_segments,
    _load_stream,
    _realtime_controller_patience,
    create_default_clusterers,
    default_clusterer_params,
    run_synthetic_boundary_metrics_benchmark,
    run_synthetic_realtime_metrics_benchmark,
    remap_labels,
)
from .postprocessing import window_labels_to_sample_labels
from .visualization import RealtimePlotter


@dataclass(frozen=True)
class DatasetChoice:
    label: str
    dataset_path: str
    category: str
    kind: str
    default_window_size: int
    default_series_count: int
    default_step: int
    default_max_points: int | None = None
    description: str = ""
    balanced: bool = False


@dataclass(frozen=True)
class RealtimeSession:
    dataset_label: str
    stream_signal: np.ndarray
    source_labels: np.ndarray
    requested_window_size: int
    effective_window_size: int
    reference_signal: np.ndarray | None
    method_names: list[str]
    segments_by_method: dict[str, list]
    online_labels_by_method: dict[str, list[int]]
    evaluation_labels_by_method: dict[str, list[int]]
    boundary_metrics_by_method: dict[str, BoundaryMetrics | None]
    frame_states: list[RealtimeFrameState]
    title_prefix: str



def build_dataset_catalog(project_root: Path) -> list[DatasetChoice]:
    choices: list[DatasetChoice] = []

    for spec in list_synthetic_dataset_specs():
        choices.append(
            DatasetChoice(
                label=f"Synthetic | {spec.name}",
                dataset_path=f"synthetic:{spec.key}",
                category="Synthetic",
                kind="synthetic",
                default_window_size=10,
                default_series_count=1,
                default_step=1,
                description="",
            )
        )

    real_streams = [
        ("NAB traffic occupancy", str(project_root / "datasets" / "nab_traffic_occupancy_1000_with_hint.csv"), 54, 1, 1000, "Small regime dataset for traffic occupancy."),
        ("SKAB valve1 flow", str(project_root / "datasets" / "skab_valve1_flow.csv"), 54, 1, 500, "Operating regime dataset for pump flow."),
        ("SKAB rotor imbalance vibration", str(project_root / "datasets" / "skab_rotor_imbalance_vibration.csv"), 54, 1, 500, "Operating regime dataset for rotor vibration."),
        ("SKAB hot water temperature", str(project_root / "datasets" / "skab_hot_water_temperature.csv"), 54, 1, 500, "Operating regime dataset for hot water temperature."),
        ("UCI Occupancy light", str(project_root / "datasets" / "uci_occupancy_light.csv"), 54, 1, 500, "Operating regime dataset for occupancy light."),
        ("NAB EC2 CPU", str(project_root / "datasets" / "nab_ec2_cpu.csv"), 54, 1, 500, "Operating regime dataset for EC2 CPU."),
    ]
    for label, dataset_path, window_size, series_count, max_points, description in real_streams:
        choices.append(
            DatasetChoice(
                label=label,
                dataset_path=dataset_path,
                category="Real datasets",
                kind="labeled",
                default_window_size=window_size,
                default_series_count=series_count,
                default_step=1,
                default_max_points=max_points,
                description=description,
            )
        )

    return choices


class ClusterProjectApp:
    def __init__(self, root: Tk, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.root.title("Cluster Project")
        self.root.geometry("1560x980")
        self._collapsed_realtime_width = 430
        self._expanded_realtime_width = 820
        self._collapsed_window_geometry = "1560x980"
        self._expanded_window_geometry = "1920x980"
        self._advanced_expanded = False
        self._manual_pipeline_defaults_active = False
        self._window_size_manual_override = False
        self._window_size_internal_update = False
        self.settings_path = self.project_root / ".clusterproject_gui_settings.json"

        self.dataset_choices = build_dataset_catalog(project_root)
        self.synthetic_dataset_choices = [choice for choice in self.dataset_choices if choice.kind == "synthetic"]
        self.choice_by_label = {choice.label: choice for choice in self.dataset_choices}
        self.method_names = ["CluStream", "DenStream"]
        self._queue: Queue[tuple[str, object]] = Queue()
        self._realtime_after_id: str | None = None
        self._realtime_run_id = 0
        self._current_canvas: FigureCanvasTkAgg | None = None
        self._current_toolbar: NavigationToolbar2Tk | None = None
        self._current_plotter: RealtimePlotter | None = None
        self._current_session: RealtimeSession | None = None

        self.dataset_var = StringVar(value=self.dataset_choices[0].label)
        self.series_count_var = IntVar(value=self.dataset_choices[0].default_series_count)
        self.window_size_var = IntVar(value=self.dataset_choices[0].default_window_size)
        self.max_points_var = IntVar(value=self.dataset_choices[0].default_max_points or 0)
        self.delay_var = DoubleVar(value=0.0)
        self.window_mode_var = StringVar(value="expanding")
        self.feature_profile_var = StringVar(value="basic")
        self.standardize_var = BooleanVar(value=True)
        self.clustream_var = BooleanVar(value=True)
        self.denstream_var = BooleanVar(value=True)
        self.metric_mode_var = StringVar(value="boundary")
        self.metric_runs_var = IntVar(value=5)
        self.metric_standardize_var = BooleanVar(value=True)
        self.metric_dataset_var = StringVar(value=self.synthetic_dataset_choices[0].label)
        self.metric_output_var = StringVar(value="Current synthetic metrics will appear below.")
        self.metric_window_mode_var = StringVar(value="expanding")
        self.metric_feature_profile_var = StringVar(value="basic")
        self.metric_window_size_var = IntVar(value=10)
        self.advanced_visible_var = BooleanVar(value=False)
        self.use_custom_cluster_params_var = BooleanVar(value=False)
        self.clustream_max_micro_clusters_var = IntVar(value=36)
        self.clustream_radius_factor_var = DoubleVar(value=1.24)
        self.clustream_min_radius_var = DoubleVar(value=0.82)
        self.clustream_stale_after_var = IntVar(value=90)
        self.clustream_tentative_distance_var = DoubleVar(value=0.00)
        self.clustream_emit_micro_labels_var = BooleanVar(value=False)
        self.denstream_epsilon_var = DoubleVar(value=0.72)
        self.denstream_beta_var = DoubleVar(value=0.35)
        self.denstream_mu_var = DoubleVar(value=1.80)
        self.denstream_lambd_var = DoubleVar(value=0.0015)
        self.denstream_prune_interval_var = IntVar(value=8)
        self.denstream_radius_tolerance_var = DoubleVar(value=1.16)
        self.denstream_emit_noise_var = BooleanVar(value=True)
        self.denstream_grace_period_var = IntVar(value=2)
        self.denstream_outlier_radius_tolerance_var = DoubleVar(value=1.16)
        self.denstream_outlier_promote_weight_var = DoubleVar(value=1.10)
        self.denstream_outlier_promote_age_var = IntVar(value=1)
        self.denstream_min_points_before_noise_var = IntVar(value=3)
        self.denstream_noise_score_threshold_var = DoubleVar(value=3.40)
        self.status_var = StringVar(value="Ready")

        self._load_gui_settings()
        self._build_ui()
        self.root.after(100, self._poll_queue)
        self.dataset_var.trace_add("write", lambda *_: self._apply_dataset_defaults())
        self.window_size_var.trace_add("write", lambda *_: self._mark_window_size_manual_override())
        self.metric_dataset_var.trace_add("write", lambda *_: self._apply_metric_dataset_defaults())
        self.use_custom_cluster_params_var.trace_add("write", lambda *_: self._sync_manual_pipeline_defaults())
        self._apply_dataset_defaults()
        self._apply_metric_dataset_defaults()

    def _build_ui(self) -> None:
        self._configure_style()
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=BOTH, expand=True)

        self.realtime_tab = ttk.Frame(notebook, padding=12, style="App.TFrame")
        self.metrics_tab = ttk.Frame(notebook, padding=12, style="App.TFrame")
        self.datasets_tab = ttk.Frame(notebook, padding=12, style="App.TFrame")
        self.logs_tab = ttk.Frame(notebook, padding=12, style="App.TFrame")

        notebook.add(self.realtime_tab, text="Старт")
        notebook.add(self.metrics_tab, text="Метрики")
        notebook.add(self.datasets_tab, text="Наборы данных")
        notebook.add(self.logs_tab, text="Логи")

        self._build_realtime_tab(self.realtime_tab)
        self._build_metrics_tab(self.metrics_tab)
        self._build_datasets_tab(self.datasets_tab)
        self._build_logs_tab(self.logs_tab)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", style="Status.TLabel")
        status_bar.pack(fill=X, side="bottom")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#f4f1ea"
        panel = "#ffffff"
        accent = "#264653"
        muted = "#6b7280"
        self.root.configure(background=bg)
        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabelframe", background=bg, padding=8)
        style.configure("TLabelframe.Label", background=bg, foreground=accent, font=("Segoe UI Semibold", 10))
        style.configure("TLabel", background=bg, foreground="#1f2937")
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure("Status.TLabel", background="#e8dfcf", foreground=accent, padding=(10, 4))
        style.configure("TButton", padding=(10, 6))
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI Semibold", 10))
        style.map("TNotebook.Tab", background=[("selected", panel), ("active", "#ede7da")])
        style.configure("Card.TFrame", background=panel, relief="flat")
        style.configure("Accent.TButton", background=accent, foreground="white")
        style.map("Accent.TButton", background=[("active", "#1f3f4c"), ("pressed", "#1b3340")])
        self._panel_bg = panel
        self._page_bg = bg
        self._accent_color = accent
        self._muted_color = muted

    def _build_realtime_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, style="App.TFrame")
        wrapper.pack(fill=BOTH, expand=True)

        controls_shell, controls = self._make_scrollable_panel(wrapper, width=self._collapsed_realtime_width, height=860)
        controls_shell.pack(side=LEFT, fill=Y, padx=(0, 10))
        self.realtime_controls_shell = controls_shell

        plot_area = ttk.Frame(wrapper, style="Card.TFrame")
        plot_area.pack(side=RIGHT, fill=BOTH, expand=True)
        self.realtime_plot_host = plot_area

        self._build_dataset_selector(controls)
        self._build_algorithm_selector(controls)
        self._build_realtime_parameters(controls)

        button_row = ttk.Frame(controls, style="Card.TFrame")
        button_row.pack(fill=X, pady=(12, 0))
        ttk.Button(button_row, text="Старт", style="Accent.TButton", command=self.start_realtime_run).pack(fill=X, pady=(0, 6))
        ttk.Button(button_row, text="Стоп", command=self.stop_realtime_run).pack(fill=X)

        self.realtime_log = scrolledtext.ScrolledText(controls, width=44, height=20, wrap="word")
        self._style_text_widget(self.realtime_log)
        self.realtime_log.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.realtime_log.insert(END, "Вывод будет показан здесь.\n")

    def _build_metrics_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, style="App.TFrame")
        wrapper.pack(fill=BOTH, expand=True)

        controls_shell, controls = self._make_scrollable_panel(wrapper, width=1380, height=220, horizontal=False)
        controls_shell.pack(side=TOP, fill=X)

        dataset_row = ttk.Frame(controls)
        dataset_row.pack(fill=X)
        ttk.Label(dataset_row, text="Синтетический набор").pack(side=LEFT)
        dataset_combo = ttk.Combobox(
            dataset_row,
            textvariable=self.metric_dataset_var,
            values=[choice.label for choice in self.synthetic_dataset_choices],
            state="readonly",
            width=60,
        )
        dataset_combo.pack(side=LEFT, padx=8)
        ttk.Label(controls, textvariable=self.metric_output_var, style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

        algo_row = ttk.Frame(controls)
        algo_row.pack(fill=X, pady=(6, 0))
        ttk.Label(algo_row, text="Алгоритмы").pack(side=LEFT)
        ttk.Checkbutton(algo_row, text="CluStream", variable=self.clustream_var).pack(side=LEFT, padx=8)
        ttk.Checkbutton(algo_row, text="DenStream", variable=self.denstream_var).pack(side=LEFT)

        options_row = ttk.Frame(controls)
        options_row.pack(fill=X, pady=(6, 0))
        ttk.Label(options_row, text="Режим метрик").pack(side=LEFT)
        ttk.Combobox(
            options_row,
            textvariable=self.metric_mode_var,
            values=["boundary", "quality", "full"],
            state="readonly",
            width=14,
        ).pack(side=LEFT, padx=8)
        ttk.Label(options_row, text="Прогонов").pack(side=LEFT, padx=(10, 0))
        ttk.Spinbox(options_row, from_=1, to=100, textvariable=self.metric_runs_var, width=8).pack(side=LEFT, padx=8)
        ttk.Label(options_row, text="Окно").pack(side=LEFT, padx=(10, 0))
        ttk.Spinbox(options_row, from_=2, to=500, textvariable=self.metric_window_size_var, width=8).pack(side=LEFT, padx=8)

        actions = ttk.Frame(controls)
        actions.pack(fill=X, pady=(8, 0))
        ttk.Button(actions, text="Запустить метрики", style="Accent.TButton", command=self.start_metrics_run).pack(side=LEFT)
        suite_box = ttk.LabelFrame(controls, text="Синтетические сценарии")
        suite_box.pack(fill=X, pady=(10, 0))
        ttk.Button(suite_box, text="Граница режимов", command=self.run_synthetic_boundary_suite).pack(fill=X, pady=2)
        ttk.Button(suite_box, text="Метрики потока", command=self.run_synthetic_realtime_metrics_suite).pack(fill=X, pady=2)

        self.metrics_output = scrolledtext.ScrolledText(wrapper, wrap="word")
        self._style_text_widget(self.metrics_output)
        self.metrics_output.pack(fill=BOTH, expand=True, pady=(10, 0))
        self.metrics_output.insert(END, "Вывод будет показан здесь.\n")

    def _build_datasets_tab(self, parent: ttk.Frame) -> None:
        columns = ("dataset", "category", "kind", "window", "series", "points", "description")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=24)
        tree.heading("dataset", text="Dataset")
        tree.column("dataset", width=220, anchor="w")
        tree.heading("category", text="Category")
        tree.heading("kind", text="Kind")
        tree.heading("window", text="Window")
        tree.heading("series", text="Series")
        tree.heading("points", text="Max points")
        tree.heading("description", text="Description")
        tree.column("category", width=130, anchor="w")
        tree.column("kind", width=110, anchor="w")
        tree.column("window", width=80, anchor="center")
        tree.column("series", width=70, anchor="center")
        tree.column("points", width=90, anchor="center")
        tree.column("description", width=400, anchor="w")
        tree.pack(fill=BOTH, expand=True)

        for choice in self.dataset_choices:
            tree.insert(
                "",
                END,
                values=(
                    choice.label,
                    choice.category,
                    choice.kind,
                    choice.default_window_size,
                    choice.default_series_count,
                    choice.default_max_points or "",
                    choice.description,
                ),
            )

        note = ttk.Label(
            parent,
            text="Синтетические наборы генерируются на лету. Реальные потоки загружаются из файлов проекта.",
            anchor="w",
        )
        note.pack(fill=X, pady=(10, 0))

    def _build_logs_tab(self, parent: ttk.Frame) -> None:
        self.global_log = scrolledtext.ScrolledText(parent, wrap="word")
        self._style_text_widget(self.global_log)
        self.global_log.pack(fill=BOTH, expand=True)
        self.global_log.insert(END, "Журнал приложения.\n")

    def _build_dataset_selector(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Набор данных")
        box.pack(fill=X)

        ttk.Label(box, text="Источник").pack(anchor="w")
        combo = ttk.Combobox(
            box,
            textvariable=self.dataset_var,
            values=[choice.label for choice in self.dataset_choices],
            state="readonly",
            width=46,
        )
        combo.pack(fill=X, pady=(2, 6))

        self.dataset_summary = ttk.Label(box, text="", justify="left", wraplength=300)
        self.dataset_summary.pack(fill=X)

    def _build_algorithm_selector(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Алгоритмы")
        box.pack(fill=X, pady=(10, 0))
        ttk.Checkbutton(box, text="CluStream", variable=self.clustream_var).pack(anchor="w")
        ttk.Checkbutton(box, text="DenStream", variable=self.denstream_var).pack(anchor="w")

    def _build_realtime_parameters(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Настройки потока")
        box.pack(fill=X, pady=(10, 0))

        self._add_labeled_spinbox(box, "Начальный размер окна", self.window_size_var, 2, 500)
        self._add_labeled_spinbox(box, "Макс. точек", self.max_points_var, 0, 50000)

        ttk.Button(box, text="Показать / скрыть расширенные параметры", command=self._toggle_advanced_panel).pack(fill=X, pady=(8, 0))
        self.advanced_panel = ttk.LabelFrame(box, text="Расширенные параметры кластеризации")
        self.advanced_panel.pack(fill=X, pady=(8, 0))
        self._build_advanced_panel(self.advanced_panel)
        self.advanced_panel.pack_forget()

    def _add_labeled_spinbox(self, parent: ttk.Frame, label: str, variable: IntVar, minimum: int, maximum: int) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=2)
        ttk.Label(row, text=label).pack(side=LEFT)
        ttk.Spinbox(row, from_=minimum, to=maximum, textvariable=variable, width=10).pack(side=RIGHT)

    def _add_labeled_double(self, parent: ttk.Frame, label: str, variable: DoubleVar, minimum: float, maximum: float, increment: float = 0.01) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=2)
        ttk.Label(row, text=label).pack(side=LEFT)
        ttk.Spinbox(row, from_=minimum, to=maximum, increment=increment, textvariable=variable, width=12).pack(side=RIGHT)

    def _build_advanced_panel(self, parent: ttk.Frame) -> None:
        shell, body = self._make_scrollable_panel(parent, width=760, height=500)
        shell.pack(fill=BOTH, expand=True)

        ttk.Checkbutton(
            body,
            text="Применять свои параметры кластеризации",
            variable=self.use_custom_cluster_params_var,
        ).pack(anchor="w", pady=(0, 8))

        actions = ttk.Frame(body)
        actions.pack(fill=X, pady=(0, 8))
        ttk.Button(actions, text="Применить как постоянные настройки", style="Accent.TButton", command=self.apply_persistent_advanced_settings).pack(fill=X)
        ttk.Label(
            actions,
            text="Сохраняет текущие расширенные параметры для будущих запусков.",
            style="Muted.TLabel",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        top = ttk.Frame(body)
        top.pack(fill=X, pady=(0, 8))
        pipeline_box = ttk.LabelFrame(top, text="Параметры запуска")
        pipeline_box.pack(fill=X)
        self._add_labeled_spinbox(pipeline_box, "Начальный размер окна", self.window_size_var, 2, 500)
        self._add_labeled_spinbox(pipeline_box, "Макс. точек", self.max_points_var, 0, 50000)

        left = ttk.Frame(body)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        right = ttk.Frame(body)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(8, 0))

        clustream_box = ttk.LabelFrame(left, text="CluStream")
        clustream_box.pack(fill=X)
        self._add_labeled_spinbox(clustream_box, "Max microclusters", self.clustream_max_micro_clusters_var, 2, 200)
        self._add_labeled_double(clustream_box, "Radius factor", self.clustream_radius_factor_var, 0.1, 5.0, 0.05)
        self._add_labeled_double(clustream_box, "Min radius", self.clustream_min_radius_var, 0.0, 5.0, 0.05)
        self._add_labeled_spinbox(clustream_box, "Stale after", self.clustream_stale_after_var, 1, 1000)
        self._add_labeled_double(clustream_box, "Tentative distance", self.clustream_tentative_distance_var, 0.0, 10.0, 0.05)
        ttk.Checkbutton(clustream_box, text="Emit micro labels", variable=self.clustream_emit_micro_labels_var).pack(anchor="w", pady=(4, 0))

        denstream_box = ttk.LabelFrame(right, text="DenStream")
        denstream_box.pack(fill=X)
        self._add_labeled_double(denstream_box, "Epsilon", self.denstream_epsilon_var, 0.1, 10.0, 0.05)
        self._add_labeled_double(denstream_box, "Beta", self.denstream_beta_var, 0.01, 1.0, 0.01)
        self._add_labeled_double(denstream_box, "Mu", self.denstream_mu_var, 0.1, 20.0, 0.1)
        self._add_labeled_double(denstream_box, "Lambda", self.denstream_lambd_var, 0.0001, 0.1, 0.0005)
        self._add_labeled_spinbox(denstream_box, "Prune interval", self.denstream_prune_interval_var, 1, 1000)
        self._add_labeled_double(denstream_box, "Radius tolerance", self.denstream_radius_tolerance_var, 1.0, 5.0, 0.01)
        ttk.Checkbutton(denstream_box, text="Emit noise", variable=self.denstream_emit_noise_var).pack(anchor="w", pady=(4, 0))
        self._add_labeled_spinbox(denstream_box, "Grace period", self.denstream_grace_period_var, 0, 100)
        self._add_labeled_double(denstream_box, "Outlier radius tolerance", self.denstream_outlier_radius_tolerance_var, 1.0, 5.0, 0.01)
        self._add_labeled_double(denstream_box, "Outlier promote weight", self.denstream_outlier_promote_weight_var, 0.0, 20.0, 0.1)
        self._add_labeled_spinbox(denstream_box, "Outlier promote age", self.denstream_outlier_promote_age_var, 0, 100)
        self._add_labeled_spinbox(denstream_box, "Min points before noise", self.denstream_min_points_before_noise_var, 0, 100)
        self._add_labeled_double(denstream_box, "Noise score threshold", self.denstream_noise_score_threshold_var, 0.5, 10.0, 0.05)

    def _toggle_advanced_panel(self) -> None:
        if self.advanced_panel.winfo_ismapped():
            self.advanced_panel.pack_forget()
            self._advanced_expanded = False
            self.realtime_controls_shell.configure(width=self._collapsed_realtime_width)
            self.root.geometry(self._collapsed_window_geometry)
        else:
            self.advanced_panel.pack(fill=X, pady=(8, 0))
            self._advanced_expanded = True
            self.realtime_controls_shell.configure(width=self._expanded_realtime_width)
            self.root.geometry(self._expanded_window_geometry)
        self.root.update_idletasks()

    def _sync_manual_pipeline_defaults(self) -> None:
        self._manual_pipeline_defaults_active = bool(self.use_custom_cluster_params_var.get())
        if not self._manual_pipeline_defaults_active:
            self._refresh_advanced_defaults_for_current_profile()

    def _refresh_advanced_defaults_for_current_profile(self) -> None:
        if bool(self.use_custom_cluster_params_var.get()):
            return
        clustream, denstream = default_clusterer_params()
        self._set_cluster_param_vars(clustream, denstream)

    def _set_cluster_param_vars(self, clustream: dict[str, object], denstream: dict[str, object]) -> None:
        self.clustream_max_micro_clusters_var.set(int(clustream.get("max_micro_clusters", self.clustream_max_micro_clusters_var.get())))
        self.clustream_radius_factor_var.set(float(clustream.get("radius_factor", self.clustream_radius_factor_var.get())))
        self.clustream_min_radius_var.set(float(clustream.get("min_radius", self.clustream_min_radius_var.get())))
        self.clustream_stale_after_var.set(int(clustream.get("stale_after", self.clustream_stale_after_var.get())))
        self.clustream_tentative_distance_var.set(float(clustream.get("tentative_distance", self.clustream_tentative_distance_var.get())))
        self.clustream_emit_micro_labels_var.set(bool(clustream.get("emit_micro_labels", self.clustream_emit_micro_labels_var.get())))

        self.denstream_epsilon_var.set(float(denstream.get("epsilon", self.denstream_epsilon_var.get())))
        self.denstream_beta_var.set(float(denstream.get("beta", self.denstream_beta_var.get())))
        self.denstream_mu_var.set(float(denstream.get("mu", self.denstream_mu_var.get())))
        self.denstream_lambd_var.set(float(denstream.get("lambd", self.denstream_lambd_var.get())))
        self.denstream_prune_interval_var.set(int(denstream.get("prune_interval", self.denstream_prune_interval_var.get())))
        self.denstream_radius_tolerance_var.set(float(denstream.get("radius_tolerance", self.denstream_radius_tolerance_var.get())))
        self.denstream_emit_noise_var.set(bool(denstream.get("emit_noise", self.denstream_emit_noise_var.get())))
        self.denstream_grace_period_var.set(int(denstream.get("grace_period", self.denstream_grace_period_var.get())))
        self.denstream_outlier_radius_tolerance_var.set(float(denstream.get("outlier_radius_tolerance", self.denstream_outlier_radius_tolerance_var.get())))
        self.denstream_outlier_promote_weight_var.set(float(denstream.get("outlier_promote_weight", self.denstream_outlier_promote_weight_var.get())))
        self.denstream_outlier_promote_age_var.set(int(denstream.get("outlier_promote_age", self.denstream_outlier_promote_age_var.get())))
        self.denstream_min_points_before_noise_var.set(int(denstream.get("min_points_before_noise", self.denstream_min_points_before_noise_var.get())))
        self.denstream_noise_score_threshold_var.set(float(denstream.get("noise_score_threshold", self.denstream_noise_score_threshold_var.get())))

    def apply_persistent_advanced_settings(self) -> None:
        first_confirm = messagebox.askyesno(
            "Применить расширенные настройки",
            "Применить текущие расширенные настройки как постоянные настройки приложения?",
        )
        if not first_confirm:
            return
        second_confirm = messagebox.askyesno(
            "Подтверждение сохранения",
            "Эти значения будут сохранены на диск и будут использоваться при следующих запусках. Продолжить?",
        )
        if not second_confirm:
            return
        self.use_custom_cluster_params_var.set(True)
        self._manual_pipeline_defaults_active = True
        self._save_gui_settings()
        self.status_var.set("Расширенные настройки сохранены")
        self._append_text(self.global_log, "[Настройки] Расширенные параметры сохранены на диск.")

    def _apply_dataset_defaults(self) -> None:
        choice = self.choice_by_label.get(self.dataset_var.get())
        if choice is None:
            return
        self.dataset_summary.configure(text=f"{choice.category} / {choice.kind}\n{choice.description}")
        if not self._window_size_manual_override:
            self._set_window_size_programmatically(choice.default_window_size)
        if not self._manual_pipeline_defaults_active:
            self.series_count_var.set(choice.default_series_count)
            self.max_points_var.set(choice.default_max_points or 0)
        self._refresh_advanced_defaults_for_current_profile()

    def _apply_metric_dataset_defaults(self) -> None:
        choice = self.choice_by_label.get(self.metric_dataset_var.get())
        if choice is None:
            return
        self.metric_output_var.set(f"Выбран синтетический набор: {choice.label}")
        self.metric_window_size_var.set(10)

    def _set_window_size_programmatically(self, value: int) -> None:
        self._window_size_internal_update = True
        try:
            self.window_size_var.set(max(2, int(value)))
        finally:
            self._window_size_internal_update = False

    def _mark_window_size_manual_override(self) -> None:
        if self._window_size_internal_update:
            return
        self._window_size_manual_override = True

    def _selected_methods(self) -> list[str]:
        methods: list[str] = []
        if self.clustream_var.get():
            methods.append("CluStream")
        if self.denstream_var.get():
            methods.append("DenStream")
        return methods

    def _selected_cluster_params(self) -> tuple[dict[str, object], dict[str, object]]:
        clustream_params = {
            "max_micro_clusters": int(self.clustream_max_micro_clusters_var.get()),
            "radius_factor": float(self.clustream_radius_factor_var.get()),
            "min_radius": float(self.clustream_min_radius_var.get()),
            "stale_after": int(self.clustream_stale_after_var.get()),
            "tentative_distance": float(self.clustream_tentative_distance_var.get()),
            "emit_micro_labels": bool(self.clustream_emit_micro_labels_var.get()),
        }
        denstream_params = {
            "epsilon": float(self.denstream_epsilon_var.get()),
            "beta": float(self.denstream_beta_var.get()),
            "mu": float(self.denstream_mu_var.get()),
            "lambd": float(self.denstream_lambd_var.get()),
            "prune_interval": int(self.denstream_prune_interval_var.get()),
            "radius_tolerance": float(self.denstream_radius_tolerance_var.get()),
            "emit_noise": bool(self.denstream_emit_noise_var.get()),
            "grace_period": int(self.denstream_grace_period_var.get()),
            "outlier_radius_tolerance": float(self.denstream_outlier_radius_tolerance_var.get()),
            "outlier_promote_weight": float(self.denstream_outlier_promote_weight_var.get()),
            "outlier_promote_age": int(self.denstream_outlier_promote_age_var.get()),
            "min_points_before_noise": int(self.denstream_min_points_before_noise_var.get()),
            "noise_score_threshold": float(self.denstream_noise_score_threshold_var.get()),
        }
        return clustream_params, denstream_params

    def _cluster_param_overrides(self) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        if not bool(self.use_custom_cluster_params_var.get()):
            return None, None
        return self._selected_cluster_params()

    def start_realtime_run(self) -> None:
        methods = self._selected_methods()
        if not methods:
            messagebox.showerror("Не выбран алгоритм", "Выберите хотя бы один алгоритм.")
            return

        self.stop_realtime_run()
        self._realtime_run_id += 1
        run_id = self._realtime_run_id
        choice = self.choice_by_label.get(self.dataset_var.get())
        if choice is None:
            messagebox.showerror("Ошибка набора данных", "Сначала выберите набор данных.")
            return
        clustream_params, denstream_params = self._cluster_param_overrides()

        payload = {
            "run_id": run_id,
            "choice": choice,
            "methods": methods,
            "window_size": max(2, int(self.window_size_var.get())),
            "series_count": max(1, int(self.series_count_var.get())),
            "max_points": int(self.max_points_var.get()) or None,
            "window_mode": self.window_mode_var.get(),
            "feature_profile": self.feature_profile_var.get(),
            "standardize": bool(self.standardize_var.get()),
            "delay": self._parse_delay(),
            "clustream_params": clustream_params or {},
            "denstream_params": denstream_params or {},
        }
        self._append_realtime_log(f"Старт потока для {choice.label}; алгоритмы: {', '.join(methods)}")
        self.status_var.set(f"Подготовка потока: {choice.label}")
        threading.Thread(target=self._prepare_realtime_session, args=(payload,), daemon=True).start()

    def stop_realtime_run(self) -> None:
        if self._realtime_after_id is not None:
            try:
                self.root.after_cancel(self._realtime_after_id)
            except Exception:
                pass
            self._realtime_after_id = None
        self._current_plotter = None
        self._current_session = None

    def _prepare_realtime_session(self, payload: dict[str, object]) -> None:
        try:
            run_id = int(payload["run_id"])
            choice = payload["choice"]
            assert isinstance(choice, DatasetChoice)

            stream = _load_stream(
                choice.dataset_path,
                series_index=0,
                series_count=int(payload["series_count"]),
                balanced=choice.balanced,
                max_points=payload["max_points"],
            )
            session = self._build_realtime_session(
                run_id=run_id,
                choice=choice,
                stream=stream,
                methods=list(payload["methods"]),
                window_size=int(payload["window_size"]),
                window_mode=str(payload.get("window_mode", "expanding")),
                feature_profile=str(payload.get("feature_profile", "basic")),
                standardize=bool(payload.get("standardize", True)),
                clustream_params=dict(payload["clustream_params"]),
                denstream_params=dict(payload["denstream_params"]),
            )
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"Ошибка подготовки потока: {exc}"))
            return

        self._queue.put(("realtime-ready", (run_id, session, float(payload["delay"]))))

    def _build_realtime_session(
        self,
        run_id: int,
        choice: DatasetChoice,
        stream: StreamSignal,
        methods: list[str],
        window_size: int,
        window_mode: str,
        feature_profile: str,
        standardize: bool,
        clustream_params: dict[str, object],
        denstream_params: dict[str, object],
    ) -> RealtimeSession:
        if window_mode != "expanding":
            raise ValueError("Поддерживается только режим окна expanding")
        if feature_profile != "basic":
            raise ValueError("Поддерживается только профиль признаков basic")
        stream_signal = np.asarray(stream.signal, dtype=float)
        reference_signal = stream.reference_signal
        source_labels = np.asarray(stream.source_labels, dtype=int)
        signal_length = len(stream_signal)
        effective_window_size = max(2, int(window_size))
        unlabeled_real_stream = bool(np.unique(source_labels).size <= 1 and reference_signal is None)
        clusterers = create_default_clusterers(
            selected_methods=methods,
            clustream_params=clustream_params,
            denstream_params=denstream_params,
        )
        controllers = {
            clusterer.name: ModelDrivenExpandingWindow(
                clusterer=clusterer,
                min_window_size=effective_window_size,
                feature_profile=feature_profile,
                standardize=standardize,
                max_window_size=_adaptive_max_window_size(signal_length, effective_window_size),
                patience=_realtime_controller_patience(clusterer.name, signal_length, unlabeled_real_stream),
                episode_reuse_distance=(0.20 if unlabeled_real_stream else 3.50),
            )
            for clusterer in clusterers
        }

        frame_states: list[RealtimeFrameState] = []
        for sample_index, value in enumerate(stream_signal):
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
            controller.finish(len(stream_signal))

        frame_states.append(
            RealtimeFrameState(
                visible_end=len(stream_signal),
                processed_segments=0,
                processed_segments_by_method={name: len(controller.segments) for name, controller in controllers.items()},
                open_start_by_method={name: None for name in controllers},
                open_size_by_method={name: 0 for name in controllers},
                min_window_size=effective_window_size,
            )
        )

        segments_by_method = {name: controller.segments for name, controller in controllers.items()}
        online_labels_by_method = {name: controller.labels for name, controller in controllers.items()}
        evaluation_labels_by_method = {
            name: remap_labels(labels) for name, labels in online_labels_by_method.items()
        }
        boundary_metrics_by_method = {
            name: _compute_boundary_metrics_by_method(
                source_labels,
                segments_by_method[name],
                {name: evaluation_labels_by_method[name]},
                boundary_tolerance=_default_boundary_tolerance_from_segments(segments_by_method[name]),
            ).get(name)
            for name in evaluation_labels_by_method
        }
        return RealtimeSession(
            dataset_label=choice.label,
            stream_signal=stream_signal,
            source_labels=source_labels,
            requested_window_size=window_size,
            effective_window_size=effective_window_size,
            reference_signal=reference_signal,
            method_names=list(controllers),
            segments_by_method=segments_by_method,
            online_labels_by_method=online_labels_by_method,
            evaluation_labels_by_method=evaluation_labels_by_method,
            boundary_metrics_by_method=boundary_metrics_by_method,
            frame_states=frame_states,
            title_prefix=choice.label,
        )

    def _start_realtime_animation(self, session: RealtimeSession, delay: float) -> None:
        self._clear_plot_host(self.realtime_plot_host)
        self._current_session = session
        self._current_plotter = RealtimePlotter(
            session.stream_signal,
            session.method_names,
            view_width=len(session.stream_signal),
            stability=1,
            title_prefix=session.title_prefix,
            show_future_signal=False,
            reference_signal=session.reference_signal,
        )
        canvas = FigureCanvasTkAgg(self._current_plotter.fig, master=self.realtime_plot_host)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(background=self._panel_bg, highlightthickness=0)
        canvas_widget.pack(side=TOP, fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.realtime_plot_host)
        toolbar.update()
        toolbar.pack(side=BOTTOM, fill=X)
        canvas.draw()
        canvas_widget.update_idletasks()
        self.realtime_plot_host.update_idletasks()
        self.root.update_idletasks()
        self._current_canvas = canvas
        self._current_toolbar = toolbar
        self.status_var.set(f"Поток: {session.dataset_label}")
        self._append_realtime_log(
            f"Сессия подготовлена: кадров {len(session.frame_states)}, алгоритмы: {', '.join(session.method_names)}"
        )
        if not canvas_widget.winfo_ismapped():
            self._append_realtime_log("Встроенный график не отобразился; проверьте раскладку окна.")
        self._animate_realtime_frame(0, delay)

    def _animate_realtime_frame(self, frame_index: int, delay: float) -> None:
        if self._current_plotter is None or self._current_session is None or self._current_canvas is None:
            return
        if frame_index >= len(self._current_session.frame_states):
            self.status_var.set("Поток завершен")
            self._append_realtime_log("Поток завершен.")
            self._realtime_after_id = None
            return
        frame_state = self._current_session.frame_states[frame_index]
        self._current_plotter.render_frame(
            frame_state,
            self._current_session.segments_by_method,
            self._current_session.online_labels_by_method,
        )
        self._current_canvas.draw()
        self.status_var.set(
            f"Кадр потока {frame_index + 1}/{len(self._current_session.frame_states)}"
        )
        self._realtime_after_id = self.root.after(max(1, int(delay * 1000)), lambda: self._animate_realtime_frame(frame_index + 1, delay))

    def start_metrics_run(self) -> None:
        methods = self._selected_methods()
        if not methods:
            messagebox.showerror("Не выбран алгоритм", "Выберите хотя бы один алгоритм.")
            return
        choice = self.choice_by_label.get(self.metric_dataset_var.get())
        if choice is None:
            messagebox.showerror("Ошибка набора данных", "Сначала выберите набор данных.")
            return
        if choice.kind != "synthetic":
            messagebox.showerror("Ошибка набора данных", "Вкладка метрик работает только с синтетическими наборами.")
            return

        payload = {
            "choice": choice,
            "methods": methods,
            "runs": max(1, int(self.metric_runs_var.get())),
            "metric_mode": self.metric_mode_var.get().strip().lower(),
            "standardize": True,
            "series_count": max(1, int(choice.default_series_count)),
            "window_size": max(2, int(self.metric_window_size_var.get())),
        }
        self._append_metrics_log(f"Старт метрик для {choice.label}")
        self.status_var.set(f"Метрики: {choice.label}")
        self.metrics_output.configure(state="normal")
        self.metrics_output.delete("1.0", END)
        self.metrics_output.insert(END, "Вычисление синтетических метрик...\n")
        
        threading.Thread(target=self._compute_metrics_report, args=(payload,), daemon=True).start()

    def run_synthetic_boundary_suite(self) -> None:
        self._run_captured_named(
            "synthetic suite",
            self.metrics_output,
            run_synthetic_boundary_metrics_benchmark,
            output_dir=self.project_root / "results" / "SyntheticBoundaryBenchmark",
            runs_per_dataset=max(1, int(self.metric_runs_var.get())),
            window_size=max(2, int(self.metric_window_size_var.get())),
            standardize=True,
        )

    def run_synthetic_realtime_metrics_suite(self) -> None:
        self._run_captured_named(
            "synthetic suite",
            self.metrics_output,
            run_synthetic_realtime_metrics_benchmark,
            output_dir=self.project_root / "results" / "SyntheticTrueRealtimeMetricsBenchmark",
            runs_per_dataset=max(1, int(self.metric_runs_var.get())),
            window_size=max(2, int(self.metric_window_size_var.get())),
        )

    def _run_captured_named(self, title: str, widget: scrolledtext.ScrolledText, func, **kwargs) -> None:
        self._set_text(widget, f"Running {title}...\n")
        threading.Thread(target=self._run_captured_task, args=(title, widget, func, kwargs), daemon=True).start()

    def _run_captured_task(self, title: str, widget: scrolledtext.ScrolledText, func, kwargs: dict[str, object]) -> None:
        buffer = io.StringIO()
        preview_plot = bool(kwargs.pop("__preview_plot__", False))
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            output = buffer.getvalue() + f"\nERROR: {exc}\n"
            self._queue.put(("widget-text", (widget, f"{output}\n")))
            self._queue.put(("error", f"{title} run failed: {exc}"))
            return
        output = buffer.getvalue().strip()
        if not output:
            output = f"{title} completed."
        self._queue.put(("widget-text", (widget, output + "\n")))

    def _compute_metrics_report(self, payload: dict[str, object]) -> None:
        try:
            choice = payload["choice"]
            assert isinstance(choice, DatasetChoice)
            report = self._build_metrics_report(
                choice=choice,
                methods=list(payload["methods"]),
                runs=int(payload["runs"]),
                metric_mode=str(payload["metric_mode"]),
                standardize=bool(payload["standardize"]),
                series_count=int(payload["series_count"]),
                window_size=int(payload["window_size"]),
            )
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"Metrics run failed: {exc}"))
            return
        self._queue.put(("metrics-ready", report))

    def _build_metrics_report(
        self,
        choice: DatasetChoice,
        methods: list[str],
        runs: int,
        metric_mode: str,
        standardize: bool,
        series_count: int,
        window_size: int,
    ) -> str:
        clustream_params, denstream_params = self._cluster_param_overrides()
        boundary_metrics: dict[str, list[BoundaryMetrics]] = {}
        quality_metrics: dict[str, list[dict[str, float | int]]] = {}
        run_summaries: list[str] = []

        for run_index in range(runs):
            stream = _load_stream(
                choice.dataset_path,
                series_index=0,
                series_count=series_count,
                balanced=choice.balanced,
                max_points=choice.default_max_points,
            )
            session = self._build_realtime_session(
                run_id=run_index + 1,
                choice=choice,
                stream=stream,
                methods=methods,
                window_size=window_size,
                window_mode="expanding",
                feature_profile="basic",
                standardize=standardize,
                clustream_params=clustream_params or {},
                denstream_params=denstream_params or {},
            )

            run_summaries.append(
                f"Run {run_index + 1}: methods={', '.join(session.method_names)}, frames={len(session.frame_states)}"
            )
            for method_name, metrics in session.boundary_metrics_by_method.items():
                if metrics is not None:
                    boundary_metrics.setdefault(method_name, []).append(metrics)
            for method_name, labels in session.evaluation_labels_by_method.items():
                source_classes = [
                    int(stream.source_labels[min(segment.center, len(stream.source_labels) - 1)])
                    for segment in session.segments_by_method[method_name]
                ]
                sample_labels = window_labels_to_sample_labels(
                    len(stream.signal),
                    session.segments_by_method[method_name],
                    labels,
                )
                cluster_metrics = cluster_quality_metrics(
                    labels,
                    source_classes,
                    sample_labels=sample_labels,
                )
                quality_metrics.setdefault(method_name, []).append(
                    {
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
                )

        lines = [
            f"Dataset: {choice.label}",
            f"Runs: {runs}",
            f"Algorithms: {', '.join(methods)}",
            f"Window size: {window_size}",
            "",
            "Run details:",
            *[f"  {line}" for line in run_summaries],
            "",
        ]

        metric_mode = metric_mode.strip().lower()
        show_boundary = metric_mode in {"boundary", "full"}
        show_quality = metric_mode in {"quality", "full"}

        if show_boundary:
            lines.append("Boundary metrics:")
            for method_name in methods:
                aggregated = (
                    _aggregate_boundary_metrics(boundary_metrics[method_name], method_name)
                    if boundary_metrics.get(method_name)
                    else None
                )
                if aggregated is None:
                    lines.extend([f"{method_name}:", "  boundary metrics unavailable"])
                else:
                    lines.extend(self._format_boundary_block(aggregated))
                lines.append("")

        if show_quality:
            lines.append("Cluster quality metrics:")
            for method_name in methods:
                aggregated_quality = (
                    _aggregate_cluster_quality_metrics(quality_metrics[method_name], method_name)
                    if quality_metrics.get(method_name)
                    else {}
                )
                lines.append(f"{method_name}:")
                if aggregated_quality:
                    lines.extend(_format_quality_block_from_dict(aggregated_quality))
                else:
                    lines.append("  quality metrics unavailable")
                lines.append("")

        return "\n".join(lines).rstrip()

    def _format_boundary_block(self, metrics: BoundaryMetrics) -> list[str]:
        return [
            f"{metrics.method}:",
            f"  boundary MAE: {self._format_float(metrics.boundary_mae)}",
            f"  boundary max error: {self._format_float(metrics.boundary_max_error)}",
            f"  FAR: {self._format_float(metrics.far)}",
            f"  FRR: {self._format_float(metrics.frr)}",
            f"  Accuracy: {self._format_float(metrics.accuracy)}",
            f"  Recall: {self._format_float(metrics.recall)}",
            f"  Precision: {self._format_float(metrics.precision)}",
        ]

    @staticmethod
    def _format_float(value: float | None, digits: int = 3) -> str:
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"

    def _append_realtime_log(self, text: str) -> None:
        self._append_text(self.realtime_log, text)
        self._append_text(self.global_log, f"[Поток] {text}")

    def _append_metrics_log(self, text: str) -> None:
        self._append_text(self.metrics_output, text)
        self._append_text(self.global_log, f"[Метрики] {text}")

    def _set_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", END)
        widget.insert(END, text)
        widget.see(END)

    def _clear_plot_host(self, host: ttk.Frame | None = None) -> None:
        if host is None:
            host = self.realtime_plot_host
        for child in host.winfo_children():
            child.destroy()
        if host is self.realtime_plot_host:
            self._current_canvas = None
            self._current_toolbar = None
            self._current_plotter = None

    def _append_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(END, text + "\n")
        widget.see(END)

    def _style_text_widget(self, widget: scrolledtext.ScrolledText) -> None:
        widget.configure(
            background=self._panel_bg,
            foreground="#111827",
            insertbackground="#111827",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#d8ccb6",
            highlightcolor=self._accent_color,
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        widget.bind(
            "<KeyPress>",
            lambda event: None
            if (
                event.state & 0x4
                and event.keysym.lower() in {"c", "a"}
            )
            else "break",
        )

    def _gui_settings_payload(self) -> dict[str, object]:
        return {
            "use_custom_cluster_params": bool(self.use_custom_cluster_params_var.get()),
            "pipeline": {
                "window_size": int(self.window_size_var.get()),
                "series_count": int(self.series_count_var.get()),
                "max_points": int(self.max_points_var.get()),
                "delay": float(self.delay_var.get()),
                        },
            "clustream": {
                "max_micro_clusters": int(self.clustream_max_micro_clusters_var.get()),
                "radius_factor": float(self.clustream_radius_factor_var.get()),
                "min_radius": float(self.clustream_min_radius_var.get()),
                "stale_after": int(self.clustream_stale_after_var.get()),
                "tentative_distance": float(self.clustream_tentative_distance_var.get()),
                "emit_micro_labels": bool(self.clustream_emit_micro_labels_var.get()),
            },
            "denstream": {
                "epsilon": float(self.denstream_epsilon_var.get()),
                "beta": float(self.denstream_beta_var.get()),
                "mu": float(self.denstream_mu_var.get()),
                "lambd": float(self.denstream_lambd_var.get()),
                "prune_interval": int(self.denstream_prune_interval_var.get()),
                "radius_tolerance": float(self.denstream_radius_tolerance_var.get()),
                "emit_noise": bool(self.denstream_emit_noise_var.get()),
                "grace_period": int(self.denstream_grace_period_var.get()),
                "outlier_radius_tolerance": float(self.denstream_outlier_radius_tolerance_var.get()),
                "outlier_promote_weight": float(self.denstream_outlier_promote_weight_var.get()),
                "outlier_promote_age": int(self.denstream_outlier_promote_age_var.get()),
                "min_points_before_noise": int(self.denstream_min_points_before_noise_var.get()),
                "noise_score_threshold": float(self.denstream_noise_score_threshold_var.get()),
            },
        }

    def _save_gui_settings(self) -> None:
        payload = self._gui_settings_payload()
        try:
            self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            if hasattr(self, "global_log"):
                self._append_text(self.global_log, f"[Settings] Failed to save GUI settings: {exc}")
            self.status_var.set(f"Failed to save settings: {exc}")

    def _apply_saved_gui_settings(self, payload: dict[str, object]) -> None:
        pipeline = payload.get("pipeline", {})
        if isinstance(pipeline, dict):
            self.window_size_var.set(int(pipeline.get("window_size", self.window_size_var.get())))
            self.series_count_var.set(int(pipeline.get("series_count", self.series_count_var.get())))
            self.max_points_var.set(int(pipeline.get("max_points", self.max_points_var.get())))
            self.delay_var.set(float(pipeline.get("delay", self.delay_var.get())))

        clustream = payload.get("clustream", {})
        if isinstance(clustream, dict):
            self.clustream_max_micro_clusters_var.set(int(clustream.get("max_micro_clusters", self.clustream_max_micro_clusters_var.get())))
            self.clustream_radius_factor_var.set(float(clustream.get("radius_factor", self.clustream_radius_factor_var.get())))
            self.clustream_min_radius_var.set(float(clustream.get("min_radius", self.clustream_min_radius_var.get())))
            self.clustream_stale_after_var.set(int(clustream.get("stale_after", self.clustream_stale_after_var.get())))
            self.clustream_tentative_distance_var.set(float(clustream.get("tentative_distance", self.clustream_tentative_distance_var.get())))
            self.clustream_emit_micro_labels_var.set(bool(clustream.get("emit_micro_labels", self.clustream_emit_micro_labels_var.get())))

        denstream = payload.get("denstream", {})
        if isinstance(denstream, dict):
            self.denstream_epsilon_var.set(float(denstream.get("epsilon", self.denstream_epsilon_var.get())))
            self.denstream_beta_var.set(float(denstream.get("beta", self.denstream_beta_var.get())))
            self.denstream_mu_var.set(float(denstream.get("mu", self.denstream_mu_var.get())))
            self.denstream_lambd_var.set(float(denstream.get("lambd", self.denstream_lambd_var.get())))
            self.denstream_prune_interval_var.set(int(denstream.get("prune_interval", self.denstream_prune_interval_var.get())))
            self.denstream_radius_tolerance_var.set(float(denstream.get("radius_tolerance", self.denstream_radius_tolerance_var.get())))
            self.denstream_emit_noise_var.set(bool(denstream.get("emit_noise", self.denstream_emit_noise_var.get())))
            self.denstream_grace_period_var.set(int(denstream.get("grace_period", self.denstream_grace_period_var.get())))
            self.denstream_outlier_radius_tolerance_var.set(float(denstream.get("outlier_radius_tolerance", self.denstream_outlier_radius_tolerance_var.get())))
            self.denstream_outlier_promote_weight_var.set(float(denstream.get("outlier_promote_weight", self.denstream_outlier_promote_weight_var.get())))
            self.denstream_outlier_promote_age_var.set(int(denstream.get("outlier_promote_age", self.denstream_outlier_promote_age_var.get())))
            self.denstream_min_points_before_noise_var.set(int(denstream.get("min_points_before_noise", self.denstream_min_points_before_noise_var.get())))
            self.denstream_noise_score_threshold_var.set(float(denstream.get("noise_score_threshold", self.denstream_noise_score_threshold_var.get())))

        self.use_custom_cluster_params_var.set(bool(payload.get("use_custom_cluster_params", self.use_custom_cluster_params_var.get())))
        self._manual_pipeline_defaults_active = bool(self.use_custom_cluster_params_var.get())

    def _load_gui_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Failed to load GUI settings: {exc}")
            return
        if not isinstance(payload, dict):
            return
        self._apply_saved_gui_settings(payload)

    def _make_scrollable_panel(
        self,
        parent: ttk.Frame,
        width: int = 420,
        height: int = 860,
        horizontal: bool = True,
    ) -> tuple[ttk.Frame, ttk.Frame]:
        shell = ttk.Frame(parent, style="Card.TFrame")
        shell.configure(width=width, height=height)
        shell.pack_propagate(False)

        canvas = Canvas(shell, background=self._panel_bg, highlightthickness=0, borderwidth=0)
        scrollbar = Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = ttk.Frame(canvas, style="Card.TFrame")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _configure_inner(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _configure_canvas(event) -> None:
            canvas.itemconfigure(inner_id, width=event.width)

        inner.bind("<Configure>", _configure_inner)
        canvas.bind("<Configure>", _configure_canvas)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        def _on_mousewheel(event) -> None:
            delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")

        def _bind_mousewheel(_event) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_event) -> None:
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        return shell, inner

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "error":
                    self.status_var.set(str(payload))
                    self._append_text(self.global_log, str(payload))
                elif kind == "realtime-ready":
                    run_id, session, delay = payload
                    if int(run_id) == self._realtime_run_id:
                        self._start_realtime_animation(session, delay)
                elif kind == "metrics-ready":
                    self._append_metrics_log(str(payload))
                    self.metrics_output.configure(state="normal")
                    self.metrics_output.delete("1.0", END)
                    self.metrics_output.insert(END, str(payload))
                    self.status_var.set("Метрики завершены")
                elif kind == "widget-text":
                    widget, text = payload
                    if isinstance(widget, scrolledtext.ScrolledText):
                        self._set_text(widget, str(text))
        except Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _parse_delay(self) -> float:
        try:
            return max(0.01, float(self.delay_var.get()))
        except Exception:
            return 0.08


def _format_quality_block_from_dict(metrics: dict[str, float | int]) -> list[str]:
    return [
        f"  windows: {metrics.get('windows', 0)}",
        f"  cluster count: {metrics.get('cluster_count', 0)}",
        f"  noise count: {metrics.get('noise_count', 0)}",
        f"  noise ratio: {ClusterProjectApp._format_float(float(metrics.get('noise_ratio', 0.0)))}",
        f"  largest cluster ratio: {ClusterProjectApp._format_float(float(metrics.get('largest_cluster_ratio', 0.0)))}",
        f"  label switch ratio: {ClusterProjectApp._format_float(float(metrics.get('label_switch_ratio', 0.0)))}",
        f"  visual run count: {metrics.get('visual_run_count', 0)}",
        f"  mean visual run length: {ClusterProjectApp._format_float(float(metrics.get('mean_visual_run_length', 0.0)))}",
        f"  short visual run ratio: {ClusterProjectApp._format_float(float(metrics.get('short_visual_run_ratio', 0.0)))}",
        f"  transitions per 100 samples: {ClusterProjectApp._format_float(float(metrics.get('transitions_per_100_samples', 0.0)))}",
        f"  entropy: {ClusterProjectApp._format_float(float(metrics.get('entropy', 0.0)))}",
        f"  purity: {ClusterProjectApp._format_float(float(metrics.get('purity', 0.0)))}",
        f"  score: {ClusterProjectApp._format_float(float(metrics.get('score', 0.0)))}",
    ]


def launch_application(project_root: str | Path | None = None) -> None:
    root_path = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    try:
        root = Tk()
    except TclError as exc:  # pragma: no cover - GUI environments only
        raise RuntimeError("Tkinter GUI is unavailable in this environment") from exc
    ClusterProjectApp(root, root_path)
    root.mainloop()



