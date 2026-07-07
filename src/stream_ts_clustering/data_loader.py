from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


@dataclass(frozen=True)
class TimeSeriesDataset:
    labels: np.ndarray
    series: np.ndarray


@dataclass(frozen=True)
class StreamSignal:
    signal: np.ndarray
    source_labels: np.ndarray
    series_ranges: list[tuple[int, int]]
    reference_signal: np.ndarray | None = None


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    key: str
    name: str


@dataclass(frozen=True)
class UciStreamSpec:
    key: str
    name: str
    url: str


def load_ucr_tsv(path: str | Path) -> TimeSeriesDataset:
    """Р—Р°РіСЂСѓР·РёС‚СЊ UCR-style TSV/TXT, РіРґРµ СЃС‚РѕР»Р±РµС† 0 вЂ” РєР»Р°СЃСЃ, Р° РґР°Р»СЊС€Рµ РёРґС‘С‚ СЂСЏРґ."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    delimiter = "\t" if path.suffix.lower() == ".tsv" else None
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected UCR TSV with at least 2 columns: {path}")

    return TimeSeriesDataset(labels=data[:, 0].astype(int), series=data[:, 1:].astype(float))


def load_univariate_csv_stream(
    path: str | Path,
    value_column: str = "value",
    max_points: int | None = None,
) -> StreamSignal:
    """Р—Р°РіСЂСѓР·РёС‚СЊ РЅРµРїСЂРµСЂС‹РІРЅС‹Р№ РѕРґРЅРѕРјРµСЂРЅС‹Р№ CSV-РїРѕС‚РѕРє СЃ Р·Р°РіРѕР»РѕРІРєРѕРј Рё РѕРґРЅРёРј СЃС‚РѕР»Р±С†РѕРј."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    values: np.ndarray | None = None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if getattr(data, "dtype", None) is not None and data.dtype.names and value_column in data.dtype.names:
            values = np.asarray(data[value_column], dtype=float)
    except Exception:
        values = None

    if values is None:
        raw = np.genfromtxt(path, delimiter=",", dtype=str, encoding="utf-8")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 2:
            raise ValueError(f"CSV stream must contain at least two columns: {path}")
        try:
            values = np.asarray(raw[:, 1], dtype=float)
        except ValueError:
            if len(raw) <= 1:
                raise ValueError(f"CSV stream is too short: {path}")
            values = np.asarray(raw[1:, 1], dtype=float)

    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
    if len(values) < 2:
        raise ValueError(f"CSV stream is too short: {path}")

    if max_points is not None and max_points > 0 and len(values) > max_points:
        values = values[:max_points]

    return StreamSignal(
        signal=values,
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def load_labeled_csv_stream(
    path: str | Path,
    value_column: str,
    label_column: str | None = None,
    max_points: int | None = None,
    delimiter: str = ",",
) -> StreamSignal:
    """Load a single-channel CSV stream and optionally extract source labels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    values: list[float] = []
    labels: list[int] = []
    label_lookup: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV stream has no header: {path}")
        header = [item.strip().strip('"') for item in header]
        value_index = _find_column_index(header, value_column)
        if value_index is None:
            raise ValueError(f"Value column not found: {value_column}")
        label_index = _find_column_index(header, label_column) if label_column else None

        for row in reader:
            if value_index >= len(row):
                continue
            value = _parse_float(row[value_index])
            if value is None:
                continue
            values.append(value)
            if label_index is not None and label_index < len(row):
                label_item = row[label_index].strip().strip('"')
                label_value = _parse_float(label_item)
                if label_value is not None:
                    labels.append(int(label_value))
                else:
                    labels.append(_label_to_int(label_item, label_lookup))
            else:
                labels.append(0)
            if max_points is not None and max_points > 0 and len(values) >= max_points:
                break

    if len(values) < 2:
        raise ValueError(f"CSV stream is too short: {path}")

    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.asarray(labels, dtype=int),
        series_ranges=[(0, len(values))],
    )


def load_multivariate_csv_stream(
    path: str | Path,
    feature_columns: list[str],
    label_column: str | None = None,
    max_points: int | None = None,
    delimiter: str = ",",
) -> StreamSignal:
    """Load a multi-column CSV stream and collapse each row into one scalar signal."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    values: list[float] = []
    labels: list[int] = []
    label_lookup: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV stream has no header: {path}")
        header = [item.strip().strip('"') for item in header]
        feature_indices = []
        for column in feature_columns:
            index = _find_column_index(header, column)
            if index is None:
                raise ValueError(f"Feature column not found: {column}")
            feature_indices.append(index)
        label_index = _find_column_index(header, label_column) if label_column else None

        for row in reader:
            numeric_values: list[float] = []
            for index in feature_indices:
                if index >= len(row):
                    continue
                parsed = _parse_float(row[index])
                if parsed is not None:
                    numeric_values.append(parsed)
            if not numeric_values:
                continue
            values.append(float(np.sqrt(np.mean(np.square(numeric_values)))))
            if label_index is not None and label_index < len(row):
                label_item = row[label_index].strip().strip('"')
                label_value = _parse_float(label_item)
                if label_value is not None:
                    labels.append(int(label_value))
                else:
                    labels.append(_label_to_int(label_item, label_lookup))
            else:
                labels.append(0)
            if max_points is not None and max_points > 0 and len(values) >= max_points:
                break

    if len(values) < 2:
        raise ValueError(f"CSV stream is too short: {path}")

    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.asarray(labels, dtype=int),
        series_ranges=[(0, len(values))],
    )


SYNTHETIC_DATASETS = [
    # Р‘Р°Р·РѕРІС‹Рµ СЃС†РµРЅР°СЂРёРё СЃРѕ СЃРјРµРЅРѕР№ РѕРґРЅРѕРіРѕ СЂРµР¶РёРјР°.
    SyntheticDatasetSpec("mean_shift", "Synthetic Mean Shift"),
    SyntheticDatasetSpec("variance_shift", "Synthetic Variance Shift"),
    SyntheticDatasetSpec("amplitude_shift", "Synthetic Amplitude Shift"),
    SyntheticDatasetSpec("graph_1", "Synthetic Graph 1"),
    SyntheticDatasetSpec("graph_2", "Synthetic Graph 2"),
    SyntheticDatasetSpec("graph_3", "Synthetic Graph 3"),
    # РЎР»РѕР¶РЅС‹Рµ СЃС†РµРЅР°СЂРёРё СЃ РЅРµСЃРєРѕР»СЊРєРёРјРё РїРµСЂРµС…РѕРґР°РјРё.
    SyntheticDatasetSpec("mixed_regime", "Synthetic Mixed Regime"),
    SyntheticDatasetSpec("parametric_regime", "Synthetic Parametric Regime"),
]


def list_synthetic_dataset_specs() -> list[SyntheticDatasetSpec]:
    return list(SYNTHETIC_DATASETS)


def list_synthetic_validation_dataset_specs() -> list[SyntheticDatasetSpec]:
    return [
        spec
        for spec in SYNTHETIC_DATASETS
        if spec.key not in {
            "graph_1",
            "amplitude_shift",
            "mixed_regime",
            "parametric_regime",
        }
    ]


def make_synthetic_segmentation_stream(
    dataset_key: str,
    segment_length: int = 100,
    noise_seed: int | None = None,
) -> StreamSignal:
    """РџРѕСЃС‚СЂРѕРёС‚СЊ РѕРґРёРЅ synthetic-СЃРёРіРЅР°Р» РїРѕРґ РІС‹Р±СЂР°РЅРЅС‹Р№ СЃС†РµРЅР°СЂРёР№ СЃРјРµРЅС‹ СЂРµР¶РёРјР°."""
    if segment_length < 100:
        raise ValueError("segment_length must be at least 100")

    rng = np.random.default_rng(noise_seed)
    t = np.arange(segment_length, dtype=float)
    split_a = segment_length // 2
    split_b = segment_length // 3
    split_c = 2 * segment_length // 3
    base_period = max(12.0, segment_length / 8.0)

    def sine(period: float, amplitude: float = 1.0, offset: float = 0.0) -> np.ndarray:
        return offset + amplitude * np.sin(2.0 * np.pi * t / period)

    if dataset_key == "mean_shift":
        # РњРµРЅСЏРµС‚СЃСЏ С‚РѕР»СЊРєРѕ СЃСЂРµРґРЅРёР№ СѓСЂРѕРІРµРЅСЊ, С„РѕСЂРјР° СЃРёРЅСѓСЃР° СЃРѕС…СЂР°РЅСЏРµС‚СЃСЏ.
        signal = sine(base_period, amplitude=2.5)
        signal[:split_a] += rng.normal(0.0, 0.06, size=split_a)
        signal[split_a:] += 3.1 + rng.normal(0.0, 0.04, size=segment_length - split_a)
        source_labels = np.concatenate(
            [
                np.zeros(split_a, dtype=int),
                np.ones(segment_length - split_a, dtype=int),
            ]
        )
    elif dataset_key == "variance_shift":
        # РњРµРЅСЏРµС‚СЃСЏ РґРёСЃРїРµСЂСЃРёСЏ: РїРѕСЃР»Рµ РіСЂР°РЅРёС†С‹ С€СѓРј СЃС‚Р°РЅРѕРІРёС‚СЃСЏ Р·Р°РјРµС‚РЅРѕ СЃРёР»СЊРЅРµРµ.
        signal = sine(max(12.0, segment_length / 10.0), amplitude=1.4)
        signal[:split_a] += rng.normal(0.0, 0.04, size=split_a)
        signal[split_a:] += rng.normal(0.0, 0.9, size=segment_length - split_a)
        source_labels = np.concatenate(
            [
                np.zeros(split_a, dtype=int),
                np.ones(segment_length - split_a, dtype=int),
            ]
        )
    elif dataset_key == "amplitude_shift":
        # Р§Р°СЃС‚РѕС‚Р° РЅРµ РјРµРЅСЏРµС‚СЃСЏ, РЅРѕ Р°РјРїР»РёС‚СѓРґР° СЃС‚Р°РЅРѕРІРёС‚СЃСЏ РІС‹С€Рµ.
        signal = np.empty(segment_length, dtype=float)
        signal[:split_a] = 1.0 * np.sin(2.0 * np.pi * t[:split_a] / base_period) + rng.normal(
            0.0,
            0.10,
            size=split_a,
        )
        signal[split_a:] = 5.5 * np.sin(2.0 * np.pi * t[split_a:] / base_period) + rng.normal(
            0.0,
            0.06,
            size=segment_length - split_a,
        )
        source_labels = np.concatenate(
            [
                np.zeros(split_a, dtype=int),
                np.ones(segment_length - split_a, dtype=int),
            ]
        )
    elif dataset_key == "graph_1":
        mean_shift = np.where(t >= 50.0, 18.0, 0.0)
        reference_signal = (
            0.12 * t
            + 5.0 * np.sin(2.0 * np.pi * t / 35.0)
            + 2.5 * np.sin(2.0 * np.pi * t / 2.0)
            + mean_shift
        )
        signal = reference_signal + rng.normal(0.0, 2.0, size=segment_length)
        source_labels = np.zeros(segment_length, dtype=int)
        source_labels[t >= 50.0] = 1
    elif dataset_key == "graph_2":
        # Three regimes: near-flat baseline, a clearly visible middle burst,
        # then a separate upward-trending regime. The burst is limited to
        # samples 40..60 so it is a regime, not a tail leaking into regime 3.
        middle_mask = (t >= 40.0) & (t < 70.0)
        spike_up = np.where(middle_mask, 5.2 * np.exp(-(t - 40.0) / 6.0), 0.0)
        spike_down = np.where(middle_mask & (t >= 49.0), -3.5 * np.exp(-(t - 49.0) / 6.5), 0.0)
        middle_oscillation = np.where(
            middle_mask,
            0.90 * np.sin(2.0 * np.pi * (t - 40.0) / 6.0),
            0.0,
        )
        trend = np.where(t >= 70.0, 3.2 + 0.300 * (t - 70.0), 0.0)
        signal = 21.0 + spike_up + spike_down + middle_oscillation + trend + rng.normal(0.0, 0.08, size=segment_length)
        source_labels = np.zeros(segment_length, dtype=int)
        source_labels[t >= 40.0] = 1
        source_labels[t >= 70.0] = 2
    elif dataset_key == "graph_3":
        signal = np.empty(segment_length, dtype=float)
        first_regime = t < 20.0
        signal[first_regime] = 25.0 - 0.15 * t[first_regime] + rng.normal(0.0, 2.5, size=int(first_regime.sum()))
        signal[~first_regime] = 19.2 + rng.normal(0.0, 0.25, size=int((~first_regime).sum()))
        source_labels = np.zeros(segment_length, dtype=int)
        source_labels[~first_regime] = 1
    elif dataset_key == "mixed_regime":
        # Р—РґРµСЃСЊ СЃРѕР±СЂР°РЅС‹ СЃСЂР°Р·Сѓ РЅРµСЃРєРѕР»СЊРєРѕ С‚РёРїРѕРІ РёР·РјРµРЅРµРЅРёР№, С‡С‚РѕР±С‹ РїСЂРѕРІРµСЂРёС‚СЊ
        # СѓСЃС‚РѕР№С‡РёРІРѕСЃС‚СЊ СЃРµРіРјРµРЅС‚Р°С†РёРё Рё РєР»Р°СЃС‚РµСЂРёР·Р°С†РёРё РЅР° Р±РѕР»РµРµ СЃР»РѕР¶РЅРѕР№ РєР°СЂС‚РёРЅРµ.
        signal = np.empty(segment_length, dtype=float)
        split_b = segment_length // 3
        split_c = 2 * segment_length // 3
        signal[:split_b] = 0.95 * np.sin(2.0 * np.pi * t[:split_b] / max(7.0, segment_length / 14.0)) + rng.normal(0.0, 0.08, size=split_b)
        signal[split_b:split_c] = 3.5 * np.sin(
            2.0 * np.pi * t[split_b:split_c] / max(5.0, segment_length / 22.0)
        ) + 1.4 + rng.normal(0.0, 0.24, size=split_c - split_b)
        signal[split_c:] = 1.6 * np.sin(2.0 * np.pi * t[split_c:] / max(4.5, segment_length / 24.0)) - 0.9 + rng.normal(0.0, 0.30, size=segment_length - split_c)
        source_labels = np.concatenate(
            [
                np.zeros(split_b, dtype=int),
                np.ones(split_c - split_b, dtype=int),
                np.full(segment_length - split_c, 2, dtype=int),
            ]
        )
    elif dataset_key == "parametric_regime":
        # РўСЂРё СѓС‡Р°СЃС‚РєР° СЃ СЂР°Р·РЅС‹РјРё РїР°СЂР°РјРµС‚СЂР°РјРё A(t), mu(t), T(t) Рё sigma(t).
        mu = np.concatenate(
            [
                np.zeros(split_b, dtype=float),
                np.full(split_c - split_b, 8.0, dtype=float),
                np.full(segment_length - split_c, -4.0, dtype=float),
            ]
        )
        amplitude = np.concatenate(
            [
                np.full(split_b, 0.9, dtype=float),
                np.full(split_c - split_b, 0.5, dtype=float),
                np.full(segment_length - split_c, 1.4, dtype=float),
            ]
        )
        period = np.concatenate(
            [
                np.full(split_b, max(5.0, segment_length / 22.0), dtype=float),
                np.full(split_c - split_b, max(4.0, segment_length / 28.0), dtype=float),
                np.full(segment_length - split_c, max(7.0, segment_length / 16.0), dtype=float),
            ]
        )
        sigma = np.concatenate(
            [
                np.full(split_b, 0.18, dtype=float),
                np.full(split_c - split_b, 0.38, dtype=float),
                np.full(segment_length - split_c, 0.30, dtype=float),
            ]
        )
        signal = amplitude * np.sin(2.0 * np.pi * t / period) + mu + rng.normal(0.0, sigma)
        source_labels = np.concatenate(
            [
                np.zeros(split_b, dtype=int),
                np.ones(split_c - split_b, dtype=int),
                np.full(segment_length - split_c, 2, dtype=int),
            ]
        )
    else:
        raise ValueError(f"Unsupported synthetic dataset: {dataset_key}")

    return StreamSignal(
        signal=np.asarray(signal, dtype=float),
        source_labels=source_labels,
        series_ranges=[(0, segment_length)],
        reference_signal=np.asarray(reference_signal, dtype=float) if dataset_key == "graph_1" else None,
    )


UCI_STREAMS = {
    "gas_drift": UciStreamSpec(
        key="gas_drift",
        name="Gas Sensor Array Drift at Different Concentrations",
        url="https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip",
    ),
    "air_quality": UciStreamSpec(
        key="air_quality",
        name="Air Quality",
        url="https://archive.ics.uci.edu/static/public/387/air%2Bquality.zip",
    ),
    "electricity_load": UciStreamSpec(
        key="electricity_load",
        name="ElectricityLoadDiagrams20112014",
        url="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
    ),
}


def load_uci_stream(dataset_source: str | Path, max_points: int | None = None) -> StreamSignal:
    path = Path(dataset_source)
    if path.exists():
        return _load_local_uci_stream(path, max_points=max_points)
    if str(dataset_source) == "har":
        return _load_har_stream(max_points=max_points)
    if str(dataset_source) == "mex":
        return _load_mex_stream(max_points=max_points)
    if str(dataset_source) == "gas_drift":
        return _load_gas_drift_stream(max_points=max_points)
    if str(dataset_source) == "air_quality":
        return _load_air_quality_stream(max_points=max_points)
    if str(dataset_source) == "electricity_load":
        return _load_electricity_load_stream(max_points=max_points)
    raise ValueError(f"Unsupported UCI dataset: {dataset_source}")


def _load_local_uci_stream(path: Path, max_points: int | None = None) -> StreamSignal:
    parts = {part.lower() for part in path.parts}
    normalized = path.as_posix().lower()
    if "uci har dataset" in parts:
        return _load_har_stream_from_directory(path, max_points=max_points)
    if "mex" in parts:
        return _load_mex_stream_from_directory(path, max_points=max_points)
    if path.is_dir() or "gas_drift" in normalized or path.suffix.lower() == ".dat":
        return _load_gas_drift_from_directory(path, max_points=max_points)
    if "airquality" in normalized or path.suffix.lower() == ".csv":
        return _load_air_quality_from_csv(path, max_points=max_points)
    if "ld2011_2014" in normalized or path.suffix.lower() == ".txt":
        return _load_electricity_load_from_txt(path, max_points=max_points)
    raise ValueError(f"Unsupported local UCI dataset path: {path}")


def _load_gas_drift_from_directory(directory: Path, max_points: int | None = None) -> StreamSignal:
    if directory.is_file():
        batch_paths = [directory]
    else:
        batch_paths = sorted(directory.glob("batch*.dat"), key=_batch_sort_key)
        if not batch_paths:
            batch_paths = sorted(directory.glob("*.dat"), key=_batch_sort_key)
    values: list[float] = []
    for batch_path in batch_paths:
        with batch_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = _parse_numeric_tokens(line)
                if len(parsed) < 2:
                    continue
                values.append(float(np.mean(parsed[1:])))
                if max_points is not None and max_points > 0 and len(values) >= max_points:
                    return StreamSignal(
                        signal=np.asarray(values, dtype=float),
                        source_labels=np.zeros(len(values), dtype=int),
                        series_ranges=[(0, len(values))],
                    )
    if len(values) < 2:
        raise ValueError("Gas drift stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def _load_air_quality_from_csv(path: Path, max_points: int | None = None) -> StreamSignal:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header:
            raise ValueError("Air Quality file has no header")
        selected_column = _find_first_column(
            header,
            [
                "PT08.S1(CO)",
                "PT08.S2(NMHC)",
                "PT08.S3(NOx)",
                "PT08.S4(NO2)",
                "PT08.S5(O3)",
            ],
        )
        if selected_column is None:
            selected_column = _first_numeric_column(header)
        for row in reader:
            if selected_column >= len(row):
                continue
            value = _parse_float(row[selected_column])
            if value is None or value == -200:
                continue
            values.append(value)
            if max_points is not None and max_points > 0 and len(values) >= max_points:
                break
    if len(values) < 2:
        raise ValueError("Air Quality stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def _load_electricity_load_from_txt(path: Path, max_points: int | None = None) -> StreamSignal:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header:
            raise ValueError("Electricity load file has no header")
        for row in reader:
            numeric = [_parse_float(item) for item in row[1:]]
            numeric_values = [value for value in numeric if value is not None]
            if not numeric_values:
                continue
            values.append(float(np.mean(numeric_values)))
            if max_points is not None and max_points > 0 and len(values) >= max_points:
                break
    if len(values) < 2:
        raise ValueError("Electricity load stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def _load_har_stream(max_points: int | None = None) -> StreamSignal:
    root = Path("datasets") / "UCI" / "har" / "UCI HAR Dataset"
    return _load_har_stream_from_directory(root, max_points=max_points)


def _load_har_stream_from_directory(path: Path, max_points: int | None = None) -> StreamSignal:
    root = path
    if root.is_file():
        root = root.parent
    if root.name != "UCI HAR Dataset" and (root / "UCI HAR Dataset").exists():
        root = root / "UCI HAR Dataset"
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        label_path = root / split / f"y_{split}.txt"
        axis_paths = [
            root / split / "Inertial Signals" / f"body_acc_{axis}_{split}.txt"
            for axis in ("x", "y", "z")
        ]
        if not label_path.exists() or not all(axis_path.exists() for axis_path in axis_paths):
            raise FileNotFoundError(f"HAR split is incomplete: {split}")

        labels = np.loadtxt(label_path, dtype=int)
        axes = [np.loadtxt(axis_path) for axis_path in axis_paths]
        axes = [axis if axis.ndim == 2 else axis.reshape(1, -1) for axis in axes]
        # Keep more of the three-axis movement energy than a plain mean would.
        # Each row is a 128-sample inertial window, so the scalar stream is the
        # per-window magnitude energy with a small variability term.
        magnitude = np.sqrt(axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2)
        signal = magnitude.mean(axis=1) + 0.25 * magnitude.std(axis=1)
        return signal.astype(float), labels.astype(int)

    train_signal, train_labels = load_split("train")
    test_signal, test_labels = load_split("test")
    signal = np.concatenate([train_signal, test_signal])
    source_labels = np.concatenate([train_labels, test_labels])

    if max_points is not None and max_points > 0 and len(signal) > max_points:
        signal = signal[:max_points]
        source_labels = source_labels[:max_points]

    return StreamSignal(
        signal=np.asarray(signal, dtype=float),
        source_labels=np.asarray(source_labels, dtype=int),
        series_ranges=[(0, len(signal))],
    )


def _load_mex_stream(max_points: int | None = None) -> StreamSignal:
    root = Path("datasets") / "UCI" / "mex" / "act"
    return _load_mex_stream_from_directory(root, max_points=max_points)


def _load_mex_stream_from_directory(path: Path, max_points: int | None = None) -> StreamSignal:
    root = path
    if root.is_file():
        root = root.parent
    if root.name.lower() != "act" and (root / "act").exists():
        root = root / "act"
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    files = [p for p in root.rglob("*.csv") if p.is_file()]
    if not files:
        raise ValueError(f"No CSV files found in {root}")

    grouped: dict[int, list[Path]] = {}
    for csv_path in files:
        match = re.match(r"(\d+)_act_", csv_path.name)
        if not match:
            continue
        exercise_label = int(match.group(1))
        grouped.setdefault(exercise_label, []).append(csv_path)

    selected_files: list[tuple[int, Path]] = []
    for exercise_label in sorted(grouped):
        selected_files.append((exercise_label, sorted(grouped[exercise_label], key=lambda p: (p.parent.as_posix(), p.name))[0]))
    if not selected_files:
        raise ValueError(f"No exercise files found in {root}")

    per_file_limit = None
    if max_points is not None and max_points > 0:
        per_file_limit = max(50, max_points // len(selected_files))

    values: list[float] = []
    labels: list[int] = []
    for exercise_label, csv_path in selected_files:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            used_from_file = 0
            for row in reader:
                numeric = [_parse_float(item) for item in row[1:]]
                numeric_values = [value for value in numeric if value is not None]
                if not numeric_values:
                    continue
                values.append(float(np.sqrt(np.mean(np.square(numeric_values)))))
                labels.append(exercise_label)
                used_from_file += 1
                if max_points is not None and max_points > 0 and len(values) >= max_points:
                    return StreamSignal(
                        signal=np.asarray(values, dtype=float),
                        source_labels=np.asarray(labels, dtype=int),
                        series_ranges=[(0, len(values))],
                    )
                if per_file_limit is not None and used_from_file >= per_file_limit:
                    break
    if len(values) < 2:
        raise ValueError("MEx stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.asarray(labels, dtype=int),
        series_ranges=[(0, len(values))],
    )


def _download_uci_zip(spec: UciStreamSpec) -> Path:
    cache_dir = Path("datasets") / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{spec.key}.zip"
    if not cache_path.exists():
        urlretrieve(spec.url, cache_path)
    return cache_path


def _load_gas_drift_stream(max_points: int | None = None) -> StreamSignal:
    spec = UCI_STREAMS["gas_drift"]
    zip_path = _download_uci_zip(spec)
    values: list[float] = []
    with zipfile.ZipFile(zip_path) as archive:
        batch_names = sorted(
            [name for name in archive.namelist() if name.lower().endswith(".dat")],
            key=_batch_sort_key,
        )
        for batch_name in batch_names:
            with archive.open(batch_name) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
                for line in text:
                    parsed = _parse_numeric_tokens(line)
                    if len(parsed) < 2:
                        continue
                    values.append(float(np.mean(parsed[1:])))
                    if max_points is not None and max_points > 0 and len(values) >= max_points:
                        return StreamSignal(
                            signal=np.asarray(values, dtype=float),
                            source_labels=np.zeros(len(values), dtype=int),
                            series_ranges=[(0, len(values))],
                        )
    if len(values) < 2:
        raise ValueError("Gas drift stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def _load_air_quality_stream(max_points: int | None = None) -> StreamSignal:
    spec = UCI_STREAMS["air_quality"]
    zip_path = _download_uci_zip(spec)
    with zipfile.ZipFile(zip_path) as archive:
        member_name = _first_member_with_suffix(archive, (".csv", ".txt"))
        with archive.open(member_name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
            reader = csv.reader(text, delimiter=";")
            header = next(reader, None)
            if not header:
                raise ValueError("Air Quality archive has no header")
            selected_column = _find_first_column(
                header,
                [
                    "PT08.S1(CO)",
                    "PT08.S2(NMHC)",
                    "PT08.S3(NOx)",
                    "PT08.S4(NO2)",
                    "PT08.S5(O3)",
                ],
            )
            if selected_column is None:
                selected_column = _first_numeric_column(header)

            values: list[float] = []
            for row in reader:
                if selected_column >= len(row):
                    continue
                value = _parse_float(row[selected_column])
                if value is None or value == -200:
                    continue
                values.append(value)
                if max_points is not None and max_points > 0 and len(values) >= max_points:
                    break
    if len(values) < 2:
        raise ValueError("Air Quality stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def _load_electricity_load_stream(max_points: int | None = None) -> StreamSignal:
    spec = UCI_STREAMS["electricity_load"]
    zip_path = _download_uci_zip(spec)
    with zipfile.ZipFile(zip_path) as archive:
        member_name = _first_member_with_suffix(archive, (".txt", ".csv"))
        with archive.open(member_name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
            reader = csv.reader(text, delimiter=";")
            header = next(reader, None)
            if not header:
                raise ValueError("Electricity load archive has no header")
            values: list[float] = []
            for row in reader:
                numeric = [_parse_float(item) for item in row[1:]]
                numeric_values = [value for value in numeric if value is not None]
                if not numeric_values:
                    continue
                values.append(float(np.mean(numeric_values)))
                if max_points is not None and max_points > 0 and len(values) >= max_points:
                    break
    if len(values) < 2:
        raise ValueError("Electricity load stream is too short")
    return StreamSignal(
        signal=np.asarray(values, dtype=float),
        source_labels=np.zeros(len(values), dtype=int),
        series_ranges=[(0, len(values))],
    )


def _first_member_with_suffix(archive: zipfile.ZipFile, suffixes: tuple[str, ...]) -> str:
    members = [name for name in archive.namelist() if name.lower().endswith(suffixes)]
    if not members:
        raise ValueError("Archive does not contain a supported data file")
    return sorted(members)[0]


def _find_first_column(header: list[str], candidates: list[str]) -> int | None:
    normalized = [item.strip() for item in header]
    for candidate in candidates:
        if candidate in normalized:
            return normalized.index(candidate)
    return None


def _find_column_index(header: list[str], column_name: str | None) -> int | None:
    if column_name is None:
        return None
    normalized = [item.strip().strip('"') for item in header]
    target = column_name.strip().strip('"')
    if target in normalized:
        return normalized.index(target)
    lowered = [item.lower() for item in normalized]
    target_lower = target.lower()
    if target_lower in lowered:
        return lowered.index(target_lower)
    return None


def _label_to_int(label: str, label_lookup: dict[str, int]) -> int:
    key = label.strip()
    if not key:
        return 0
    if key not in label_lookup:
        label_lookup[key] = len(label_lookup)
    return label_lookup[key]


def _first_numeric_column(header: list[str]) -> int:
    for index, value in enumerate(header):
        if value is None:
            continue
        if value.strip().lower() in {"date", "time"}:
            continue
        return index
    return 0


def _parse_numeric_tokens(line: str) -> list[float]:
    tokens = [token for token in re.split(r"[,\s;]+", line.strip()) if token]
    parsed: list[float] = []
    for token in tokens:
        if ":" in token:
            token = token.split(":", 1)[1]
        value = _parse_float(token)
        if value is not None:
            parsed.append(value)
    return parsed


def _parse_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", ".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _batch_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", Path(name).stem)
    return (int(match.group(1)) if match else 0, name)


def build_stream_signal(
    dataset: TimeSeriesDataset,
    series_index: int = 0,
    series_count: int = 20,
    balanced: bool = False,
) -> StreamSignal:
    """РЎРєР»РµРёС‚СЊ РЅРµСЃРєРѕР»СЊРєРѕ СЃС‚СЂРѕРє ECG200 РІ РѕРґРёРЅ Р±РѕР»РµРµ РґР»РёРЅРЅС‹Р№ РїРѕС‚РѕРє.

    Р’ ECG200 РєР°Р¶РґР°СЏ СЃРµСЂРёСЏ РєРѕСЂРѕС‚РєР°СЏ Рё С„РёРєСЃРёСЂРѕРІР°РЅРЅРѕР№ РґР»РёРЅС‹. РЎРєР»РµР№РєР° РґР°С‘С‚ РґРѕСЃС‚Р°С‚РѕС‡РЅРѕ
    РѕРєРѕРЅ РґР»СЏ РЅР°РіР»СЏРґРЅРѕР№ РґРµРјРѕРЅСЃС‚СЂР°С†РёРё РїРѕС‚РѕРєРѕРІРѕР№ РѕР±СЂР°Р±РѕС‚РєРё, РїСЂРё СЌС‚РѕРј РїРѕСЂСЏРґРѕРє СЃСЌРјРїР»РѕРІ
    РІРЅСѓС‚СЂРё РєР°Р¶РґРѕР№ СЃС‚СЂРѕРєРё РЅРµ РЅР°СЂСѓС€Р°РµС‚СЃСЏ.
    """
    if series_index < 0 or series_index >= len(dataset.series):
        raise IndexError(f"series_index={series_index} is outside dataset size {len(dataset.series)}")

    if balanced:
        selected_indices = _balanced_indices(dataset.labels, series_index, max(1, series_count))
    else:
        end = min(series_index + max(1, series_count), len(dataset.series))
        selected_indices = list(range(series_index, end))

    selected_series = dataset.series[selected_indices]
    selected_labels = dataset.labels[selected_indices]

    series_length = dataset.series.shape[1]
    signal = selected_series.reshape(-1)
    source_labels = np.repeat(selected_labels, series_length)
    series_ranges = [
        (offset * series_length, (offset + 1) * series_length)
        for offset in range(len(selected_series))
    ]
    return StreamSignal(signal=signal, source_labels=source_labels, series_ranges=series_ranges)


def _balanced_indices(labels: np.ndarray, start_index: int, count: int) -> list[int]:
    available = np.arange(len(labels))[start_index:]
    buckets = {
        label: list(available[labels[available] == label])
        for label in sorted(np.unique(labels[available]))
    }

    selected: list[int] = []
    while len(selected) < count:
        before_round = len(selected)
        for label in buckets:
            if buckets[label] and len(selected) < count:
                selected.append(buckets[label].pop(0))
        if len(selected) == before_round:
            break
    return selected


