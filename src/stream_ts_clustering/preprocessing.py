from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowSegment:
    index: int
    start: int
    end: int
    center: int
    values: np.ndarray
    feature: np.ndarray


class FeatureScaler:
    """Fixed feature scaler for streaming clustering.

    The scaler maps raw segment features into one stable distance space. Earlier
    versions mixed fixed scaling with online z-normalization, but this changed the
    coordinate system while microclusters had already been created. The current
    implementation deliberately keeps the transform fixed during the whole run so
    CluStream and DenStream compare all committed segments in the same feature
    space.
    """

    def __init__(
        self,
        n_features: int,
        clip_value: float = 5.0,
    ) -> None:
        self.n_features = n_features
        self.clip_value = float(clip_value)
        self.feature_weights = self._default_feature_weights(n_features)
        self.prior_scale = self._default_prior_scale(n_features)

    def fit_transform(self, features: list[np.ndarray]) -> list[np.ndarray]:
        if not features:
            return []
        matrix = np.vstack([self._pretransform(np.asarray(feature, dtype=float)) for feature in features])
        if matrix.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {matrix.shape[1]}")
        fixed_scaled = self._fixed_scale_matrix(matrix)
        return [row.astype(float, copy=True) for row in fixed_scaled]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform a feature vector without updating any runtime statistics."""
        x = np.asarray(x, dtype=float)
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
        shaped = self._pretransform(x)
        return self._fixed_scale_vector(shaped).astype(float, copy=True)

    def transform_update(self, x: np.ndarray) -> np.ndarray:
        """Transform a committed segment feature in the same fixed space."""
        return self.transform(x)

    def _fixed_scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        scaled = matrix / self.prior_scale
        scaled = np.clip(scaled, -self.clip_value, self.clip_value)
        return scaled * self.feature_weights

    def _fixed_scale_vector(self, x: np.ndarray) -> np.ndarray:
        scaled = x / self.prior_scale
        scaled = np.clip(scaled, -self.clip_value, self.clip_value)
        return scaled * self.feature_weights

    def _pretransform(self, x: np.ndarray) -> np.ndarray:
        transformed = x.astype(float, copy=True)

        # Basic/regime profile layout: 0..6 are level, robust-scale, trend, and
        # volatility features. Signed log compression keeps large real-world levels
        # from dominating while preserving mean shifts.
        for index in range(min(self.n_features, 7)):
            transformed[index] = np.sign(transformed[index]) * np.log1p(abs(transformed[index]))

        # Bounded 0..1 features are mapped to a symmetric range. Autocorrelation at
        # index 8 is already in [-1, 1] and is intentionally left unchanged.
        for index in (7, 9, 10):
            if index < self.n_features:
                transformed[index] = 2.0 * np.clip(transformed[index], 0.0, 1.0) - 1.0

        return np.nan_to_num(transformed, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)

    @staticmethod
    def _default_feature_weights(n_features: int) -> np.ndarray:
        if n_features == 11:
            return np.array(
                [
                    0.82,  # mean
                    1.85,  # std
                    0.46,  # median
                    1.80,  # IQR
                    1.35,  # robust range
                    0.36,  # slope
                    1.85,  # diff std
                    0.34,  # zero crossings
                    0.42,  # autocorr
                    0.22,  # high frequency ratio
                    1.25,  # roughness / volatility ratio
                ],
                dtype=float,
            )
        return np.ones(n_features, dtype=float)

    @staticmethod
    def _default_prior_scale(n_features: int) -> np.ndarray:
        if n_features == 11:
            return np.array(
                [
                    1.40,
                    1.10,
                    1.40,
                    0.95,
                    1.20,
                    0.75,
                    1.05,
                    1.00,
                    1.00,
                    1.00,
                    1.00,
                ],
                dtype=float,
            )
        return np.ones(n_features, dtype=float)

def extract_features(window: np.ndarray) -> np.ndarray:
    return extract_features_with_profile(window)


def extract_features_with_profile(window: np.ndarray) -> np.ndarray:
    """Extract the default compact feature set from a univariate window."""
    values = np.asarray(window, dtype=float)
    if len(values) == 0:
        raise ValueError("window must contain at least one value")

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    diffs = np.diff(values)
    x_axis = np.arange(len(values), dtype=float)
    mean = float(values.mean())
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(q75 - q25)
    std = float(values.std())
    minimum = float(values.min())
    maximum = float(values.max())
    value_range = maximum - minimum
    q10, q90 = np.percentile(values, [10.0, 90.0])
    robust_range = float(q90 - q10)
    mad = float(np.median(np.abs(values - median)))
    centered = values - mean
    safe_std = std if std > 1e-8 else 1.0

    slope = 0.0
    if len(values) > 1:
        centered_x = x_axis - x_axis.mean()
        denom = float(np.dot(centered_x, centered_x))
        if denom > 0:
            slope = float(np.dot(centered_x, centered) / denom)

    diff_mean = float(diffs.mean()) if len(diffs) else 0.0
    diff_std = float(diffs.std()) if len(diffs) else 0.0
    zero_crossings = 0.0
    if len(centered) > 1:
        zero_crossings = float(np.sum(np.diff(np.signbit(centered)) != 0)) / max(1, len(centered) - 1)
    autocorr_lag_1 = _autocorrelation(centered / safe_std, lag=1)
    dominant_frequency, high_frequency_ratio, spectral_entropy = _spectral_features(centered / safe_std)
    roughness_ratio = float(np.tanh(diff_std / max(std, 1e-6))) if len(diffs) else 0.0

    return np.array(
        [
            mean,
            std,
            median,
            max(iqr, mad * 1.4826),
            robust_range,
            slope,
            diff_std,
            zero_crossings,
            autocorr_lag_1,
            high_frequency_ratio,
            roughness_ratio,
        ],
        dtype=float,
    )


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _spectral_features(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) < 4:
        return 0.0, 0.0, 0.0

    spectrum = np.abs(np.fft.rfft(values)) ** 2
    if len(spectrum) <= 1:
        return 0.0, 0.0, 0.0
    spectrum = spectrum[1:]
    total_energy = float(spectrum.sum())
    if total_energy <= 1e-12:
        return 0.0, 0.0, 0.0

    dominant_index = int(np.argmax(spectrum)) + 1
    dominant_frequency = dominant_index / max(1, len(values))
    split = max(1, int(np.ceil(len(spectrum) * 2.0 / 3.0)))
    high_frequency_ratio = float(spectrum[split:].sum() / total_energy) if split < len(spectrum) else 0.0
    probabilities = spectrum / total_energy
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    spectral_entropy = entropy / max(np.log(len(probabilities) + 1e-12), 1e-12)
    return dominant_frequency, high_frequency_ratio, spectral_entropy

