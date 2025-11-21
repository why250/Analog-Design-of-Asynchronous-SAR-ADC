"""
Spectrum analysis utilities tailored for CSV waveforms.

This module generalizes `spectrum_analysis.py` by wrapping the FFT workflow
into a reusable class that:
1. Loads time-domain samples from a CSV file.
2. Computes the single-sided amplitude spectrum via FFT.
3. Detects prominent spectral peaks.
4. Provides helper plotting and reporting methods.

Example CSV format (first row as header):
time,voltage
0.0000,0.00
0.0010,0.65
...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class SpectrumAnalyzerConfig:
    csv_path: Path
    time_column: str = "time"
    value_column: str = "signal"
    sampling_rate_hz: Optional[float] = None
    prominence: float = 0.5


@dataclass
class SpectrumAnalyzer:
    config: SpectrumAnalyzerConfig
    _time: np.ndarray = field(init=False, repr=False)
    _signal: np.ndarray = field(init=False, repr=False)
    _freq: np.ndarray = field(init=False, repr=False)
    _amplitude: np.ndarray = field(init=False, repr=False)
    _peaks: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_csv()
        self._validate_signal()
        self._derive_sampling_rate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the single-sided amplitude spectrum."""
        y_fft = np.fft.fft(self._signal)
        l = len(self._signal)
        p2 = np.abs(y_fft / l)
        p1 = p2[: l // 2 + 1]
        if l > 2:
            p1[1:-1] *= 2

        f = self.config.sampling_rate_hz * np.arange(0, l // 2 + 1) / l

        self._freq = f
        self._amplitude = p1
        return f, p1

    def detect_peaks(self) -> np.ndarray:
        """Run prominence-based peak detection on the latest amplitude spectrum."""
        if not hasattr(self, "_amplitude"):
            self.analyze()
        peaks, _ = find_peaks(self._amplitude, prominence=self.config.prominence)
        self._peaks = peaks
        return peaks

    def summary(self) -> Iterable[str]:
        """Yield formatted lines summarizing the detected peaks."""
        if not hasattr(self, "_peaks"):
            self.detect_peaks()
        if self._peaks.size == 0:
            yield "未检测到符合 prominence 条件的峰值。"
            return
        yield "检测到的主导频率:"
        yield "-" * 30
        for idx in self._peaks:
            yield f"频率 = {self._freq[idx]:8.2f} Hz | 幅度 = {self._amplitude[idx]:.3f}"

    def plot(self, show: bool = True) -> None:
        """Render time-domain waveform and amplitude spectrum."""
        if not hasattr(self, "_amplitude"):
            self.analyze()
        if not hasattr(self, "_peaks"):
            self.detect_peaks()

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self._time, self._signal)
        plt.title("Time-Domain Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self._freq, self._amplitude, label="Single-Sided Spectrum")
        plt.plot(
            self._freq[self._peaks],
            self._amplitude[self._peaks],
            "x",
            color="red",
            markersize=10,
            label="Detected Peaks",
        )
        plt.title("Frequency-Domain Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|P1(f)|")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        image_path = self.config.csv_path.with_suffix(".png")
        plt.savefig(image_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_csv(self) -> None:
        """Load CSV columns into NumPy arrays."""
        df = pd.read_csv(self.config.csv_path)
        missing = {self.config.time_column, self.config.value_column} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV 缺少必需列: {', '.join(sorted(missing))}. "
                f"请检查 {self.config.csv_path}。"
            )
        self._time = df[self.config.time_column].to_numpy(dtype=float)
        self._signal = df[self.config.value_column].to_numpy(dtype=float)

    def _validate_signal(self) -> None:
        if self._time.ndim != 1 or self._signal.ndim != 1:
            raise ValueError("时间或信号列不是一维数据。")
        if len(self._time) != len(self._signal):
            raise ValueError("时间样本数量与信号样本数量不一致。")
        if len(self._time) < 4:
            raise ValueError("样本数太少，无法执行可靠的频谱分析。")

    def _derive_sampling_rate(self) -> None:
        if self.config.sampling_rate_hz is not None:
            return
        dt = np.diff(self._time)
        if not np.allclose(dt, dt[0], rtol=1e-3, atol=1e-6):
            raise ValueError("时间轴不是等间隔采样，请显式提供 sampling_rate_hz。")
        inferred_fs = 1.0 / np.mean(dt)
        self.config.sampling_rate_hz = inferred_fs


def _demo() -> None:
    """Minimal CLI demo for manual experimentation."""
    import argparse

    parser = argparse.ArgumentParser(description="CSV 频谱分析")
    parser.add_argument("csv", type=Path, help="输入 CSV 文件路径")
    parser.add_argument("--time-col", default="time", help="时间列名")
    parser.add_argument("--value-col", default="signal", help="信号列名")
    parser.add_argument("--fs", type=float, default=None, help="采样频率 (Hz)")
    parser.add_argument("--prominence", type=float, default=0.5, help="峰值 prominence 阈值")
    parser.add_argument("--no-plot", action="store_true", help="不显示图形，仅打印结果")
    args = parser.parse_args()

    analyzer = SpectrumAnalyzer(
        SpectrumAnalyzerConfig(
            csv_path=args.csv,
            time_column=args.time_col,
            value_column=args.value_col,
            sampling_rate_hz=args.fs,
            prominence=args.prominence,
        )
    )
    analyzer.analyze()
    analyzer.detect_peaks()

    for line in analyzer.summary():
        print(line)

    if not args.no_plot:
        analyzer.plot()


if __name__ == "__main__":
    _demo()

