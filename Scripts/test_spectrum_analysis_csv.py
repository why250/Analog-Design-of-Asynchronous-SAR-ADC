"""
Quick functional test for `spectrum_analysis_csv.py`.

Steps:
1. Generate a synthetic time-domain signal (two tones + noise).
2. Save it as a CSV file with `time`/`signal` columns.
3. Invoke `SpectrumAnalyzer` to compute and plot the spectrum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectrum_analysis_csv import SpectrumAnalyzer, SpectrumAnalyzerConfig


def generate_signal(fs: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a noisy two-tone waveform."""
    t = np.arange(0, duration, 1.0 / fs)
    s = 0.7 * np.sin(2 * np.pi * 50 * t) + 1.0 * np.sin(2 * np.pi * 120 * t)
    noise = 1.5 * np.random.randn(t.size)
    return t, s + noise


def save_csv(path: Path, time: np.ndarray, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"time": time, "signal": signal})
    df.to_csv(path, index=False)
    print(f"CSV saved to {path.resolve()}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fs = 1000.0
    duration = 1.5  # seconds
    csv_path = base_dir / "sample_signal.csv"

    t, x = generate_signal(fs, duration)
    save_csv(csv_path, t, x)

    analyzer = SpectrumAnalyzer(
        SpectrumAnalyzerConfig(
            csv_path=csv_path,
            time_column="time",
            value_column="signal",
            sampling_rate_hz=fs,
            prominence=0.5,
        )
    )
    analyzer.analyze()
    analyzer.detect_peaks()

    for line in analyzer.summary():
        print(line)

    analyzer.plot()


if __name__ == "__main__":
    main()

