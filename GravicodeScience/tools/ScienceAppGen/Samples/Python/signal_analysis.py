"""Find the heartbeat in a noisy sensor trace.

A single-file example for ScienceAppGen's Python to C# converter: numpy for the signal, scipy for
filtering, spectral analysis and peak finding, matplotlib for the figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

FS = 250.0          # samples per second
DURATION = 10.0     # seconds


def make_trace(seed=7):
    """A 72 bpm pulse train with baseline wander and white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, DURATION, 1 / FS)
    beat_hz = 72 / 60
    pulse = np.sin(2 * np.pi * beat_hz * t) ** 63          # sharp peaks once per beat
    wander = 0.4 * np.sin(2 * np.pi * 0.25 * t)
    noise = 0.15 * rng.standard_normal(t.size)
    return t, pulse + wander + noise


def clean(trace):
    """Band-pass 0.5-15 Hz with a zero-phase 4th-order Butterworth."""
    b, a = signal.butter(4, [0.5, 15], btype="bandpass", fs=FS)
    return signal.filtfilt(b, a, trace)


def heart_rate(filtered):
    peaks, _ = signal.find_peaks(filtered, height=0.5, distance=FS * 0.4)
    intervals = np.diff(peaks) / FS
    return peaks, 60.0 / intervals.mean(), intervals


def main():
    t, raw = make_trace()
    filtered = clean(raw)
    peaks, bpm, intervals = heart_rate(filtered)

    freqs, power = signal.welch(filtered, fs=FS, nperseg=1024)
    dominant = freqs[np.argmax(power)]

    print(f"samples:           {raw.size}")
    print(f"beats found:       {peaks.size}")
    print(f"heart rate:        {bpm:.1f} bpm")
    print(f"dominant freq:     {dominant:.2f} Hz")
    lo, hi = stats.norm.interval(0.95, loc=intervals.mean(), scale=intervals.std(ddof=1))
    print(f"interval 95% band: {lo:.3f} - {hi:.3f} s")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, raw, color="0.6", label="raw")
    axes[0].plot(t, filtered, color="tab:red", label="filtered")
    axes[0].scatter(t[peaks], filtered[peaks], color="black", label="beats")
    axes[0].set_xlim(0, 4)
    axes[0].set_title("Sensor trace")
    axes[0].set_xlabel("time (s)")
    axes[0].legend()

    axes[1].semilogy(freqs, power)
    axes[1].axvline(dominant, color="tab:red", linestyle="--")
    axes[1].set_xlim(0, 20)
    axes[1].set_title("Power spectral density")
    axes[1].set_xlabel("frequency (Hz)")

    fig.tight_layout()
    fig.savefig("heartbeat.png", dpi=100)
    print("saved heartbeat.png")


if __name__ == "__main__":
    main()
