"""Hardware-free tests for the signal processing helpers."""
import numpy as np

import sdr_mcp as s


def test_band_of_prefers_most_specific():
    assert s.band_of(100.1) == "FM broadcast (WFM)"
    assert s.band_of(926).split(" | ")[0].startswith("ISM 915") and "GSM/LTE 900 downlink" in s.band_of(926)
    assert s.band_of(433.92).startswith("ISM 433")
    assert s.band_of(145.8).startswith("amateur 2 m")
    assert s.band_of(5000) == "unknown"


def test_parse_rtl_power_splits_sweeps_on_wraparound():
    rows = []
    for t in ("10:00:00", "10:00:05"):
        rows.append(f"2026-01-01, {t}, 100000000, 101000000, 250000, 1, 1.0, 2.0, 3.0, 4.0")
        rows.append(f"2026-01-01, {t}, 101000000, 102000000, 250000, 1, 5.0, 6.0, 7.0, 8.0")
    sweeps = s.parse_rtl_power("\n".join(rows))
    assert len(sweeps) == 2
    _, f, p = sweeps[0]
    assert len(f) == 8 and f[0] == 100e6 and p[-1] == 8.0


def test_find_peaks_detects_carriers():
    f = np.linspace(400e6, 410e6, 2000)
    p = np.random.default_rng(0).normal(-30, 0.5, f.size)
    p[500:505] += 25
    p[1500] += 15
    peaks, _ = s.find_peaks(f, p, thresh_db=7)
    found = sorted(round(x["peak_mhz"], 1) for x in peaks)
    assert found == [402.5, 407.5]
    assert peaks[0]["snr_db"] > peaks[1]["snr_db"]


def test_analyze_iq_measures_tone_offset_and_bursts():
    fs, center, tune = 2.048e6, 434.0e6, 434.0e6 - 512e3
    t = np.arange(int(fs * 1.0)) / fs
    tone = np.exp(2j * np.pi * (512e3 + 20e3) * t)  # +20 kHz from center
    gate = ((t % 0.25) < 0.1).astype(float)          # 4 bursts, 40 % duty
    rng = np.random.default_rng(1)
    iq = 20 * tone * gate + rng.normal(0, 1, t.size) + 1j * rng.normal(0, 1, t.size)
    _, _, _, st = s.analyze_iq(iq, fs, tune, center, 200e3)
    assert abs(st["strongest_offset_khz"] - 20) < 2
    assert 0.3 < st["active_time_fraction"] < 0.5
    assert st["bursts_count"] == 4


def test_activity_channels_flags_intermittent():
    f = np.linspace(433e6, 435e6, 400)
    M = np.full((20, 400), -40.0)
    M[:, 100] += 30          # always on
    M[3, 300] += 20          # one burst
    M[11, 300] += 20         # another burst
    chans, _ = s.activity_channels(f, M, [f"t{i}" for i in range(20)], 8)
    kinds = {round(c["freq_mhz"], 2): c["kind"] for c in chans}
    assert kinds[round(f[300] / 1e6, 2)] == "intermittent"
    assert kinds[round(f[100] / 1e6, 2)] == "continuous"


def test_write_wav_decimates(tmp_path):
    audio = np.sin(np.arange(32000) / 10) * 1000
    a, rate = s.write_wav(tmp_path / "x.wav", audio, 32000)
    assert rate == 16000 and len(a) == 16000
    assert (tmp_path / "x.wav").stat().st_size > 32000


def test_server_registers_tools():
    import asyncio
    mcp = s.build_server()
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"sdr_info", "spectrum_sweep", "waterfall", "listen", "activity_monitor",
            "decode_devices", "satellite_passes"} <= names
