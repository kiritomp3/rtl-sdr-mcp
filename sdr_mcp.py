#!/usr/bin/env python3
"""rtl-sdr-mcp: give an LLM agent a real radio.

An MCP server that exposes an RTL-SDR dongle as tools: spectrum sweeps, waterfalls
(returned as images the model can look at), demodulated audio with speech-to-text,
activity monitoring over time, ISM device decoding and satellite pass prediction.

Run:  sdr-mcp            (stdio MCP server)
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- configuration

DATA = Path(os.environ.get("SDR_DATA_DIR", Path.home() / "rtl-sdr-mcp-data")).expanduser()
DEVICE = os.environ.get("SDR_DEVICE", "0")
WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", Path.home() / ".cache/whisper/ggml-small.bin")).expanduser()
LAT = os.environ.get("SDR_LAT")
LON = os.environ.get("SDR_LON")

# Approximate usable ranges (MHz) for common RTL-SDR tuners.
TUNER_RANGES = {
    "R820T": (24, 1766), "R828D": (24, 1766), "R860": (24, 1766),
    "E4000": (52, 2200), "FC0013": (22, 1100), "FC0012": (22, 948), "FC2580": (146, 924),
}

# Coarse band plan (ITU Region 1 flavoured, with a few worldwide allocations).
BANDS = [
    (26.96, 27.41, "CB radio 27 MHz (AM/FM)"), (28.0, 29.7, "amateur 10 m"),
    (50.0, 54.0, "amateur 6 m"), (65.9, 74.0, "OIRT FM broadcast (legacy)"),
    (87.5, 108.0, "FM broadcast (WFM)"), (108.0, 118.0, "aviation navigation VOR/ILS"),
    (118.0, 137.0, "aviation voice (AM)"), (137.0, 138.0, "weather satellites (Meteor-M LRPT 137.9)"),
    (144.0, 148.0, "amateur 2 m (FM, APRS 144.39/144.80, ISS 145.80)"), (148.0, 156.0, "land mobile (NFM)"),
    (156.0, 162.1, "marine VHF (NFM)"), (162.4, 162.55, "NOAA weather radio (US)"),
    (162.1, 174.0, "land mobile, railway, taxi (NFM)"), (174.0, 230.0, "DAB / TV band III"),
    (380.0, 400.0, "TETRA / public safety (digital)"), (400.0, 406.0, "radiosondes (weather balloons)"),
    (406.0, 406.1, "COSPAS-SARSAT distress beacons"), (420.0, 450.0, "amateur 70 cm"),
    (433.05, 434.79, "ISM 433: remotes, sensors, LPD radios"), (446.0, 446.2, "PMR446 walkie-talkies"),
    (450.0, 470.0, "land mobile / UHF business"), (470.0, 790.0, "digital TV (DVB-T/ATSC)"),
    (791.0, 821.0, "LTE 800 downlink"), (832.0, 862.0, "LTE 800 uplink"),
    (863.0, 870.0, "ISM 868: LoRa, alarms, smart home"), (902.0, 928.0, "ISM 915 (Americas): LoRa, meters"),
    (880.0, 915.0, "GSM/LTE 900 uplink"), (925.0, 960.0, "GSM/LTE 900 downlink"),
    (1087.0, 1093.0, "ADS-B aircraft (1090)"), (1215.0, 1300.0, "amateur 23 cm / GNSS L2"),
    (1525.0, 1559.0, "Inmarsat L-band"), (1559.0, 1610.0, "GNSS L1 (GPS/GLONASS/Galileo)"),
]

# ---------------------------------------------------------------- helpers (pure, testable)


def band_of(mhz: float) -> str:
    """Known allocations for a frequency, most specific first (regions differ, so all are listed)."""
    hits = sorted((b for b in BANDS if b[0] <= mhz <= b[1]), key=lambda b: b[1] - b[0])
    return " | ".join(b[2] for b in hits) if hits else "unknown"


def parse_rtl_power(text: str):
    """Parse rtl_power CSV into a list of sweeps: (timestamp, freqs_hz, power_db).
    A new sweep starts whenever the hop frequency wraps around."""
    groups, prev_lo = [], None
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 7:
            continue
        lo, step = float(r[2]), float(r[4])
        if prev_lo is None or lo <= prev_lo:
            groups.append(((r[0] + r[1]).strip(), []))
        prev_lo = lo
        vals = [float(x) for x in r[6:] if x.strip() not in ("", "nan", "-nan")]
        groups[-1][1].extend((lo + i * step, v) for i, v in enumerate(vals))
    sweeps = []
    for key, pts in groups:
        if pts:
            arr = np.array(sorted(pts))
            sweeps.append((key, arr[:, 0], arr[:, 1]))
    if len(sweeps) > 1 and len(sweeps[-1][1]) < 0.9 * len(sweeps[0][1]):
        sweeps.pop()  # incomplete last sweep
    return sweeps


def noise_floor(p: np.ndarray) -> np.ndarray:
    """Rolling median noise floor."""
    w = max(30, len(p) // 40)
    return np.array([np.median(p[max(0, i - w): i + w]) for i in range(len(p))])


def find_peaks(f: np.ndarray, p: np.ndarray, thresh_db: float = 7, max_n: int = 30):
    """Group contiguous bins above the local noise floor into signals."""
    floor = noise_floor(p)
    ex = p - floor
    hot = ex > thresh_db
    out, i = [], 0
    while i < len(f):
        if hot[i]:
            j = i
            while j + 1 < len(f) and hot[j + 1]:
                j += 1
            k = i + int(np.argmax(ex[i:j + 1]))
            out.append({"peak_mhz": round(f[k] / 1e6, 4), "from_mhz": round(f[i] / 1e6, 4),
                        "to_mhz": round(f[j] / 1e6, 4), "snr_db": round(float(ex[k]), 1),
                        "band": band_of(f[k] / 1e6)})
            i = j + 1
        else:
            i += 1
    out.sort(key=lambda x: -x["snr_db"])
    return out[:max_n], floor


def analyze_iq(iq: np.ndarray, fs: float, tune_hz: float, center_hz: float, span_hz: float, rows: int = 300):
    """Spectrogram of complex IQ cropped to [center-span/2, center+span/2] plus signal statistics."""
    nfft = 1 << int(np.ceil(np.log2(fs / (span_hz / 400))))
    nfft = min(max(nfft, 256), 65536)
    frames = len(iq) // nfft
    if frames < 2:
        raise ValueError("not enough samples for a spectrogram")
    x = iq[: frames * nfft].reshape(frames, nfft) * np.hanning(nfft)
    S = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
    freqs = tune_hz + np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    sel = (freqs >= center_hz - span_hz / 2) & (freqs <= center_hz + span_hz / 2)
    S, freqs = S[:, sel], freqs[sel]
    k = max(1, frames // rows)  # frames averaged per displayed row
    rows = frames // k
    S = S[: rows * k].reshape(rows, k, -1).mean(axis=1)
    db = 10 * np.log10(S + 1e-12)
    noise = float(np.median(db))
    avg = db.mean(axis=0)
    peak_i = int(np.argmax(avg))
    peak_snr = float(avg[peak_i] - noise)
    lin = np.clip(10 ** ((avg - noise) / 10) - 1, 0, None)
    occ = None
    if lin.sum() > 0 and peak_snr > 3:
        c = np.cumsum(lin) / lin.sum()
        lo = freqs[np.searchsorted(c, 0.005)]
        hi = freqs[min(np.searchsorted(c, 0.995), len(freqs) - 1)]
        occ = round((hi - lo) / 1e3, 1)
    chan = db[:, max(0, peak_i - 3): peak_i + 4].mean(axis=1)
    active = chan > noise + 6
    duration = rows * k * nfft / fs
    stats = {
        "duration_s": round(duration, 2),
        "resolution_hz": round(fs / nfft, 1),
        "strongest_offset_khz": round((freqs[peak_i] - center_hz) / 1e3, 2),
        "strongest_snr_db_avg": round(peak_snr, 1),
        "occupied_bw_khz_99pct": occ,
        "active_time_fraction": round(float(active.mean()), 2),
        "bursts_count": int(np.sum(np.diff(active.astype(int)) == 1)) + int(active[0]),
    }
    return db, freqs, noise, stats


def activity_channels(f: np.ndarray, M: np.ndarray, times: list[str], thresh_db: float):
    """Per-channel activity from a (sweeps x bins) matrix of power."""
    base = float(np.median(M))
    ex = M - base
    act = ex > thresh_db
    frac = act.mean(axis=0)
    chans, i, n = [], 0, len(f)
    while i < n:
        if frac[i] > 0:
            j = i
            while j + 1 < n and frac[j + 1] > 0:
                j += 1
            k = i + int(np.argmax(ex[:, i:j + 1].max(axis=0)))
            seen = [times[t] for t in range(len(times)) if act[t, k]]
            chans.append({"freq_mhz": round(f[k] / 1e6, 4), "band": band_of(f[k] / 1e6),
                          "active_fraction": round(float(frac[k]), 2), "max_snr_db": round(float(ex[:, k].max()), 1),
                          "kind": "continuous" if frac[k] > 0.9 else "intermittent",
                          "first_seen": seen[0] if seen else None, "last_seen": seen[-1] if seen else None})
            i = j + 1
        else:
            i += 1
    chans.sort(key=lambda c: (c["kind"] != "intermittent", -c["max_snr_db"]))
    return chans, ex


def write_wav(path: Path, audio: np.ndarray, rate: int, out_rate: int = 16000):
    """Write int16 mono audio, decimating by an integer factor to out_rate."""
    factor = max(1, rate // out_rate)
    if factor > 1:
        n = len(audio) // factor * factor
        audio = audio[:n].reshape(-1, factor).mean(axis=1)
    a = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate // factor)
        w.writeframes(a.tobytes())
    return a.astype(np.float32), rate // factor


# ---------------------------------------------------------------- hardware layer


def tool_path(name: str) -> str:
    p = shutil.which(name)
    if not p:
        hints = {"rtl_433": "brew install rtl_433 / apt install rtl-433",
                 "whisper-cli": "brew install whisper-cpp (or build whisper.cpp)"}
        raise RuntimeError(f"`{name}` not found in PATH. Install: "
                           + hints.get(name, "brew install librtlsdr / apt install rtl-sdr"))
    return p


_HW: dict | None = None


def hardware() -> dict:
    """Detect tuner and gain table once via `rtl_test -t`."""
    global _HW
    if _HW is not None:
        return _HW
    hw = {"tuner": "unknown", "gains": [], "range_mhz": [24, 1766]}
    try:
        r = subprocess.run([tool_path("rtl_test"), "-d", DEVICE, "-t"], capture_output=True, text=True, timeout=15)
        out = r.stdout + r.stderr
        m = re.search(r"Found (.+?) tuner", out)
        if m:
            hw["tuner"] = m.group(1).strip()
        m = re.search(r"Supported gain values \(\d+\):([\d.\s-]+)", out)
        if m:
            hw["gains"] = [float(x) for x in m.group(1).split()]
        m = re.search(r"Using device \d+: (.+)", out)
        hw["device"] = m.group(1).strip() if m else None
        if "No supported devices found" in out or "No matching devices" in out:
            hw["error"] = "no RTL-SDR device found"
    except Exception as e:  # pragma: no cover - hardware dependent
        hw["error"] = str(e)
    for key, rng in TUNER_RANGES.items():
        if key.lower() in hw["tuner"].lower().replace(" ", ""):
            hw["range_mhz"] = list(rng)
    gain_env = os.environ.get("SDR_GAIN")
    hw["gain"] = float(gain_env) if gain_env else (max(hw["gains"]) if hw["gains"] else 40.0)
    _HW = hw
    return hw


def gain() -> str:
    return str(hardware()["gain"])


def check_freq(*mhz: float):
    lo, hi = hardware()["range_mhz"]
    for f in mhz:
        if not lo <= f <= hi:
            raise ValueError(f"{f} MHz is outside this tuner's range ({lo}-{hi} MHz)")


def run_for(cmd, seconds, stdout=None):
    p = subprocess.Popen(cmd, stdout=stdout or subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    time.sleep(0.3)


def rtl_power(start, stop, bin_hz, integ, total=None):
    DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir=DATA) as tf:
        path = tf.name
    cmd = [tool_path("rtl_power"), "-d", DEVICE, "-f", f"{start}M:{stop}M:{int(bin_hz)}", "-g", gain(), "-i", str(integ)]
    cmd += ["-e", f"{int(total)}s"] if total else ["-1"]
    try:
        subprocess.run(cmd + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=(total or 0) + 240)
        return parse_rtl_power(Path(path).read_text())
    finally:
        os.unlink(path)


def capture_iq(center_hz, fs, seconds):
    tune = center_hz - fs / 4  # keep the DC spike away from the signal of interest
    DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".u8", delete=False, dir=DATA) as tf:
        path = tf.name
    try:
        subprocess.run([tool_path("rtl_sdr"), "-d", DEVICE, "-f", str(int(tune)), "-s", str(int(fs)), "-g", gain(),
                        "-n", str(int(fs * seconds)), path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=seconds + 30)
        raw = np.fromfile(path, dtype=np.uint8).astype(np.float32)
    finally:
        os.unlink(path)
    return (raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5), tune


# ---------------------------------------------------------------- plotting


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _dark(fig, ax):
    fig.patch.set_facecolor("#0e1116")
    ax.set_facecolor("#0e1116")
    ax.tick_params(colors="#aab", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.xaxis.label.set_color("#aab")
    ax.yaxis.label.set_color("#aab")
    ax.title.set_color("#dde")


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    _plt().close(fig)
    return buf.getvalue()


def plot_spectrum(f, p, floor, peaks, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    _dark(fig, ax)
    ax.plot(f / 1e6, p, color="#5ad1ff", lw=0.7)
    ax.plot(f / 1e6, floor, color="#555", lw=0.6, ls="--")
    for pk in peaks[:12]:
        y = p[np.argmin(abs(f / 1e6 - pk["peak_mhz"]))]
        ax.annotate(f"{pk['peak_mhz']:.3f}", (pk["peak_mhz"], y), color="#ffd166", fontsize=7, ha="center",
                    xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("MHz")
    ax.set_ylabel("dB")
    ax.set_title(title)
    ax.grid(color="#222", lw=0.5)
    fig.tight_layout()
    return _png(fig)


def plot_waterfall(db, freqs, noise, center_hz, duration, peak_snr, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    _dark(fig, ax)
    ax.imshow(db, aspect="auto", cmap="inferno", vmin=noise - 3, vmax=noise + max(25, peak_snr),
              extent=[(freqs[0] - center_hz) / 1e3, (freqs[-1] - center_hz) / 1e3, duration, 0])
    ax.set_xlabel(f"offset from {center_hz / 1e6:g} MHz, kHz")
    ax.set_ylabel("time, s")
    ax.set_title(title)
    fig.tight_layout()
    return _png(fig)


def plot_audio(a, rate, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    _dark(fig, ax)
    ax.specgram(a, NFFT=512, Fs=rate, noverlap=256, cmap="magma")
    ax.set_ylim(0, rate / 2)
    ax.set_xlabel("time, s")
    ax.set_ylabel("Hz")
    ax.set_title(title)
    fig.tight_layout()
    return _png(fig)


def plot_activity(ex, f, n_sweeps, thresh_db, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    _dark(fig, ax)
    ax.imshow(ex, aspect="auto", cmap="viridis", vmin=0, vmax=max(20, thresh_db * 2),
              extent=[f[0] / 1e6, f[-1] / 1e6, n_sweeps, 0])
    ax.set_xlabel("MHz")
    ax.set_ylabel("sweep # (~5 s each)")
    ax.set_title(title)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------- MCP server

INSTRUCTIONS = """You control a real RTL-SDR radio receiver (receive only). Call sdr_info first to learn the
tuner, its frequency range and gain. Only one tool can use the radio at a time, calls are serialized.
A good workflow: survey (spectrum_sweep) -> watch over time (activity_monitor) -> inspect the signal shape
(waterfall) -> demodulate, listen and transcribe (listen) -> decode known devices (decode_devices).
Cheap dongles produce spurs: a narrow, steady, unmodulated line that never changes is usually an artifact;
verify with a waterfall. Whisper transcripts of music, noise or data signals are often hallucinated, so
cross-check against the audio spectrogram. Form hypotheses, test them, and state your confidence."""


def build_server():
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("rtl-sdr", instructions=INSTRUCTIONS)
    lock = asyncio.Lock()

    def stamp():
        return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    @mcp.tool()
    async def sdr_info() -> str:
        """Receiver details: tuner, usable frequency range, gain, local time, where recordings are saved."""
        hw = dict(await asyncio.to_thread(hardware))
        g = hw.pop("gains")
        hw["gain_range_db"] = [min(g), max(g)] if g else None
        return json.dumps({**hw, "local_time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                           "location": {"lat": LAT, "lon": LON} if LAT and LON else None,
                           "recordings_dir": str(DATA), "max_capture_bw_mhz": 2.0}, indent=1)

    @mcp.tool()
    async def spectrum_sweep(start_mhz: float, stop_mhz: float, bin_khz: float = 0, threshold_db: float = 7):
        """Sweep a frequency range and return a spectrum plot plus a list of signals above the noise floor,
        each tagged with the likely band allocation. bin_khz = resolution (0 = auto).
        A full 24-1700 MHz sweep takes ~30-60 s."""
        check_freq(start_mhz, stop_mhz)
        span = stop_mhz - start_mhz
        bin_hz = bin_khz * 1e3 if bin_khz else max(5e3, min(200e3, span * 1e6 / 4000))
        async with lock:
            sweeps = await asyncio.to_thread(rtl_power, start_mhz, stop_mhz, bin_hz, 2)
        if not sweeps:
            return "No data from rtl_power (is the dongle connected and not in use?)"
        _, f, p = sweeps[-1]
        peaks, floor = find_peaks(f, p, threshold_db)
        img = plot_spectrum(f, p, floor, peaks, f"Spectrum {start_mhz}-{stop_mhz} MHz, bin {bin_hz / 1e3:.1f} kHz")
        text = json.dumps({"bin_khz": round(bin_hz / 1e3, 2), "noise_floor_db": round(float(np.median(p)), 1),
                           "signals": peaks}, indent=1)
        return [text, Image(data=img, format="png")]

    @mcp.tool()
    async def waterfall(center_mhz: float, span_khz: float = 200, seconds: float = 5):
        """Waterfall (frequency x time spectrogram) around a frequency, returned as an image you can inspect:
        bandwidth, modulation shape, bursts, frequency hopping, periodicity. Also returns stats: occupied
        bandwidth, fraction of time active, number of bursts, SNR. span_khz up to 900, seconds 1-20."""
        check_freq(center_mhz)
        span = min(max(span_khz, 5), 900) * 1e3
        seconds = min(max(seconds, 1), 20)
        fs = 2.048e6
        async with lock:
            iq, tune = await asyncio.to_thread(capture_iq, center_mhz * 1e6, fs, seconds)
        db, freqs, noise, stats = analyze_iq(iq, fs, tune, center_mhz * 1e6, span)
        stats = {"center_mhz": center_mhz, "band": band_of(center_mhz), **stats}
        img = plot_waterfall(db, freqs, noise, center_mhz * 1e6, stats["duration_s"], stats["strongest_snr_db_avg"],
                             f"Waterfall {center_mhz} MHz ±{span / 2e3:.0f} kHz ({band_of(center_mhz)})")
        return [json.dumps(stats, indent=1), Image(data=img, format="png")]

    modes = {  # rtl_fm modulation, sample rate, audio rate, extra args
        "wfm": ("fm", 200000, 32000, ["-E", "deemp"]),
        "nfm": ("fm", 16000, 16000, []),
        "am": ("am", 16000, 16000, []),
        "usb": ("usb", 16000, 16000, []),
        "lsb": ("lsb", 16000, 16000, []),
    }

    @mcp.tool()
    async def listen(freq_mhz: float, mode: str = "wfm", seconds: float = 12, transcribe: bool = True,
                     language: str = "auto", squelch: int = 0):
        """Demodulate a frequency, save a WAV, transcribe speech (whisper.cpp, if installed) and return an
        audio spectrogram (shows speech vs music vs tones vs data like AFSK/POCSAG/DMR).
        mode: wfm (broadcast FM), nfm (walkie-talkies, ham, land mobile), am (aviation, CB), usb, lsb.
        seconds 3-60. squelch: rtl_fm squelch level (0 = off; ~50-200 for two-way radio)."""
        check_freq(freq_mhz)
        if mode not in modes:
            raise ValueError(f"mode must be one of {list(modes)}")
        seconds = min(max(seconds, 3), 60)
        m, fs, rate, extra = modes[mode]
        DATA.mkdir(parents=True, exist_ok=True)
        name = f"{stamp()}_{freq_mhz:.4f}_{mode}"
        raw, wav = DATA / f"{name}.raw", DATA / f"{name}.wav"
        cmd = [tool_path("rtl_fm"), "-d", DEVICE, "-M", m, "-f", f"{freq_mhz}M", "-s", str(fs), "-r", str(rate),
               "-g", gain(), *extra]
        if squelch:
            cmd += ["-l", str(squelch)]
        async with lock:
            with open(raw, "wb") as fh:
                await asyncio.to_thread(run_for, cmd, seconds, fh)
        audio = np.frombuffer(raw.read_bytes(), dtype=np.int16).astype(np.float32)
        raw.unlink()
        a, out_rate = write_wav(wav, audio, rate)
        info = {"freq_mhz": freq_mhz, "mode": mode, "band": band_of(freq_mhz),
                "seconds": round(len(a) / out_rate, 1), "wav": str(wav),
                "audio_rms": round(float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0, 1),
                "silence_fraction": round(float(np.mean(np.abs(a) < 50)) if len(a) else 1.0, 2)}
        if transcribe and len(a) > out_rate:
            whisper = shutil.which("whisper-cli")
            if whisper and WHISPER_MODEL.exists():
                r = await asyncio.to_thread(subprocess.run, [whisper, "-m", str(WHISPER_MODEL), "-l", language,
                                                             "-nt", "-np", "-f", str(wav)],
                                            capture_output=True, text=True, timeout=240)
                info["transcript"] = " ".join(r.stdout.split()) or "(no speech recognised)"
            else:
                info["transcript"] = "(transcription unavailable: install whisper.cpp and set WHISPER_MODEL)"
        out = [json.dumps(info, ensure_ascii=False, indent=1)]
        if len(a) > 4096:
            out.append(Image(data=plot_audio(a, out_rate, f"Audio spectrogram {freq_mhz} MHz ({mode})"), format="png"))
        return out

    @mcp.tool()
    async def activity_monitor(start_mhz: float, stop_mhz: float, minutes: float = 2, bin_khz: float = 0,
                               threshold_db: float = 8):
        """Watch a slice of spectrum for several minutes (a sweep every ~5 s) and return a frequency x time
        activity map. Finds intermittent transmitters (walkie-talkies, remotes, sensors, pagers) and reports
        when and how often each channel was active. Keep the slice under ~30 MHz; minutes 0.5-10."""
        check_freq(start_mhz, stop_mhz)
        minutes = min(max(minutes, 0.5), 10)
        span = stop_mhz - start_mhz
        bin_hz = bin_khz * 1e3 if bin_khz else max(2.5e3, min(50e3, span * 1e6 / 800))
        async with lock:
            sweeps = await asyncio.to_thread(rtl_power, start_mhz, stop_mhz, bin_hz, 5, minutes * 60)
        if len(sweeps) < 2:
            return "Not enough sweeps: narrow the range or watch longer."
        n = min(len(s[1]) for s in sweeps)
        f = sweeps[0][1][:n]
        M = np.array([s[2][:n] for s in sweeps])
        times = [s[0][-8:] for s in sweeps]
        chans, ex = activity_channels(f, M, times, threshold_db)
        img = plot_activity(ex, f, len(sweeps), threshold_db, f"Activity {start_mhz}-{stop_mhz} MHz over {minutes} min")
        text = json.dumps({"sweeps": len(sweeps), "bin_khz": round(bin_hz / 1e3, 2), "channels": chans[:40]}, indent=1)
        return [text, Image(data=img, format="png")]

    @mcp.tool()
    async def decode_devices(seconds: int = 60, freq_mhz: float = 433.92) -> str:
        """Decode packets from consumer devices with rtl_433 (200+ protocols): weather stations, temperature
        sensors, tyre pressure sensors, doorbells, remotes, utility meters. Typical: 433.92, 315, 868.3, 915 MHz.
        seconds 10-300."""
        check_freq(freq_mhz)
        seconds = int(min(max(seconds, 10), 300))
        async with lock:
            r = await asyncio.to_thread(subprocess.run, [tool_path("rtl_433"), "-d", DEVICE, "-f", f"{freq_mhz}M",
                                                         "-T", str(seconds), "-F", "json", "-M", "level", "-M", "time:iso"],
                                        capture_output=True, text=True, timeout=seconds + 30)
        devs: dict = {}
        for line in r.stdout.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = f"{e.get('model')}|{e.get('id')}|{e.get('channel')}"
            d = devs.setdefault(key, {"model": e.get("model"), "id": e.get("id"), "packets": 0})
            d["packets"] += 1
            d["last"] = e
        return json.dumps({"freq_mhz": freq_mhz, "seconds": seconds, "devices": list(devs.values())}, indent=1)

    @mcp.tool()
    async def satellite_passes(hours: float = 24, min_elevation: float = 10, lat: float | None = None,
                               lon: float | None = None) -> str:
        """Upcoming passes of satellites receivable with an RTL-SDR: Meteor-M2-3/M2-4 (LRPT 137.9 MHz Earth
        images) and the ISS (145.800 MHz NFM voice / SSTV images). Uses SDR_LAT/SDR_LON unless lat/lon given.
        Times are local."""
        la, lo = lat if lat is not None else LAT, lon if lon is not None else LON
        if la is None or lo is None:
            return "Location unknown: pass lat/lon or set SDR_LAT and SDR_LON environment variables."
        from skyfield.api import EarthSatellite, load, wgs84
        DATA.mkdir(parents=True, exist_ok=True)
        tle = DATA / "tle.txt"
        if not tle.exists() or time.time() - tle.stat().st_mtime > 12 * 3600:
            txt = b""
            for g in ("weather", "stations"):
                url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle"
                txt += await asyncio.to_thread(lambda u=url: urllib.request.urlopen(u, timeout=30).read())
            tle.write_bytes(txt)
        want = {"METEOR-M2 3": "LRPT 137.9 MHz", "METEOR-M2 4": "LRPT 137.9 MHz", "ISS (ZARYA)": "145.800 MHz NFM"}
        ts = load.timescale()
        lines = [ln.rstrip() for ln in tle.read_text().splitlines()]
        me = wgs84.latlon(float(la), float(lo))
        t0 = ts.now()
        t1 = ts.tt_jd(t0.tt + hours / 24)
        out = []
        for i in range(len(lines) - 2):
            name = lines[i].strip()
            if name not in want:
                continue
            sat = EarthSatellite(lines[i + 1], lines[i + 2], name, ts)
            times, events = sat.find_events(me, t0, t1, altitude_degrees=5)
            cur: dict = {}
            for t, ev in zip(times, events):
                if ev == 0:
                    cur = {"sat": name, "signal": want[name], "_t": t.utc_datetime(),
                           "start": t.utc_datetime().astimezone().strftime("%Y-%m-%d %H:%M")}
                elif ev == 1 and cur:
                    cur["max_elev"] = round((sat - me).at(t).altaz()[0].degrees, 1)
                elif ev == 2 and "max_elev" in cur:
                    cur["end"] = t.utc_datetime().astimezone().strftime("%H:%M")
                    if cur["max_elev"] >= min_elevation:
                        out.append(cur)
                    cur = {}
        out.sort(key=lambda p: p["_t"])
        for p in out:
            p.pop("_t")
        return json.dumps(out, indent=1)

    return mcp


def main():
    build_server().run()


if __name__ == "__main__":
    main()
