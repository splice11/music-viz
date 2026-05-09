"""Pre-process viz_data_v2.json into per-frame uniform arrays at TARGET_FPS.

Output is a dict of numpy arrays, one entry per output frame. The renderer
just indexes into these by frame number — no per-frame audio analysis.
"""
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass


def _resample(src: np.ndarray, n_out: int) -> np.ndarray:
    """Linear resample a 1-D array to length n_out."""
    src = np.asarray(src, dtype=np.float32)
    if len(src) == n_out:
        return src
    if len(src) == 0:
        return np.zeros(n_out, dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, len(src))
    x_out = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_out, x_src, src).astype(np.float32)


def _normalize(a: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 99.0) -> np.ndarray:
    lo = np.percentile(a, lo_pct)
    hi = np.percentile(a, hi_pct)
    if hi - lo < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth(a: np.ndarray, alpha: float) -> np.ndarray:
    """One-pole low-pass filter; alpha in (0,1], smaller = smoother."""
    out = np.empty_like(a, dtype=np.float32)
    acc = 0.0
    for i, v in enumerate(a):
        acc = acc + alpha * (float(v) - acc)
        out[i] = acc
    return out


def _onset_envelope(times: list[float], n_frames: int, fps: float, decay: float = 4.5) -> np.ndarray:
    """Convert sparse onset times into a smooth pulsing envelope."""
    env = np.zeros(n_frames, dtype=np.float32)
    if not times:
        return env
    idxs = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, n_frames - 1)
    for i in idxs:
        env[i] = 1.0
    # exponential decay forward
    out = np.empty_like(env)
    v = 0.0
    dt = 1.0 / fps
    k = np.exp(-decay * dt)
    for i in range(n_frames):
        v = max(env[i], v * k)
        out[i] = v
    return out


def _beat_phase(beat_times: list[float], n_frames: int, fps: float) -> np.ndarray:
    """0..1 phase between successive beats."""
    out = np.zeros(n_frames, dtype=np.float32)
    if not beat_times or len(beat_times) < 2:
        return out
    bt = np.asarray(beat_times, dtype=np.float64)
    for i in range(n_frames):
        t = i / fps
        idx = np.searchsorted(bt, t) - 1
        if idx < 0:
            out[i] = 0.0
            continue
        if idx >= len(bt) - 1:
            period = bt[-1] - bt[-2] if len(bt) >= 2 else 0.5
            out[i] = ((t - bt[-1]) / max(period, 1e-3)) % 1.0
            continue
        period = bt[idx + 1] - bt[idx]
        out[i] = (t - bt[idx]) / max(period, 1e-3)
    return out


def _section_mode_id(name: str) -> int:
    table = {
        "intro": 0, "verse": 1, "drop": 2, "drop_1": 2, "drop_2": 2,
        "main": 3, "chorus": 3, "breakdown": 4, "bridge": 4, "outro": 5,
    }
    return table.get(name, 1)


@dataclass
class FrameData:
    n_frames: int
    fps: float
    duration: float
    # per-frame arrays
    bass: np.ndarray
    bass_punch: np.ndarray
    drums_rms: np.ndarray
    drums_punch: np.ndarray
    vocals_rms: np.ndarray
    other_rms: np.ndarray
    global_rms: np.ndarray
    centroid: np.ndarray
    pitch: np.ndarray
    beat_phase: np.ndarray
    chroma: np.ndarray            # (n_frames, 12)
    mode_a: np.ndarray            # current section mode id (int)
    mode_b: np.ndarray            # next section mode id (int)
    mode_blend: np.ndarray        # 0..1 blend toward mode_b near boundaries
    section_t: np.ndarray         # 0..1 progress within current section


def build(json_path: str, target_fps: int = 60) -> FrameData:
    d = json.load(open(json_path))
    duration = float(d["duration"])
    src_fps = float(d["fps"])
    n_src = int(d["n_frames"])
    n_out = int(round(duration * target_fps))

    stems = d["stems"]
    bass_stem = stems["bass"]
    drums_stem = stems["drums"]
    vocals_stem = stems["vocals"]
    other_stem = stems.get("other", stems.get("lead", drums_stem))
    lead_stem = d.get("lead", vocals_stem)

    # --- band/RMS resampled to target fps and normalized ---
    bass = _normalize(_resample(bass_stem["rms"], n_out))
    drums_rms = _normalize(_resample(drums_stem["rms"], n_out))
    vocals_rms = _normalize(_resample(vocals_stem["rms"], n_out))
    other_rms = _normalize(_resample(other_stem["rms"], n_out))
    global_rms = _normalize(_resample(d["mix"]["rms"], n_out))

    # smooth a touch
    bass = _smooth(bass, 0.4)
    drums_rms = _smooth(drums_rms, 0.5)
    vocals_rms = _smooth(vocals_rms, 0.5)
    other_rms = _smooth(other_rms, 0.4)
    global_rms = _smooth(global_rms, 0.4)

    # punches from onset times → exponential envelopes
    bass_punch = _onset_envelope(bass_stem.get("onset_times", []), n_out, target_fps, decay=6.0)
    drums_punch = _onset_envelope(drums_stem.get("onset_times", []), n_out, target_fps, decay=8.0)

    # spectral centroid → 0..1 (log scale, then normalize)
    centroid_raw = np.asarray(_resample(lead_stem["centroid_hz"], n_out))
    centroid = _normalize(np.log1p(np.maximum(centroid_raw, 1.0)))

    # pitch → 0..1 across vocal range
    pitch_raw = np.asarray(_resample(lead_stem.get("pitch_hz", np.zeros(n_src)), n_out))
    # treat 80-800 Hz log-mapped as the active range
    pmask = pitch_raw > 1.0
    if pmask.any():
        lp = np.log(np.clip(pitch_raw, 1.0, 2000.0))
        lo, hi = np.log(80.0), np.log(800.0)
        pitch = np.clip((lp - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        pitch[~pmask] = 0.5
    else:
        pitch = np.full(n_out, 0.5, dtype=np.float32)
    pitch = _smooth(pitch, 0.25)

    # chroma — 12 vectors per frame, take the mix's lead chroma
    chroma_src = np.asarray(lead_stem["chroma"], dtype=np.float32)
    if chroma_src.ndim == 2 and chroma_src.shape[0] == 12:
        chroma_src = chroma_src.T  # → (n_src, 12)
    if chroma_src.shape[0] != n_out:
        chroma = np.stack([_resample(chroma_src[:, k], n_out) for k in range(12)], axis=1)
    else:
        chroma = chroma_src
    # normalize per-frame so dominant pitch ~1
    row_max = chroma.max(axis=1, keepdims=True)
    chroma = chroma / np.maximum(row_max, 1e-3)

    beat_phase = _beat_phase(d.get("beat_times", []), n_out, target_fps)

    # --- section mode + cross-fade near boundaries ---
    sections = d["sections"]
    mode_a = np.zeros(n_out, dtype=np.int32)
    mode_b = np.zeros(n_out, dtype=np.int32)
    mode_blend = np.zeros(n_out, dtype=np.float32)
    section_t = np.zeros(n_out, dtype=np.float32)

    BLEND_S = 1.5  # cross-fade window in seconds
    for i in range(n_out):
        t = i / target_fps
        # find current section
        cur = sections[0]
        nxt = sections[0]
        cur_idx = 0
        for k, s in enumerate(sections):
            if s["start"] <= t < s["end"]:
                cur = s
                cur_idx = k
                break
        else:
            cur = sections[-1]
            cur_idx = len(sections) - 1
        nxt = sections[min(cur_idx + 1, len(sections) - 1)]
        section_t[i] = (t - cur["start"]) / max(cur["end"] - cur["start"], 1e-3)

        mode_a[i] = _section_mode_id(cur["name"])
        mode_b[i] = _section_mode_id(nxt["name"])
        # blend ramps from 0 to 1 across the last BLEND_S seconds of this section
        time_left = cur["end"] - t
        if cur_idx < len(sections) - 1 and time_left < BLEND_S:
            mode_blend[i] = 1.0 - (time_left / BLEND_S)
        else:
            mode_blend[i] = 0.0

    return FrameData(
        n_frames=n_out, fps=float(target_fps), duration=duration,
        bass=bass, bass_punch=bass_punch,
        drums_rms=drums_rms, drums_punch=drums_punch,
        vocals_rms=vocals_rms, other_rms=other_rms, global_rms=global_rms,
        centroid=centroid, pitch=pitch,
        beat_phase=beat_phase, chroma=chroma,
        mode_a=mode_a, mode_b=mode_b, mode_blend=mode_blend, section_t=section_t,
    )
