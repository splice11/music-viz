"""Convert features_per_stem.json (from Colab) → viz_data_v2.json
Same shape as v1 but with REAL stems and all 4 stem layers exposed."""
import json, numpy as np, os

src = json.load(open("/home/claude/bundle/features_per_stem.json"))

DURATION = src["stems"]["other"]["duration"]
SRC_FPS  = src["stems"]["other"]["frame_rate_fps"]
TARGET_FPS = 30

n_src = len(src["stems"]["other"]["rms"])
viz_n = int(DURATION * TARGET_FPS)
viz_t = np.linspace(0, DURATION, viz_n)
src_t = np.arange(n_src) / SRC_FPS

def rs(arr):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0: return [0.0] * viz_n
    if len(a) != n_src:
        a = np.interp(np.linspace(0, len(a)-1, n_src), np.arange(len(a)), a)
    return np.interp(viz_t, src_t, a).tolist()

def rs_chroma(chroma):
    arr = np.asarray(chroma)  # 12 x N
    if arr.shape[1] != n_src:
        # Resample columns
        new = np.zeros((12, n_src))
        for i in range(12):
            new[i] = np.interp(np.linspace(0, arr.shape[1]-1, n_src), np.arange(arr.shape[1]), arr[i])
        arr = new
    out = np.zeros((12, viz_n))
    for i in range(12):
        out[i] = np.interp(viz_t, src_t, arr[i])
    return out.tolist()

def stem_block(stem):
    block = {
        "rms":         rs(stem["rms"]),
        "onset_env":   rs(stem["onset_env"]),
        "onset_times": [float(t) for t in stem["onset_times"]],
        "centroid_hz": rs(stem["centroid"]),
        "bass_db":     rs(stem["bass_db"]),
        "mid_db":      rs(stem["mid_db"]),
        "treble_db":   rs(stem["treble_db"]),
        "chroma":      rs_chroma(stem["chroma"]),
    }
    if stem.get("pitch_hz"):
        block["pitch_hz"] = rs(stem["pitch_hz"])
    return block

viz_data = {
    "title": "Cryogenic — Everyday Astronaut",
    "duration": float(DURATION),
    "fps": TARGET_FPS,
    "n_frames": viz_n,
    "tempo_bpm": float(src["tempo_bpm"]),
    "beat_times": [float(t) for t in src["beat_times"]],
    "sections": [
        {"start": 0.0,    "end": 25.0,  "name": "intro",     "mood": "ambient build"},
        {"start": 25.0,   "end": 68.0,  "name": "verse",     "mood": "groove"},
        {"start": 68.0,   "end": 100.0, "name": "drop_1",    "mood": "impact"},
        {"start": 100.0,  "end": 175.0, "name": "main",      "mood": "drive"},
        {"start": 175.0,  "end": 195.0, "name": "breakdown", "mood": "pull-back"},
        {"start": 195.0,  "end": float(DURATION), "name": "outro", "mood": "rise"},
    ],
    # Mix (synthesized: sum of stem RMS as proxy)
    "mix": {
        "rms":       rs(src["stems"]["other"]["rms"]),  # using "other" RMS as a proxy
        "onset_env": rs(src["stems"]["other"]["onset_env"]),
    },
    # All four stems exposed
    "stems": {
        "bass":   stem_block(src["stems"]["bass"]),
        "drums":  stem_block(src["stems"]["drums"]),
        "lead":   stem_block(src["stems"]["other"]),   # rename "other" -> "lead" for clarity
        "vocals": stem_block(src["stems"]["vocals"]),
    },
    # Convenience aliases for the existing v1 visualizer (lead/drums at top level)
    "lead":  stem_block(src["stems"]["other"]),
    "drums": stem_block(src["stems"]["drums"]),
}

# Better mix RMS = max across stems
mix_rms = np.maximum.reduce([
    np.array(viz_data["stems"]["bass"]["rms"]),
    np.array(viz_data["stems"]["drums"]["rms"]),
    np.array(viz_data["stems"]["lead"]["rms"]),
])
viz_data["mix"]["rms"] = mix_rms.tolist()
mix_onset = np.maximum.reduce([
    np.array(viz_data["stems"]["bass"]["onset_env"]),
    np.array(viz_data["stems"]["drums"]["onset_env"]),
    np.array(viz_data["stems"]["lead"]["onset_env"]),
])
viz_data["mix"]["onset_env"] = mix_onset.tolist()

with open("/home/claude/viz_data_v2.json", "w") as f:
    json.dump(viz_data, f)

size_mb = os.path.getsize("/home/claude/viz_data_v2.json") / 1024 / 1024
print(f"Saved viz_data_v2.json ({size_mb:.2f} MB)")
print(f"Frames: {viz_n} @ {TARGET_FPS} fps, duration {DURATION:.1f}s")

# Quick stats per stem
print("\nPer-stem RMS means:")
for k, s in viz_data["stems"].items():
    rms_arr = np.array(s["rms"])
    print(f"  {k:7s}: rms_mean={rms_arr.mean():.4f}  rms_max={rms_arr.max():.4f}  onsets={len(s['onset_times'])}")
