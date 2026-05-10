"""GPU offline renderer.

Renders one frame at a time to an HDR float framebuffer, runs a small bloom
pyramid, then composites with tonemap + chromatic aberration + grain + vignette.
Frames are read back as RGB uint8 and written to ffmpeg's stdin so we never
touch the disk for intermediate PNGs.
"""
from __future__ import annotations
import os, sys, time, subprocess, glob
import numpy as np

# Force libEGL to load only the NVIDIA ICD on Colab GPU runtimes. apt's libegl1
# installs Mesa's ICD too, and the default device-selection often picks
# llvmpipe (software), pinning the renderer to ~1 fps. We must set this before
# moderngl/libEGL is loaded, so it lives at module top.
_NV_ICDS = [p for p in glob.glob("/usr/share/glvnd/egl_vendor.d/*nvidia*.json")]
if _NV_ICDS and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ":".join(_NV_ICDS)
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import moderngl
from . import shaders
from .features import FrameData


def _is_software(renderer_str: str) -> bool:
    s = renderer_str.lower()
    return any(k in s for k in ("llvmpipe", "softpipe", "swrast", "software"))


def _make_ctx(verbose: bool = True):
    """Create a standalone EGL OpenGL 3.3 context bound to the NVIDIA GPU.

    Iterates EGL devices and picks the first one whose GL_RENDERER doesn't
    look like a software rasterizer. Raises with a clear message if none
    found, since otherwise rendering would silently fall back to ~1 fps.
    """
    icds = sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json"))
    nv_icd = [p for p in icds if "nvidia" in p.lower()]
    if not nv_icd:
        raise RuntimeError(
            "No NVIDIA EGL ICD is registered. Found ICDs: "
            f"{icds or '(none)'}.\n"
            "Re-run the Setup cell — it must apt-install `libnvidia-gl-<MAJOR>` "
            "matching the driver from `nvidia-smi`. Without that package, "
            "libEGL has no NVIDIA backend and falls back to llvmpipe (~1 fps)."
        )

    software_renderers_seen = []
    for idx in range(8):
        try:
            ctx = moderngl.create_context(
                standalone=True, backend="egl", require=330, device_index=idx,
            )
        except Exception:
            # moderngl raises once we exceed the device count; stop scanning.
            break
        renderer = ctx.info.get("GL_RENDERER", "?")
        vendor = ctx.info.get("GL_VENDOR", "?")
        if verbose:
            print(f"[colab_render] EGL device {idx}: {vendor} | {renderer}")
        if not _is_software(renderer):
            return ctx
        software_renderers_seen.append(renderer)
        ctx.release()

    raise RuntimeError(
        "All EGL devices reported software rendering "
        f"({software_renderers_seen!r}) even though an NVIDIA ICD is present. "
        "This usually means libEGL_nvidia.so.0 isn't on the loader path. "
        "Try restarting the runtime, then re-running the Setup cell."
    )


class Renderer:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        self.ctx = _make_ctx()

        # Fullscreen-triangle VAO (no vertex buffer needed; uses gl_VertexID).
        self.vao_empty = self.ctx.vertex_array(
            self.ctx.program(
                vertex_shader=shaders.FULLSCREEN_VS,
                fragment_shader="#version 330\nout vec4 f; void main(){ f = vec4(0.0); }",
            ),
            [],
        )
        self.scene_prog = self.ctx.program(
            vertex_shader=shaders.FULLSCREEN_VS,
            fragment_shader=shaders.SCENE_FS,
        )
        self.bright_prog = self.ctx.program(
            vertex_shader=shaders.FULLSCREEN_VS,
            fragment_shader=shaders.BRIGHT_FS,
        )
        self.blur_prog = self.ctx.program(
            vertex_shader=shaders.FULLSCREEN_VS,
            fragment_shader=shaders.BLUR_FS,
        )
        self.composite_prog = self.ctx.program(
            vertex_shader=shaders.FULLSCREEN_VS,
            fragment_shader=shaders.COMPOSITE_FS,
        )
        self.scene_vao = self.ctx.vertex_array(self.scene_prog, [])
        self.bright_vao = self.ctx.vertex_array(self.bright_prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])
        self.composite_vao = self.ctx.vertex_array(self.composite_prog, [])

        # FBOs
        self.fb_scene = self._make_fbo(width, height, dtype="f2")           # HDR half-float
        bw, bh = width // 4, height // 4
        self.fb_bright = self._make_fbo(bw, bh, dtype="f2")
        self.fb_blur1 = self._make_fbo(bw, bh, dtype="f2")
        self.fb_blur2 = self._make_fbo(bw, bh, dtype="f2")
        self.fb_out = self._make_fbo(width, height, dtype="f1")             # 8-bit final

        self._bw, self._bh = bw, bh

    def _make_fbo(self, w: int, h: int, dtype: str):
        tex = self.ctx.texture((w, h), 4, dtype=dtype)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return self.ctx.framebuffer(color_attachments=[tex])

    def _set_uniform(self, prog, name: str, value):
        if name in prog:
            prog[name].value = value

    def render_frame(self, fd: FrameData, frame_idx: int) -> bytes:
        i = frame_idx
        t = i / fd.fps

        # ---- 1. Scene pass ----
        self.fb_scene.use()
        self.ctx.viewport = (0, 0, self.W, self.H)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        p = self.scene_prog
        self._set_uniform(p, "u_res", (float(self.W), float(self.H)))
        self._set_uniform(p, "u_time", float(t))
        self._set_uniform(p, "u_modeA", int(fd.mode_a[i]))
        self._set_uniform(p, "u_modeB", int(fd.mode_b[i]))
        self._set_uniform(p, "u_modeBlend", float(fd.mode_blend[i]))
        self._set_uniform(p, "u_sectionT", float(fd.section_t[i]))
        self._set_uniform(p, "u_bass", float(fd.bass[i]))
        self._set_uniform(p, "u_bassPunch", float(fd.bass_punch[i]))
        self._set_uniform(p, "u_drumsRms", float(fd.drums_rms[i]))
        self._set_uniform(p, "u_drumsPunch", float(fd.drums_punch[i]))
        self._set_uniform(p, "u_vocalsRms", float(fd.vocals_rms[i]))
        self._set_uniform(p, "u_otherRms", float(fd.other_rms[i]))
        self._set_uniform(p, "u_globalRms", float(fd.global_rms[i]))
        self._set_uniform(p, "u_centroid", float(fd.centroid[i]))
        self._set_uniform(p, "u_pitch", float(fd.pitch[i]))
        self._set_uniform(p, "u_beatPhase", float(fd.beat_phase[i]))
        # chroma array (12 floats)
        if "u_chroma[0]" in p:
            for k in range(12):
                p[f"u_chroma[{k}]"].value = float(fd.chroma[i, k])
        self.scene_vao.render(moderngl.TRIANGLES, vertices=3)

        # ---- 2. Bright extract ----
        self.fb_bright.use()
        self.ctx.viewport = (0, 0, self._bw, self._bh)
        self.ctx.clear(0, 0, 0, 1)
        self.fb_scene.color_attachments[0].use(location=0)
        self._set_uniform(self.bright_prog, "u_tex", 0)
        self._set_uniform(self.bright_prog, "u_threshold", 0.85)
        self.bright_vao.render(moderngl.TRIANGLES, vertices=3)

        # ---- 3. Separable gaussian (H then V) ----
        # H pass: bright -> blur1
        self.fb_blur1.use()
        self.ctx.viewport = (0, 0, self._bw, self._bh)
        self.ctx.clear(0, 0, 0, 1)
        self.fb_bright.color_attachments[0].use(location=0)
        self._set_uniform(self.blur_prog, "u_tex", 0)
        self._set_uniform(self.blur_prog, "u_dir", (1.0, 0.0))
        self._set_uniform(self.blur_prog, "u_texel", (1.0 / self._bw, 1.0 / self._bh))
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)
        # V pass: blur1 -> blur2
        self.fb_blur2.use()
        self.ctx.clear(0, 0, 0, 1)
        self.fb_blur1.color_attachments[0].use(location=0)
        self._set_uniform(self.blur_prog, "u_dir", (0.0, 1.0))
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)
        # second blur pass for wider bloom
        self.fb_blur1.use()
        self.ctx.clear(0, 0, 0, 1)
        self.fb_blur2.color_attachments[0].use(location=0)
        self._set_uniform(self.blur_prog, "u_dir", (1.0, 0.0))
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)
        self.fb_blur2.use()
        self.ctx.clear(0, 0, 0, 1)
        self.fb_blur1.color_attachments[0].use(location=0)
        self._set_uniform(self.blur_prog, "u_dir", (0.0, 1.0))
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # ---- 4. Composite ----
        self.fb_out.use()
        self.ctx.viewport = (0, 0, self.W, self.H)
        self.ctx.clear(0, 0, 0, 1)
        self.fb_scene.color_attachments[0].use(location=0)
        self.fb_blur2.color_attachments[0].use(location=1)
        cp = self.composite_prog
        self._set_uniform(cp, "u_scene", 0)
        self._set_uniform(cp, "u_bloom", 1)
        self._set_uniform(cp, "u_time", float(t))
        # ramp post-FX with energy: more bloom in drops, tighter CA
        # bloom is kept lean — the new shaders are raymarched and have real
        # silhouettes; heavy bloom would put the fog right back.
        energy = float(fd.global_rms[i])
        bloom_amt = 0.30 + 0.45 * energy
        ca_amt = 0.0010 + 0.004 * energy
        grain_amt = 0.035
        vignette = 0.55
        exposure = 1.05 + 0.25 * energy
        self._set_uniform(cp, "u_bloomAmt", bloom_amt)
        self._set_uniform(cp, "u_caAmt", ca_amt)
        self._set_uniform(cp, "u_grainAmt", grain_amt)
        self._set_uniform(cp, "u_vignette", vignette)
        self._set_uniform(cp, "u_exposure", exposure)
        self.composite_vao.render(moderngl.TRIANGLES, vertices=3)

        # ---- 5. Read back as RGB24 ----
        # moderngl gives us RGBA; ffmpeg gets fed rgb24 to keep the pipe small.
        raw = self.fb_out.read(components=3, alignment=1)
        return raw


def render_to_video(
    fd: FrameData,
    audio_path: str,
    out_path: str,
    width: int = 1920,
    height: int = 1080,
    crf: int = 17,
    preset: str = "medium",
    progress_every: int = 60,
):
    """Render every frame in fd, pipe to ffmpeg, mux audio in one pass."""
    rend = Renderer(width, height)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        # video from stdin
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fd.fps:.6f}",
        "-i", "pipe:0",
        # audio from file
        "-i", audio_path,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.time()
    try:
        for i in range(fd.n_frames):
            buf = rend.render_frame(fd, i)
            proc.stdin.write(buf)
            if (i + 1) % progress_every == 0 or i == fd.n_frames - 1:
                done = i + 1
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                eta = (fd.n_frames - done) / max(rate, 1e-3)
                print(f"  frame {done}/{fd.n_frames}  "
                      f"{rate:.1f} fps  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s",
                      flush=True)
    finally:
        proc.stdin.close()
        proc.wait()

    print(f"\nDone → {out_path}")
