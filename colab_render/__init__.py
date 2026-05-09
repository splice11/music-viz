"""Cryogenic offline music visualizer.

`build` is import-light. The renderer needs moderngl + a GPU EGL context, so
it's imported lazily — accessing `colab_render.render_to_video` or `Renderer`
will trigger the import only when needed.
"""
from .features import build, FrameData

__all__ = ["build", "FrameData", "render_to_video", "Renderer"]


def __getattr__(name):
    if name in ("render_to_video", "Renderer"):
        from . import renderer
        return getattr(renderer, name)
    raise AttributeError(name)
