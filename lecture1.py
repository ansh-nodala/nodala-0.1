#!/usr/bin/env python3
"""
================================================================================
SILENT MATH — LECTURE #1: INTRODUCTION TO AI & MACHINE LEARNING
Pure 3Blue1Brown-Style 2D & 3D Mathematical Animation Engine (Zero Boxes/Panels)
================================================================================

Channel Name : Silent Math
Video Title  : Introduction to AI & Machine Learning — Lecture #1
Tagline      : Understand the mathematics. Visualize the intelligence.
Target Output: silent_math_lecture_01_ai_ml.mp4

Visual Philosophy (Pure 3Blue1Brown Aesthetic):
  - ZERO UI boxes, ZERO rectangular containers, ZERO dashboard cards, ZERO borders.
  - Infinite deep-black mathematical space (#000000) where only coordinate axes,
    smooth parametric curves, glowing data points, vectors, neural network graphs,
    and 3D wireframe surfaces/manifolds exist.
  - Minimalist floating equations (IBM Plex Mono / Latin Modern Mono) and a single
    unboxed, word-by-word animated caption floating cleanly at the bottom margin.

Specifications:
  - Duration     : Exactly 1800.0 seconds (30 minutes, 54,000 frames @ 30 FPS)
  - Aspect Ratio : 16:9 Widescreen (3840 x 2160 4K UHD default)
  - Frame Rate   : 30 FPS
  - Video Codec  : H.264 (libx264, yuv420p), strictly no audio track (-an)

Commands to Run:
  # 1. Fast preview across all 11 scenes:
  python3 video_lecture1.py --mode preview --res 720p --workers 2 --output preview_lecture1.mp4

  # 2. Full 30-minute (1800s) 4K UHD render with live 2-second heartbeat logs:
  python3 video_lecture1.py --mode full --res 4k --workers 4 --output silent_math_lecture_01_ai_ml.mp4

  # 3. Validate timeline & mathematical invariants:
  python3 video_lecture1.py --validate
================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Preserve LD_LIBRARY_PATH before importing cv2 so child ffmpeg/ffprobe processes
# never stall on bundled opencv-python shared library paths.
_ORIG_LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH")
import cv2

if _ORIG_LD_LIBRARY_PATH is None:
    os.environ.pop("LD_LIBRARY_PATH", None)
else:
    os.environ["LD_LIBRARY_PATH"] = _ORIG_LD_LIBRARY_PATH
os.environ.pop("QT_QPA_FONTDIR", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
cv2.setNumThreads(1)

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# 1. CONFIGURATION, MONOCHROME + SURGICAL ACCENT PALETTE, AND TYPOGRAPHY
# ==============================================================================

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "4k": (3840, 2160),
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
}


@dataclass(frozen=True)
class Palette:
    """Restrained 3Blue1Brown-inspired mathematical laboratory palette on pure black."""
    BG_BLACK: Tuple[int, int, int] = (0, 0, 0)
    GRID_FAINT: Tuple[int, int, int] = (20, 22, 28)
    GRID_MAJOR: Tuple[int, int, int] = (38, 42, 52)
    AXIS_GRAY: Tuple[int, int, int] = (125, 130, 142)
    GRAY_DARK: Tuple[int, int, int] = (72, 76, 86)
    GRAY_MID: Tuple[int, int, int] = (145, 150, 162)
    GRAY_LIGHT: Tuple[int, int, int] = (210, 214, 224)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    ACCENT_BLUE: Tuple[int, int, int] = (130, 210, 255)   # Primary mathematical curve / model
    ACCENT_GOLD: Tuple[int, int, int] = (255, 205, 105)   # Error residual / loss / active point
    ACCENT_CORAL: Tuple[int, int, int] = (245, 125, 135)  # Overfitting / high-error contrast


PALETTE = Palette()


@dataclass
class RenderConfig:
    width: int = 3840
    height: int = 2160
    fps: int = 30
    total_duration: float = 1800.0

    @property
    def scale(self) -> float:
        return self.height / 2160.0

    def px(self, val_at_4k: float) -> int:
        return max(1, int(round(val_at_4k * self.scale)))


@lru_cache(maxsize=32)
def resolve_font_path(role: str, bold: bool = False) -> str:
    if role == "mono":
        candidates = (
            [
                "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Bold.ttf",
                "/usr/share/fonts/opentype/ibm-plex/IBMPlexMono-Bold.otf",
                "/usr/share/texmf/fonts/opentype/public/lm/lmmonolt10-bold.otf",
                "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            ]
            if bold
            else [
                "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf",
                "/usr/share/fonts/opentype/ibm-plex/IBMPlexMono-Regular.otf",
                "/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf",
                "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ]
        )
        fc_queries = ["IBM Plex Mono", "Latin Modern Mono", "DejaVu Sans Mono"]
    else:
        candidates = (
            [
                "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
                "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf",
                "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
            if bold
            else [
                "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
                "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
                "/usr/share/fonts/opentype/inter/InterDisplay-Regular.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
        fc_queries = ["Inter", "DejaVu Sans"]

    for path in candidates:
        if os.path.isfile(path):
            return path

    if shutil.which("fc-match"):
        for query in fc_queries:
            try:
                out = subprocess.check_output(
                    ["fc-match", "-f", "%{file}", query],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if out and os.path.isfile(out):
                    return out
            except Exception:
                pass
    return ""


@lru_cache(maxsize=128)
def get_pil_font(role: str, size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = resolve_font_path(role, bold=bold)
    if path:
        try:
            return ImageFont.truetype(path, size=max(8, size_px))
        except Exception:
            pass
    return ImageFont.load_default()


# ==============================================================================
# 2. MATHEMATICAL EASING & INTERPOLATION FUNCTIONS
# ==============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_in_out_sine(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def window_envelope(
    t: float,
    t_start: float,
    t_end: float,
    fade_in: float = 0.6,
    fade_out: float = 0.6,
) -> float:
    if t <= t_start or t >= t_end:
        return 0.0
    a_in = smoothstep(t_start, t_start + max(1e-4, fade_in), t)
    a_out = 1.0 - smoothstep(t_end - max(1e-4, fade_out), t_end, t)
    return clamp(a_in * a_out, 0.0, 1.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t, 0.0, 1.0)


def lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    u = clamp(t, 0.0, 1.0)
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * u)),
        int(round(c1[1] + (c2[1] - c1[1]) * u)),
        int(round(c1[2] + (c2[2] - c1[2]) * u)),
    )


def scale_color(c: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    a = clamp(alpha, 0.0, 1.0)
    return (int(round(c[0] * a)), int(round(c[1] * a)), int(round(c[2] * a)))


# ==============================================================================
# 3. 2D & 3D VECTOR PROJECTION AND DRAWING PRIMITIVES (ZERO BOXES)
# ==============================================================================

@lru_cache(maxsize=1024)
def render_text_sprite(
    text: str,
    role: str,
    size_px: int,
    color: Tuple[int, int, int],
    bold: bool = False,
) -> np.ndarray:
    if not text:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    font = get_pil_font(role, size_px, bold=bold)
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    w = max(1, bbox[2] - bbox[0] + 8)
    h = max(1, bbox[3] - bbox[1] + 8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0] + 4, -bbox[1] + 4), text, font=font, fill=(color[0], color[1], color[2], 255))
    return np.array(img, dtype=np.uint8)


def draw_text(
    canvas: np.ndarray,
    text: str,
    x: float,
    y: float,
    size_4k: int,
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.WHITE,
    role: str = "sans",
    bold: bool = False,
    alpha: float = 1.0,
    anchor: str = "center",
) -> Tuple[int, int, int, int]:
    """Render floating typography directly onto the black canvas (zero boxes)."""
    if alpha <= 0.01 or not text:
        return (int(x), int(y), int(x), int(y))
    size_px = cfg.px(size_4k)
    sprite = render_text_sprite(text, role, size_px, color, bold=bold)
    sh, sw = sprite.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if anchor == "center":
        x0, y0 = ix - sw // 2, iy - sh // 2
    elif anchor == "midleft":
        x0, y0 = ix, iy - sh // 2
    elif anchor == "midright":
        x0, y0 = ix - sw, iy - sh // 2
    elif anchor == "topcenter":
        x0, y0 = ix - sw // 2, iy
    else:
        x0, y0 = ix, iy

    x1, y1 = x0 + sw, y0 + sh
    ch, cw = canvas.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(cw, x1), min(ch, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return (x0, y0, x1, y1)

    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    sub = sprite[sy0:sy1, sx0:sx1]
    a_u16 = (sub[:, :, 3:4].astype(np.uint16) * int(round(clamp(alpha, 0.0, 1.0) * 256))) >> 8
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.uint16)
    fg = sub[:, :, :3].astype(np.uint16)
    canvas[cy0:cy1, cx0:cx1] = ((roi * (256 - a_u16) + fg * a_u16) >> 8).astype(np.uint8)
    return (cx0, cy0, cx1, cy1)


def draw_aa_line(
    canvas: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    thickness_4k: float,
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.WHITE,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    c = scale_color(color, alpha)
    th = max(1, cfg.px(thickness_4k))
    cv2.line(
        canvas,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        c,
        th,
        cv2.LINE_AA,
    )


def draw_arrow(
    canvas: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    thickness_4k: float,
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.WHITE,
    alpha: float = 1.0,
    tip_length: float = 0.03,
) -> None:
    if alpha <= 0.01:
        return
    c = scale_color(color, alpha)
    th = max(1, cfg.px(thickness_4k))
    cv2.arrowedLine(
        canvas,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        c,
        th,
        cv2.LINE_AA,
        0,
        tip_length,
    )


def draw_dashed_line(
    canvas: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    thickness_4k: float,
    cfg: RenderConfig,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
    dash_px_4k: float = 18.0,
) -> None:
    if alpha <= 0.01:
        return
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1.0:
        return
    step = max(4.0, dash_px_4k * cfg.scale)
    n = max(1, int(dist / (2.0 * step)))
    for i in range(n + 1):
        u0 = min(1.0, (2 * i * step) / dist)
        u1 = min(1.0, ((2 * i + 1) * step) / dist)
        if u1 > u0:
            draw_aa_line(
                canvas,
                (lerp(x1, x2, u0), lerp(y1, y2, u0)),
                (lerp(x1, x2, u1), lerp(y1, y2, u1)),
                thickness_4k,
                cfg,
                color,
                alpha,
            )


def draw_glow_point(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius_4k: float,
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.WHITE,
    alpha: float = 1.0,
    glow_layers: int = 3,
    filled: bool = True,
) -> None:
    if alpha <= 0.01:
        return
    ix, iy = int(round(cx)), int(round(cy))
    r_base = max(2, cfg.px(radius_4k))
    for g in range(glow_layers, 0, -1):
        rg = int(round(r_base * (1.0 + 0.5 * g)))
        gc = scale_color(color, alpha * (0.16 / g))
        cv2.circle(canvas, (ix, iy), rg, gc, max(1, cfg.px(3)), cv2.LINE_AA)
    main_c = scale_color(color, alpha)
    if filled:
        cv2.circle(canvas, (ix, iy), r_base, main_c, -1, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (ix, iy), r_base, main_c, max(2, cfg.px(4)), cv2.LINE_AA)


@dataclass
class PlotViewport:
    """2D coordinate system mapped onto open screen space."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    px_left: float
    px_top: float
    px_right: float
    px_bottom: float

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        u = (x - self.x_min) / max(1e-9, self.x_max - self.x_min)
        v = (y - self.y_min) / max(1e-9, self.y_max - self.y_min)
        sx = self.px_left + u * (self.px_right - self.px_left)
        sy = self.px_bottom - v * (self.px_bottom - self.px_top)
        return (sx, sy)


def draw_3b1b_axes(
    canvas: np.ndarray,
    vp: PlotViewport,
    cfg: RenderConfig,
    alpha: float = 1.0,
    x_ticks: Sequence[float] = (),
    y_ticks: Sequence[float] = (),
    x_label: str = "x",
    y_label: str = "y",
    x_fmt: str = "{:.0f}",
    y_fmt: str = "{:.0f}",
    show_grid: bool = True,
) -> None:
    """Draw open 3Blue1Brown-style coordinate grid and axes with arrowheads."""
    if alpha <= 0.01:
        return
    if show_grid:
        for xt in x_ticks:
            sx, _ = vp.to_px(xt, vp.y_min)
            draw_aa_line(canvas, (sx, vp.px_top), (sx, vp.px_bottom), 1.8, cfg, PALETTE.GRID_FAINT, alpha)
        for yt in y_ticks:
            _, sy = vp.to_px(vp.x_min, yt)
            draw_aa_line(canvas, (vp.px_left, sy), (vp.px_right, sy), 1.8, cfg, PALETTE.GRID_FAINT, alpha)

    # Axes with clean arrowheads
    y0_val = 0.0 if (vp.y_min <= 0.0 <= vp.y_max) else vp.y_min
    x0_val = 0.0 if (vp.x_min <= 0.0 <= vp.x_max) else vp.x_min
    _, ay = vp.to_px(vp.x_min, y0_val)
    ax, _ = vp.to_px(x0_val, vp.y_min)

    draw_arrow(canvas, (vp.px_left, ay), (vp.px_right + cfg.px(24), ay), 4.0, cfg, PALETTE.AXIS_GRAY, alpha, tip_length=0.015)
    draw_arrow(canvas, (ax, vp.px_bottom), (ax, vp.px_top - cfg.px(24)), 4.0, cfg, PALETTE.AXIS_GRAY, alpha, tip_length=0.018)

    tick_len = cfg.px(12)
    for xt in x_ticks:
        sx, _ = vp.to_px(xt, y0_val)
        draw_aa_line(canvas, (sx, ay - tick_len), (sx, ay + tick_len), 3.0, cfg, PALETTE.GRAY_MID, alpha)
        if abs(xt - x0_val) > 1e-5:
            draw_text(canvas, x_fmt.format(xt), sx, ay + cfg.px(32), 28, cfg, PALETTE.GRAY_MID, "mono", alpha=alpha, anchor="topcenter")
    for yt in y_ticks:
        _, sy = vp.to_px(x0_val, yt)
        draw_aa_line(canvas, (ax - tick_len, sy), (ax + tick_len, sy), 3.0, cfg, PALETTE.GRAY_MID, alpha)
        if abs(yt - y0_val) > 1e-5:
            draw_text(canvas, y_fmt.format(yt), ax - cfg.px(22), sy, 28, cfg, PALETTE.GRAY_MID, "mono", alpha=alpha, anchor="midright")

    if x_label:
        draw_text(canvas, x_label, vp.px_right, ay - cfg.px(38), 34, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=alpha, anchor="midright")
    if y_label:
        draw_text(canvas, y_label, ax + cfg.px(28), vp.px_top, 34, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=alpha, anchor="midleft")


def draw_curve(
    canvas: np.ndarray,
    vp: PlotViewport,
    fn: Callable[[np.ndarray], np.ndarray],
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.ACCENT_BLUE,
    thickness_4k: float = 6.5,
    alpha: float = 1.0,
    progress: float = 1.0,
    num_samples: int = 240,
) -> None:
    prog = clamp(progress, 0.0, 1.0)
    if alpha <= 0.01 or prog <= 0.005:
        return
    x_end = lerp(vp.x_min, vp.x_max, prog)
    xs = np.linspace(vp.x_min, x_end, max(12, int(num_samples * prog)))
    ys = fn(xs)
    pts = []
    for xv, yv in zip(xs, ys):
        yc = clamp(float(yv), vp.y_min, vp.y_max)
        sx, sy = vp.to_px(float(xv), yc)
        pts.append([int(round(sx)), int(round(sy))])
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            canvas,
            [arr],
            isClosed=False,
            color=scale_color(color, alpha),
            thickness=max(1, cfg.px(thickness_4k)),
            lineType=cv2.LINE_AA,
        )


@dataclass
class Camera3D:
    """
    True 3D mathematical projection camera (orthographic + perspective depth)
    for 3Blue1Brown-style 3D surfaces, 3D axes, and 3D point clouds.
    """
    cx: float
    cy: float
    scale_px: float
    yaw: float = 0.65
    pitch: float = 0.52
    fov_dist: float = 6.5

    def project(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        # Rotate around vertical Z axis by yaw, then tilt by pitch
        cyw, syw = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x1 = x * cyw - y * syw
        y1 = x * syw + y * cyw
        z1 = z
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        persp = self.fov_dist / max(1.5, self.fov_dist + y2 * 0.35)
        sx = self.cx + x1 * self.scale_px * persp
        sy = self.cy - z2 * self.scale_px * persp
        return (sx, sy, y2)


def draw_3d_axes(
    canvas: np.ndarray,
    cam: Camera3D,
    cfg: RenderConfig,
    alpha: float = 1.0,
    axis_len: float = 2.4,
    labels: Tuple[str, str, str] = ("x", "y", "z"),
) -> None:
    if alpha <= 0.01:
        return
    o = cam.project(0.0, 0.0, 0.0)
    ex = cam.project(axis_len, 0.0, 0.0)
    ey = cam.project(0.0, axis_len, 0.0)
    ez = cam.project(0.0, 0.0, axis_len * 0.9)

    # Subtle 3D floor grid on z = -0.8
    for g in np.linspace(-2.0, 2.0, 9):
        p1 = cam.project(float(g), -2.0, -0.8)
        p2 = cam.project(float(g), 2.0, -0.8)
        q1 = cam.project(-2.0, float(g), -0.8)
        q2 = cam.project(2.0, float(g), -0.8)
        draw_aa_line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), 1.6, cfg, PALETTE.GRID_FAINT, alpha * 0.8)
        draw_aa_line(canvas, (q1[0], q1[1]), (q2[0], q2[1]), 1.6, cfg, PALETTE.GRID_FAINT, alpha * 0.8)

    draw_arrow(canvas, (o[0], o[1]), (ex[0], ex[1]), 4.0, cfg, PALETTE.AXIS_GRAY, alpha, 0.03)
    draw_arrow(canvas, (o[0], o[1]), (ey[0], ey[1]), 4.0, cfg, PALETTE.AXIS_GRAY, alpha, 0.03)
    draw_arrow(canvas, (o[0], o[1]), (ez[0], ez[1]), 4.0, cfg, PALETTE.AXIS_GRAY, alpha, 0.03)

    draw_text(canvas, labels[0], ex[0] + cfg.px(25), ex[1], 32, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=alpha)
    draw_text(canvas, labels[1], ey[0] + cfg.px(25), ey[1], 32, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=alpha)
    draw_text(canvas, labels[2], ez[0], ez[1] - cfg.px(30), 32, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=alpha)


def draw_3d_surface_wireframe(
    canvas: np.ndarray,
    cam: Camera3D,
    fn_z: Callable[[float, float], float],
    cfg: RenderConfig,
    u_range: Tuple[float, float] = (-1.8, 1.8),
    v_range: Tuple[float, float] = (-1.8, 1.8),
    grid_n: int = 17,
    alpha: float = 1.0,
    color_low: Tuple[int, int, int] = PALETTE.ACCENT_BLUE,
    color_high: Tuple[int, int, int] = PALETTE.WHITE,
) -> None:
    """Render a 3Blue1Brown-style 3D parametric surface mesh floating in black space."""
    if alpha <= 0.01:
        return
    us = np.linspace(u_range[0], u_range[1], grid_n)
    vs = np.linspace(v_range[0], v_range[1], grid_n)
    proj = np.zeros((grid_n, grid_n, 3), dtype=np.float64)
    z_vals = np.zeros((grid_n, grid_n), dtype=np.float64)

    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            zv = fn_z(float(u), float(v))
            z_vals[i, j] = zv
            proj[i, j] = cam.project(float(u), float(v), zv)

    z_min, z_max = float(z_vals.min()), max(float(z_vals.min()) + 1e-5, float(z_vals.max()))
    for i in range(grid_n):
        for j in range(grid_n - 1):
            zn = (z_vals[i, j] - z_min) / (z_max - z_min)
            c = lerp_color(color_low, color_high, zn)
            draw_aa_line(canvas, (proj[i, j, 0], proj[i, j, 1]), (proj[i, j + 1, 0], proj[i, j + 1, 1]), 2.2, cfg, c, alpha * 0.65)
    for j in range(grid_n):
        for i in range(grid_n - 1):
            zn = (z_vals[i, j] - z_min) / (z_max - z_min)
            c = lerp_color(color_low, color_high, zn)
            draw_aa_line(canvas, (proj[i, j, 0], proj[i, j, 1]), (proj[i + 1, j, 0], proj[i + 1, j, 1]), 2.2, cfg, c, alpha * 0.65)


# ==============================================================================
# 4. TIMELINE & CAPTION DATA (1800 SECONDS EXACT, ZERO OVERLAPS)
# ==============================================================================

@dataclass(frozen=True)
class CaptionSpec:
    start: float
    end: float
    line1: str
    line2: str = ""
    style: str = "narration"
    terms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneSpec:
    index: int
    slug: str
    title: str
    start_time: float
    end_time: float
    objective: str
    visual_elements: Tuple[str, ...]
    transition_behavior: str
    captions: Tuple[CaptionSpec, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


HOUSE_DATA_X = np.array([58.0, 76.0, 95.0, 115.0, 138.0, 160.0, 184.0, 208.0], dtype=np.float64)
HOUSE_DATA_Y = np.array([138.0, 152.0, 206.0, 220.0, 278.0, 292.0, 355.0, 374.0], dtype=np.float64)
UNSEEN_DATA_X = np.array([68.0, 106.0, 148.0, 172.0, 196.0], dtype=np.float64)
UNSEEN_DATA_Y = np.array([148.0, 214.0, 280.0, 326.0, 360.0], dtype=np.float64)


def compute_house_mse(w: float, b: float) -> float:
    preds = w * HOUSE_DATA_X + b
    return float(np.mean((preds - HOUSE_DATA_Y) ** 2))


def build_lecture_timeline() -> List[SceneSpec]:
    return [
        SceneSpec(
            index=1,
            slug="01_cinematic_intro",
            title="00:00–01:30 — Cinematic Introduction",
            start_time=0.0,
            end_time=90.0,
            objective="Open on pure black, reveal a single moving point, form 2D/3D coordinate axes and a curve, show the title, and pose the central question.",
            visual_elements=("Single point", "Open coordinate plane", "Smooth curve & points", "Minimal typography"),
            transition_behavior="Smooth axis transition into Scene 2.",
            captions=(
                CaptionSpec(2.0, 9.5, "In complete silence, mathematics begins with a single point.", "A lone coordinate in empty space."),
                CaptionSpec(10.0, 18.5, "When that point moves, its path traces a relationship", "between horizontal position x and vertical height y."),
                CaptionSpec(19.0, 28.0, "Around this motion, a coordinate plane emerges—", "an open space where hidden patterns become visible."),
                CaptionSpec(28.5, 38.0, "Observations appear as isolated points,", "while a continuous curve threads through them."),
                CaptionSpec(39.0, 54.0, "SILENT MATH — LECTURE #1: INTRODUCTION TO AI & MACHINE LEARNING", "Understand the mathematics. Visualize the intelligence.", "definition"),
                CaptionSpec(55.0, 67.5, "Every day, machines recognize objects, translate speech,", "and predict outcomes from raw measurements."),
                CaptionSpec(68.0, 81.0, "How can a machine learn patterns from examples?", "Hold this central question in mind as we build the mathematics.", "question"),
                CaptionSpec(81.5, 89.5, "We begin at absolute zero: what is happening geometrically", "when we speak of Artificial Intelligence?"),
            ),
        ),
        SceneSpec(
            index=2,
            slug="02_what_is_ai",
            title="01:30–04:30 — What Is Artificial Intelligence?",
            start_time=90.0,
            end_time=270.0,
            objective="Visually demonstrate Input -> Processing -> Output through open geometric shape recognition, wave pattern detection, and vector trajectory selection.",
            visual_elements=("Open 3D/2D geometric shape morph", "Signal wave & anomaly pulse", "Vector field trajectory"),
            transition_behavior="Continuous morph into Scene 3 coordinate plane.",
            captions=(
                CaptionSpec(91.0, 102.0, "Any intelligent system transforms an Input into an Output", "through mathematical processing:  x  →  f(x)  →  y", "equation"),
                CaptionSpec(103.0, 115.0, "First, a machine receives raw numerical measurements—", "such as coordinates or grid intensities."),
                CaptionSpec(116.0, 128.0, "Watch this point cloud in space:", "by measuring distances and curvature, the system recognizes a Circle."),
                CaptionSpec(129.0, 141.0, "As the coordinates shift into three sharp vertices,", "the geometric signature changes to a Triangle, and then a Square."),
                CaptionSpec(142.0, 154.0, "The input is raw coordinates; the processing measures geometry;", "the output is the recognized structure."),
                CaptionSpec(155.0, 168.0, "In a second example, a system tracks a continuous signal over time,", "uncovering a smooth periodic wave."),
                CaptionSpec(169.0, 182.0, "When an unexpected spike breaks the rhythm,", "measuring the deviation immediately isolates the anomaly."),
                CaptionSpec(183.0, 196.0, "In a third example, a system explores possible paths through space", "from an initial state to a target coordinate."),
                CaptionSpec(197.0, 210.0, "Comparing candidate trajectories reveals the smooth optimal path", "that curves cleanly around obstacles."),
                CaptionSpec(211.0, 225.0, "Artificial Intelligence (AI) is the broad pursuit of building systems", "that perform tasks associated with intelligent behavior.", "definition"),
                CaptionSpec(226.0, 240.0, "Not all AI systems learn from data.", "Early AI relied on humans writing rigid, explicit rules."),
                CaptionSpec(241.0, 255.0, "Explicit rules work for fixed logic,", "but real-world patterns are rarely rigid."),
                CaptionSpec(256.0, 269.0, "When rules are too subtle to write by hand,", "we turn to Machine Learning."),
            ),
        ),
        SceneSpec(
            index=3,
            slug="03_what_is_ml",
            title="04:30–08:00 — What Is Machine Learning?",
            start_time=270.0,
            end_time=480.0,
            objective="Show why manual step-rules fail, let a smooth curve emerge from raw points, and visualize Data -> Model -> Prediction on the open graph.",
            visual_elements=("Rigid step function vs smooth curve", "Data scatter points", "Prediction projection & residual gap"),
            transition_behavior="Scale coordinate axes into house area and price.",
            captions=(
                CaptionSpec(271.0, 285.0, "Manual IF-THEN rules carve space into rigid staircases.", "They jump abruptly and miss smooth relationships."),
                CaptionSpec(286.0, 300.0, "Look at how poorly a rigid staircase approximates", "observations that curve gently across the plane."),
                CaptionSpec(301.0, 316.0, "Machine Learning takes a different path:", "instead of hand-coding rules, we let a curve adapt to examples.", "definition"),
                CaptionSpec(317.0, 332.0, "Start with an empty plane where only raw data points appear.", "Each glowing dot is one observed example."),
                CaptionSpec(333.0, 348.0, "At first, there is no curve—only isolated coordinates.", "Yet their collective arrangement suggests a clear trajectory."),
                CaptionSpec(349.0, 365.0, "Watch a continuous curve emerge and bend smoothly", "until it threads through the heart of the data."),
                CaptionSpec(366.0, 382.0, "Three core concepts appear directly on this graph:", "Data, Model, and Prediction.", "definition"),
                CaptionSpec(383.0, 399.0, "DATA is the set of observed white points (xᵢ, yᵢ).", "MODEL is the blue mathematical curve f(x)."),
                CaptionSpec(400.0, 416.0, "PREDICTION is the height ŷ on the curve", "for any chosen input x."),
                CaptionSpec(417.0, 433.0, "Follow the projection:  Data  →  Model  →  Prediction.", "An input x rises to meet the model curve at ŷ = f(x).", "equation"),
                CaptionSpec(434.0, 450.0, "Notice the vertical segment between the curve and the true point:", "a prediction is a model's estimate, not guaranteed perfection."),
                CaptionSpec(451.0, 466.0, "Even a well-fitted curve leaves small vertical gaps", "because real measurements contain natural variation."),
                CaptionSpec(467.0, 479.0, "Let us make this concrete with a real prediction problem:", "estimating house prices from floor area."),
            ),
        ),
        SceneSpec(
            index=4,
            slug="04_data_features_patterns",
            title="08:00–11:30 — Data, Features, and Patterns",
            start_time=480.0,
            end_time=690.0,
            objective="Map 8 houses onto the (Area, Price) coordinate plane, define Feature, Target, and Training Example via geometric projections, and show noise.",
            visual_elements=("Minimal house wireframes morphing into points", "Orthogonal dashed projections", "Trend band"),
            transition_behavior="Keep axes fixed and introduce y = wx + b.",
            captions=(
                CaptionSpec(481.0, 496.0, "Consider eight homes, each with a known floor area", "and a recorded sale price."),
                CaptionSpec(497.0, 512.0, "Horizontal position x measures House Area in square meters (m²).", "Vertical height y measures Price in thousands of dollars ($k)."),
                CaptionSpec(513.0, 528.0, "Watch each minimal house wireframe settle at its exact coordinate", "(Area xᵢ, Price yᵢ) on the plane."),
                CaptionSpec(529.0, 545.0, "A compact 58 m² home sits lower-left at $138k,", "while a spacious 208 m² home sits upper-right at $374k."),
                CaptionSpec(546.0, 562.0, "In Machine Learning, the input measurement along the horizontal axis", "is called a FEATURE (x).", "definition"),
                CaptionSpec(563.0, 579.0, "The vertical quantity we aim to predict", "is called the TARGET (y).", "definition"),
                CaptionSpec(580.0, 596.0, "A single paired coordinate (xᵢ, yᵢ) combining Feature and Target", "is one TRAINING EXAMPLE.", "definition"),
                CaptionSpec(597.0, 613.0, "Look at training example (138 m², $278k):", "its horizontal projection is the Feature; its vertical height is the Target."),
                CaptionSpec(614.0, 630.0, "Across all eight training examples,", "larger floor areas clearly correspond to higher prices."),
                CaptionSpec(631.0, 647.0, "Yet the points do not fall on a perfectly straight line—", "real-world data siempre carries natural scatter and noise."),
                CaptionSpec(648.0, 665.0, "Location, sunlight, and age pull individual prices", "slightly above or below the central trend."),
                CaptionSpec(666.0, 678.0, "We do not want a jagged line that chases random noise;", "we want a clean mathematical line that captures the trend."),
                CaptionSpec(679.0, 689.5, "What is the simplest equation that draws a straight line?", "Meet the linear model:  y = wx + b.", "equation"),
            ),
        ),
        SceneSpec(
            index=5,
            slug="05_linear_model_y_wx_b",
            title="11:30–15:00 — The Mathematical Heart: y = wx + b",
            start_time=690.0,
            end_time=900.0,
            objective="Animate y = wx + b on an open plane, sweeping weight w (slope triangle) and bias b (vertical shift) in exact mathematical lockstep.",
            visual_elements=("Floating equation y = wx + b", "Live slope triangle Δy/Δx = w", "Intercept marker (0, b)"),
            transition_behavior="Morph linear operation into a computational graph node.",
            captions=(
                CaptionSpec(691.0, 705.0, "At the heart of linear models lies a four-symbol equation:", "y = w · x + b", "equation"),
                CaptionSpec(706.0, 720.0, "Every symbol has a direct geometric meaning on this plane.", "Watch how the line responds to each component."),
                CaptionSpec(721.0, 735.0, "x is the INPUT feature along the horizontal axis.", "y is the PREDICTED OUTPUT height on the line."),
                CaptionSpec(736.0, 751.0, "w is the WEIGHT—geometrically, the slope of the line,", "controlling how strongly input x influences output y.", "definition"),
                CaptionSpec(752.0, 767.0, "b is the BIAS—the vertical intercept at x = 0,", "shifting the entire line up or down.", "definition"),
                CaptionSpec(768.0, 782.0, "Look at the right triangle attached to the line:", "for a horizontal step Δx, the vertical rise is Δy = w · Δx."),
                CaptionSpec(783.0, 799.0, "Hold bias b = 0 and smoothly increase the weight w.", "As w grows from 0.0 to 2.3, the line rotates steeper upward."),
                CaptionSpec(800.0, 816.0, "When w = 0, the line is flat—input x has zero influence.", "When w is large, a small change in x produces a large change in y."),
                CaptionSpec(817.0, 832.0, "When w becomes negative, the slope tilts downward:", "increasing x now reduces the predicted output y."),
                CaptionSpec(833.0, 849.0, "Now lock the weight at w = 1.65 and sweep the bias b.", "Changing b translates the line vertically without altering its tilt."),
                CaptionSpec(850.0, 865.0, "Weight w controls rotation; Bias b controls vertical elevation.", "Together, these two numbers define every possible straight line."),
                CaptionSpec(866.0, 883.0, "When w = 1.65 and b = 38, our line y = 1.65x + 38", "threads cleanly through the eight house-price examples.", "equation"),
                CaptionSpec(884.0, 899.5, "A single straight line is simple,", "but what happens when we connect many such units together?"),
            ),
        ),
        SceneSpec(
            index=6,
            slug="06_neural_network",
            title="15:00–18:30 — From a Simple Model to a Neural Network",
            start_time=900.0,
            end_time=1110.0,
            objective="Expand a single linear unit into an open 3Blue1Brown-style 3-layer Neural Network with luminous signal pulses and a smooth activation curve.",
            visual_elements=("Open 3-layer neural graph (3 -> 4 -> 1)", "Traveling light pulses", "Smooth sigmoid/ReLU curve"),
            transition_behavior="Transition into 3D Loss surface and 2D line fit.",
            captions=(
                CaptionSpec(901.0, 916.0, "We can picture y = wx + b as a single input circle x", "connected by a weighted line w to an output circle y."),
                CaptionSpec(917.0, 932.0, "Real problems rarely depend on a single feature,", "and their patterns curve in ways a straight line cannot match."),
                CaptionSpec(933.0, 949.0, "To capture richer patterns, we expand our single unit", "into an interconnected web of mathematical nodes."),
                CaptionSpec(950.0, 966.0, "On the left, three nodes form the INPUT LAYER,", "receiving features x₁, x₂, and x₃.", "definition"),
                CaptionSpec(967.0, 983.0, "In the center, four nodes form a HIDDEN LAYER.", "Every input connects to every hidden neuron with its own weight wᵢⱼ.", "definition"),
                CaptionSpec(984.0, 1000.0, "On the right, the hidden neurons combine into the OUTPUT LAYER,", "producing the final prediction ŷ.", "definition"),
                CaptionSpec(1001.0, 1017.0, "Each circle is an artificial NEURON.", "Each connecting line carries a numerical WEIGHT.", "definition"),
                CaptionSpec(1018.0, 1034.0, "Watch luminous signals travel from left to right:", "inputs are multiplied by weights and summed inside each neuron."),
                CaptionSpec(1035.0, 1051.0, "Crucially, each neuron applies a smooth non-linear activation curve", "σ(w·x + b), bending straight lines into flexible curves.", "equation"),
                CaptionSpec(1052.0, 1068.0, "By stacking neurons into layers,", "a neural network can sculpt almost any continuous curve or surface."),
                CaptionSpec(1069.0, 1085.0, "Yet there is no mystery inside the network:", "it is pure linear algebra—multiplications, additions, and smooth curves."),
                CaptionSpec(1086.0, 1098.0, "Neural networks are one expressive family of machine-learning models,", "built from the same weights and biases we just explored."),
                CaptionSpec(1099.0, 1109.5, "How does a model actually find the right weights and biases?", "Let us visualize the geometry of learning in 3D.", "question"),
            ),
        ),
        SceneSpec(
            index=7,
            slug="07_how_a_model_learns",
            title="18:30–22:00 — How Does a Model Learn?",
            start_time=1110.0,
            end_time=1320.0,
            objective="Pair the 2D data fit and vertical error residuals with a true 3Blue1Brown-style 3D Loss Surface bowl L(w, b) and descending parameter trajectory.",
            visual_elements=("Left: 2D data & gold residual lines", "Right: 3D wireframe Loss bowl L(w, b)", "Descending trajectory ball"),
            transition_behavior="Transition into 3 learning paradigms.",
            captions=(
                CaptionSpec(1111.0, 1125.0, "Learning begins with an uncalibrated guess:", "here, w = 0.35 and b = 85 produce a flat line far from our data."),
                CaptionSpec(1126.0, 1140.0, "For each data point, look at the vertical gold segment", "stretching between the model's prediction ŷ and the true target y."),
                CaptionSpec(1141.0, 1156.0, "That vertical distance (ŷᵢ - yᵢ) is the prediction ERROR", "on a single training example.", "definition"),
                CaptionSpec(1157.0, 1172.0, "Averaging the squared vertical errors across all points", "gives a single number called the LOSS:  L(w, b).", "equation"),
                CaptionSpec(1173.0, 1189.0, "On the right, behold the LOSS SURFACE in three dimensions:", "horizontal axes are parameters (w, b); vertical height is Loss L(w, b)."),
                CaptionSpec(1190.0, 1206.0, "Every possible pair of parameters (w, b) corresponds to a point", "on this curved 3D valley."),
                CaptionSpec(1207.0, 1223.0, "High up on the rim of the 3D bowl, the Loss is large", "and the 2D line fits the data poorly."),
                CaptionSpec(1224.0, 1240.0, "At the bottom of the 3D valley, the Loss reaches its minimum", "and the 2D line threads cleanly through the points."),
                CaptionSpec(1241.0, 1257.0, "TRAINING is the iterative loop:", "Prediction → Measure Error → Update Parameters (w, b) → New Prediction.", "equation"),
                CaptionSpec(1258.0, 1274.0, "Watch the gold point roll down the 3D loss surface", "as we update w and b step by step."),
                CaptionSpec(1275.0, 1291.0, "If a step is slightly too large, the point can briefly overshoot", "to the opposite wall of the valley before correcting."),
                CaptionSpec(1292.0, 1306.0, "As the 3D point settles into the valley floor,", "the vertical error bars on the 2D graph shrink to a minimum."),
                CaptionSpec(1307.0, 1319.5, "By following the geometry of the loss surface downward,", "the machine calibrates its own parameters from data."),
            ),
        ),
        SceneSpec(
            index=8,
            slug="08_three_learning_approaches",
            title="22:00–25:00 — Three Important Learning Approaches",
            start_time=1320.0,
            end_time=1500.0,
            objective="Present Supervised, Unsupervised (3D/2D clustering), and Reinforcement Learning sequentially on a clean, unboxed open canvas.",
            visual_elements=("Phase 1: Supervised curve fit", "Phase 2: 3D/2D Unsupervised cluster separation", "Phase 3: RL vector field"),
            transition_behavior="Morph into Training vs. Generalization coordinate plane.",
            captions=(
                CaptionSpec(1321.0, 1336.0, "Depending on the feedback available, machine learning", "naturally unfolds into three major approaches."),
                CaptionSpec(1337.0, 1358.0, "1. SUPERVISED LEARNING: every input xᵢ is paired with a known target yᵢ,", "guiding the curve directly toward known answers.", "definition"),
                CaptionSpec(1359.0, 1379.0, "Whether fitting a continuous line or separating two classes,", "known targets provide direct error signals at every point."),
                CaptionSpec(1380.0, 1400.0, "2. UNSUPERVISED LEARNING: what if we have a 3D cloud of points", "with no target labels at all?", "definition"),
                CaptionSpec(1401.0, 1420.0, "Watch the unlabeled coordinates self-organize in space:", "points with similar geometry gravitate into natural clusters."),
                CaptionSpec(1421.0, 1439.0, "Unsupervised learning discovers hidden structure and groupings", "purely from the mutual distances between data points."),
                CaptionSpec(1440.0, 1460.0, "3. REINFORCEMENT LEARNING: an Agent moves through a state space,", "choosing actions along a vector field.", "definition"),
                CaptionSpec(1461.0, 1480.0, "By receiving numeric rewards (+10 at the goal, -1 near hazards),", "the agent reshapes its trajectory toward higher cumulative reward.", "equation"),
                CaptionSpec(1481.0, 1499.5, "Across all three approaches, one question decides success:", "did the model learn the true pattern, or merely memorize the dots?"),
            ),
        ),
        SceneSpec(
            index=9,
            slug="09_training_vs_generalization",
            title="25:00–27:30 — Training vs. Generalization",
            start_time=1500.0,
            end_time=1650.0,
            objective="Compare an oscillating degree-7 polynomial (Overfitting) with a smooth line (Generalization) on a single spacious coordinate plane with unseen data.",
            visual_elements=("Solid white training points", "Oscillating coral polynomial curve", "Smooth blue line", "Hollow gold unseen points"),
            transition_behavior="Fade axes into open constellation roadmap.",
            captions=(
                CaptionSpec(1501.0, 1516.0, "Look at our eight solid white TRAINING DATA points—", "the examples the model sees during training.", "definition"),
                CaptionSpec(1517.0, 1533.0, "If we use a high-degree polynomial with excessive flexibility,", "it can twist wildly to pass through every single training point."),
                CaptionSpec(1534.0, 1550.0, "On the training points alone, this oscillating coral curve", "achieves zero error—yet it swings violently between them."),
                CaptionSpec(1551.0, 1567.0, "Compare it with the smooth blue line,", "which accepts small training gaps to follow the underlying trend."),
                CaptionSpec(1568.0, 1584.0, "Chasing the random noise of training data is called OVERFITTING.", "The model has memorized coordinates rather than learning the relationship.", "definition"),
                CaptionSpec(1585.0, 1602.0, "Now reveal UNSEEN DATA—the hollow gold circles", "representing new homes the model never saw during training.", "definition"),
                CaptionSpec(1603.0, 1620.0, "Look at the tall vertical error lines to the oscillating curve:", "it misses unseen data badly."),
                CaptionSpec(1621.0, 1636.0, "Meanwhile, the smooth trend line stays close to every unseen point.", "Performing accurately on new, unseen examples is GENERALIZATION.", "definition"),
                CaptionSpec(1637.0, 1649.5, "True intelligence is never about memorizing the past;", "it is about generalizing reliably to the unseen."),
            ),
        ),
        SceneSpec(
            index=10,
            slug="10_roadmap_next",
            title="27:30–29:00 — What Will We Learn Next?",
            start_time=1650.0,
            end_time=1740.0,
            objective="Display an open, unboxed geometric constellation roadmap from Mathematics to Deep Learning and preview PyTorch.",
            visual_elements=("Open constellation nodes & curved arcs", "Micro geometric icons above nodes", "PyTorch future beacon"),
            transition_behavior="Return to the 3D/2D parametric curve for Scene 11.",
            captions=(
                CaptionSpec(1651.0, 1665.0, "You now have the geometric intuition of AI and Machine Learning.", "Where does our journey on Silent Math lead next?"),
                CaptionSpec(1666.0, 1680.0, "Follow the path across five connected mathematical milestones:", "Mathematics → Python → Machine Learning → Neural Networks → Deep Learning.", "equation"),
                CaptionSpec(1681.0, 1695.0, "First, we master the MATHEMATICS—vectors, matrices, and derivatives—", "visualized in 2D and 3D space."),
                CaptionSpec(1696.0, 1710.0, "Next, we express those equations in clean PYTHON,", "building classical Machine Learning and Neural Networks from scratch."),
                CaptionSpec(1711.0, 1726.0, "As our architectures scale into Deep Learning, we will unlock:", "PyTorch — Coming in a Future Lecture.", "definition"),
                CaptionSpec(1727.0, 1739.5, "PyTorch will turn our mathematical graphs and surfaces", "into high-speed tensor computation."),
            ),
        ),
        SceneSpec(
            index=11,
            slug="11_recap_closing",
            title="29:00–30:00 — Recap and Closing",
            start_time=1740.0,
            end_time=1800.0,
            objective="Return to the moving point and curve in open space, float the 6 recap insights cleanly one by one, pose the final question, and fade to black at 30:00.",
            visual_elements=("Open 3D/2D parametric curve & glowing point", "Unboxed floating recap typography", "Fade to black at 1800.0s"),
            transition_behavior="Fade smoothly to pure black (#000000) at 1800.0s.",
            captions=(
                CaptionSpec(1741.0, 1749.0, "AI: Intelligent behavior in machines.", "ML: Learning patterns from data.", "definition"),
                CaptionSpec(1749.5, 1758.0, "Model: A mathematical system that produces predictions.", "Loss: A measure of prediction error.", "definition"),
                CaptionSpec(1758.5, 1767.5, "Training: Adjusting parameters using data.", "Generalization: Performing well on unseen examples.", "definition"),
                CaptionSpec(1768.5, 1782.0, "If a machine can learn a pattern from examples,", "what else can we teach it to discover?", "question"),
                CaptionSpec(1783.0, 1796.5, "SILENT MATH — LECTURE #1 COMPLETE", "Next: Mathematics Behind Machine Learning", "definition"),
            ),
        ),
    ]


LECTURE_SCENES: List[SceneSpec] = build_lecture_timeline()


def get_scene_for_time(t: float) -> SceneSpec:
    t_clamped = clamp(t, 0.0, 1800.0 - 1e-6)
    for scene in LECTURE_SCENES:
        if scene.start_time <= t_clamped < scene.end_time:
            return scene
    return LECTURE_SCENES[-1]


def get_active_caption(scene: SceneSpec, t: float) -> Optional[CaptionSpec]:
    for cap in scene.captions:
        if cap.start <= t <= cap.end:
            return cap
    return None


# ==============================================================================
# 5. UNBOXED 3BLUE1BROWN FLOATING CAPTION COMPOSITOR
# ==============================================================================

def draw_floating_captions(
    canvas: np.ndarray,
    scene: SceneSpec,
    t: float,
    cfg: RenderConfig,
) -> None:
    """
    Render minimalist floating captions in the bottom margin with ZERO boxes,
    ZERO borders, and ZERO background panels—purely crisp antialiased text on black.
    """
    W, H = cfg.width, cfg.height
    master_fade = 1.0
    if t < 1.5:
        master_fade = smoothstep(0.0, 1.5, t)
    elif t > 1796.5:
        master_fade = 1.0 - smoothstep(1796.5, 1799.6, t)

    cap = get_active_caption(scene, t)
    if cap is None:
        return

    env = window_envelope(t, cap.start, cap.end, fade_in=0.45, fade_out=0.45) * master_fade
    if env <= 0.02:
        return

    y_float = cfg.px(14.0 * (1.0 - smoothstep(cap.start, cap.start + 0.5, t)))
    y1 = H - cfg.px(195) + y_float
    y2 = H - cfg.px(115) + y_float

    cap_dur = max(1.0, cap.end - cap.start)
    reveal_u = clamp((t - cap.start) / max(0.25, min(1.6, cap_dur * 0.30)), 0.0, 1.0)

    words1 = cap.line1.split(" ")
    n1 = max(1, int(math.ceil(len(words1) * clamp(reveal_u * 1.35, 0.15, 1.0))))
    text1 = " ".join(words1[:n1])

    col1 = PALETTE.WHITE
    role1 = "sans"
    if cap.style == "equation":
        col1 = PALETTE.ACCENT_BLUE
        role1 = "mono"
    elif cap.style == "question":
        col1 = PALETTE.ACCENT_GOLD

    draw_text(canvas, text1, W * 0.5, y1 if cap.line2 else (y1 + y2) * 0.5, 44, cfg, col1, role1, bold=True, alpha=env)

    if cap.line2:
        u2 = clamp((reveal_u - 0.22) / 0.78, 0.0, 1.0)
        if u2 > 0.02:
            words2 = cap.line2.split(" ")
            n2 = max(1, int(math.ceil(len(words2) * u2)))
            text2 = " ".join(words2[:n2])
            draw_text(
                canvas,
                text2,
                W * 0.5,
                y2,
                38,
                cfg,
                PALETTE.GRAY_LIGHT,
                "sans",
                bold=False,
                alpha=env * smoothstep(0.0, 0.25, u2),
            )


# ==============================================================================
# 6. PURE 3BLUE1BROWN SCENE RENDERERS (2D GRAPHS & 3D SURFACES, ZERO BOXES)
# ==============================================================================

def render_scene_01_intro(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    00:00–01:30 — Cinematic Introduction
    Pure open black space -> glowing white point -> open coordinate plane -> smooth curve
    & data points -> floating minimal title typography -> central question.
    """
    W, H = cfg.width, cfg.height
    vp = PlotViewport(0.0, 10.0, 0.0, 10.0, cfg.px(420), cfg.px(220), W - cfg.px(420), cfg.px(1680))

    title_dim = 1.0 - 0.65 * window_envelope(t, 38.0, 66.0, 1.8, 1.8)
    grid_alpha = smoothstep(9.0, 21.0, t) * title_dim

    if grid_alpha > 0.01:
        draw_3b1b_axes(
            canvas, vp, cfg, alpha=grid_alpha,
            x_ticks=[2, 4, 6, 8, 10], y_ticks=[2, 4, 6, 8, 10],
            x_label="x", y_label="y",
        )

    curve_fn = lambda x: 1.8 + 0.62 * x + 1.15 * np.sin(0.58 * x)
    curve_prog = smootherstep(18.0, 34.0, t)
    if curve_prog > 0.01:
        draw_curve(canvas, vp, curve_fn, cfg, PALETTE.ACCENT_BLUE, 6.5, grid_alpha, progress=curve_prog)

    sample_xs = [1.2, 2.5, 3.8, 5.2, 6.5, 7.9, 9.1]
    sample_dy = [-0.4, 0.45, -0.32, 0.5, -0.35, 0.32, -0.25]
    for i, (sx_v, dy) in enumerate(zip(sample_xs, sample_dy)):
        pa = smoothstep(25.0 + i * 1.1, 26.5 + i * 1.1, t) * grid_alpha
        if pa > 0.01:
            yv = float(curve_fn(np.array([sx_v]))[0]) + dy
            px, py = vp.to_px(sx_v, yv)
            cx_p, cy_p = vp.to_px(sx_v, float(curve_fn(np.array([sx_v]))[0]))
            draw_dashed_line(canvas, (px, py), (cx_p, cy_p), 2.5, cfg, PALETTE.GRAY_MID, pa * 0.75)
            draw_glow_point(canvas, px, py, 11.0, cfg, PALETTE.WHITE, pa, glow_layers=2)

    pt_alpha = smoothstep(1.5, 4.0, t)
    if pt_alpha > 0.01:
        xt = 5.0 + 3.8 * math.sin(0.38 * max(0.0, t - 3.0))
        yt = float(curve_fn(np.array([xt]))[0])
        px, py = vp.to_px(xt, yt)
        b_plane = smoothstep(3.0, 14.0, t)
        cx = lerp(W * 0.5, px, b_plane)
        cy = lerp(H * 0.45, py, b_plane)
        draw_glow_point(canvas, cx, cy, 16.0, cfg, PALETTE.WHITE, pt_alpha * title_dim, glow_layers=4)
        if grid_alpha > 0.3 and not (38.0 <= t <= 66.0):
            draw_text(canvas, f"({xt:.1f}, {yt:.1f})", cx + cfg.px(28), cy - cfg.px(28), 32, cfg, PALETTE.WHITE, "mono", alpha=grid_alpha, anchor="midleft")

    # Minimal floating Opening Title (no boxes!)
    t_env = window_envelope(t, 38.0, 66.5, 1.8, 1.8)
    if t_env > 0.01:
        draw_text(canvas, "SILENT MATH", W * 0.5, cfg.px(640), 104, cfg, PALETTE.WHITE, "sans", bold=True, alpha=t_env)
        draw_aa_line(canvas, (W * 0.5 - cfg.px(320), cfg.px(750)), (W * 0.5 + cfg.px(320), cfg.px(750)), 2.5, cfg, PALETTE.GRAY_MID, t_env)
        draw_text(canvas, "LECTURE #1", W * 0.5, cfg.px(845), 50, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=t_env * smoothstep(40.0, 42.5, t))
        draw_text(canvas, "INTRODUCTION TO AI & MACHINE LEARNING", W * 0.5, cfg.px(980), 62, cfg, PALETTE.WHITE, "sans", bold=True, alpha=t_env * smoothstep(42.0, 45.0, t))
        draw_text(canvas, "Understand the mathematics. Visualize the intelligence.", W * 0.5, cfg.px(1120), 40, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=t_env * smoothstep(45.0, 48.5, t))

    # Central Question floating cleanly in open space
    q_env = window_envelope(t, 67.0, 89.5, 1.5, 1.5)
    if q_env > 0.01:
        draw_text(canvas, "How can a machine learn patterns from examples?", W * 0.5, cfg.px(230), 56, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=q_env)


def render_scene_02_what_is_ai(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    01:30–04:30 — What Is Artificial Intelligence?
    Pure geometric visualizations on open coordinate planes / 3D space (zero boxes):
    - 90..154s : 3D/2D geometric point cloud morphing between Circle, Triangle, and Square on open axes.
    - 154..183s: Continuous wave signal on open axes with anomaly spike detection.
    - 183..225s: Vector field and smooth optimal trajectory curving around an obstacle circle.
    - 225..270s: Side-by-side open coordinate graphs comparing Rigid Step Rules vs Smooth Learned Curve.
    """
    W, H = cfg.width, cfg.height
    local_t = t - 90.0
    env = window_envelope(t, 90.0, 270.0, 1.0, 1.0)

    # Subtle floating equation at top: Input x  --->  f(x)  --->  Output y
    draw_text(canvas, "Input x     ───►     f(x)     ───►     Output y", W * 0.5, cfg.px(130), 42, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=env * 0.85)

    # Phase 1 (90..154s): Geometric Object Recognition via 3D rotating point ring/polygon
    p1_env = window_envelope(t, 90.0, 154.5, 1.0, 1.2)
    if p1_env > 0.01:
        cam = Camera3D(W * 0.5, H * 0.47, cfg.px(320), yaw=0.35 * math.sin(local_t * 0.35), pitch=0.38)
        draw_3d_axes(canvas, cam, cfg, alpha=p1_env * 0.75, axis_len=2.2, labels=("x₁", "x₂", "z"))

        # Morph smoothly between Circle (0), Triangle (1), Square (2)
        cycle = (local_t / 14.0) % 3.0
        shape_idx = int(cycle)
        shape_names = ["Recognized Structure: CIRCLE", "Recognized Structure: TRIANGLE", "Recognized Structure: SQUARE"]
        n_pts = 48
        pts_2d = []
        for k in range(n_pts):
            theta = (2.0 * math.pi * k) / n_pts
            # Circle radius
            r_circ = 1.55
            # Square radius
            r_sq = 1.45 / max(abs(math.cos(theta)), abs(math.sin(theta)))
            # Triangle radius
            ang_tri = ((theta + math.pi / 6.0) % (2.0 * math.pi / 3.0)) - (math.pi / 3.0)
            r_tri = 1.15 / max(0.25, math.cos(ang_tri))

            if shape_idx == 0:
                r = lerp(r_circ, r_tri, smoothstep(0.7, 1.0, cycle - 0.0))
            elif shape_idx == 1:
                r = lerp(r_tri, r_sq, smoothstep(0.7, 1.0, cycle - 1.0))
            else:
                r = lerp(r_sq, r_circ, smoothstep(0.7, 1.0, cycle - 2.0))

            xv, yv = r * math.cos(theta), r * math.sin(theta)
            sx, sy, _ = cam.project(xv, yv, 0.0)
            pts_2d.append((sx, sy))

        for k in range(n_pts):
            p_a = pts_2d[k]
            p_b = pts_2d[(k + 1) % n_pts]
            draw_aa_line(canvas, p_a, p_b, 4.5, cfg, PALETTE.ACCENT_BLUE, p1_env)
            if k % 2 == 0:
                draw_glow_point(canvas, p_a[0], p_a[1], 8.0, cfg, PALETTE.WHITE, p1_env, glow_layers=2)

        draw_text(canvas, shape_names[shape_idx], W * 0.5, cfg.px(250), 46, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=p1_env)

    # Phase 2 (154..183s): Continuous Wave Pattern & Anomaly Spike on Open Axes
    p2_env = window_envelope(t, 154.0, 183.5, 1.0, 1.0)
    if p2_env > 0.01:
        vp = PlotViewport(0.0, 12.0, -2.5, 2.5, cfg.px(380), cfg.px(280), W - cfg.px(380), cfg.px(1620))
        draw_3b1b_axes(canvas, vp, cfg, alpha=p2_env, x_ticks=[2, 4, 6, 8, 10, 12], y_ticks=[-2, -1, 1, 2], x_label="t", y_label="s(t)")
        phase = local_t * 0.9
        wave_fn = lambda x: 1.3 * np.sin(1.35 * x - phase)
        draw_curve(canvas, vp, wave_fn, cfg, PALETTE.ACCENT_BLUE, 6.0, p2_env)
        for sx in np.linspace(0.6, 11.4, 25):
            is_anom = abs(sx - 8.1) < 0.22
            yv = float(wave_fn(np.array([sx]))[0]) + (1.15 if is_anom else 0.0)
            px, py = vp.to_px(float(sx), clamp(yv, -2.3, 2.3))
            base_x, base_y = vp.to_px(float(sx), float(wave_fn(np.array([sx]))[0]))
            if is_anom:
                draw_dashed_line(canvas, (base_x, base_y), (px, py), 3.5, cfg, PALETTE.ACCENT_GOLD, p2_env)
                draw_text(canvas, "Anomaly Δs", px + cfg.px(24), py - cfg.px(20), 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=p2_env, anchor="midleft")
            draw_glow_point(canvas, px, py, 14.0 if is_anom else 8.5, cfg, PALETTE.ACCENT_GOLD if is_anom else PALETTE.WHITE, p2_env)

    # Phase 3 (183..225s): Optimal Action Trajectory in Open Vector Field
    p3_env = window_envelope(t, 183.0, 225.5, 1.0, 1.0)
    if p3_env > 0.01:
        vp = PlotViewport(0.0, 10.0, 0.0, 6.0, cfg.px(380), cfg.px(280), W - cfg.px(380), cfg.px(1620))
        draw_3b1b_axes(canvas, vp, cfg, alpha=p3_env, x_ticks=[2, 4, 6, 8, 10], y_ticks=[1, 2, 3, 4, 5], x_label="x", y_label="y")
        # Circular obstacle potential in the center
        ox, oy = vp.to_px(5.0, 3.0)
        draw_glow_point(canvas, ox, oy, 95.0, cfg, PALETTE.ACCENT_CORAL, p3_env * 0.55, glow_layers=3, filled=False)
        draw_text(canvas, "Obstacle", ox, oy, 32, cfg, PALETTE.ACCENT_CORAL, "mono", alpha=p3_env)

        opt_fn = lambda x: 3.0 + 1.95 * np.sin(math.pi * (x - 1.0) / 8.0)
        sub_fn = lambda x: 3.0 - 2.15 * np.sin(math.pi * (x - 1.0) / 8.0)
        draw_curve(canvas, vp, sub_fn, cfg, PALETTE.GRAY_DARK, 4.0, p3_env * 0.7)
        draw_curve(canvas, vp, opt_fn, cfg, PALETTE.ACCENT_BLUE, 7.0, p3_env)

        sx, sy = vp.to_px(1.0, 3.0)
        gx, gy = vp.to_px(9.0, 3.0)
        draw_glow_point(canvas, sx, sy, 15.0, cfg, PALETTE.WHITE, p3_env)
        draw_glow_point(canvas, gx, gy, 17.0, cfg, PALETTE.ACCENT_GOLD, p3_env)

        u_ag = ((local_t - 93.0) * 0.22) % 1.0
        ax_v = lerp(1.0, 9.0, u_ag)
        ay_v = float(opt_fn(np.array([ax_v]))[0])
        apx, apy = vp.to_px(ax_v, ay_v)
        draw_glow_point(canvas, apx, apy, 15.0, cfg, PALETTE.ACCENT_GOLD, p3_env, glow_layers=4)

    # Phase 4 (225..270s): Open Dual Axes — Explicit Step Rules vs Smooth Learned Curve
    p4_env = window_envelope(t, 225.0, 270.0, 1.0, 1.0)
    if p4_env > 0.01:
        vp_l = PlotViewport(0.0, 10.0, 0.0, 10.0, cfg.px(280), cfg.px(360), cfg.px(1740), cfg.px(1580))
        vp_r = PlotViewport(0.0, 10.0, 0.0, 10.0, cfg.px(2100), cfg.px(360), W - cfg.px(280), cfg.px(1580))
        draw_3b1b_axes(canvas, vp_l, cfg, alpha=p4_env, x_ticks=[2, 5, 8], y_ticks=[2, 5, 8], x_label="x", y_label="y")
        draw_3b1b_axes(canvas, vp_r, cfg, alpha=p4_env, x_ticks=[2, 5, 8], y_ticks=[2, 5, 8], x_label="x", y_label="y")

        stair_fn = lambda x: np.floor(x / 2.2) * 2.0 + 1.8
        smooth_fn = lambda x: 1.5 + 0.65 * x + 0.02 * (x ** 2)
        draw_curve(canvas, vp_l, stair_fn, cfg, PALETTE.ACCENT_CORAL, 5.5, p4_env)
        draw_curve(canvas, vp_r, smooth_fn, cfg, PALETTE.ACCENT_BLUE, 6.5, p4_env, progress=smoothstep(235.0, 250.0, t))

        for xv in [1.5, 3.0, 4.5, 6.0, 7.5, 9.0]:
            yv = float(smooth_fn(np.array([xv]))[0])
            p_l = vp_l.to_px(xv, yv)
            p_r = vp_r.to_px(xv, yv)
            draw_glow_point(canvas, p_l[0], p_l[1], 10.0, cfg, PALETTE.WHITE, p4_env)
            draw_glow_point(canvas, p_r[0], p_r[1], 10.0, cfg, PALETTE.WHITE, p4_env)

        draw_text(canvas, "Explicit Hand-Coded Rules", (vp_l.px_left + vp_l.px_right) * 0.5, cfg.px(280), 38, cfg, PALETTE.ACCENT_CORAL, "sans", bold=True, alpha=p4_env)
        draw_text(canvas, "Machine Learning Pattern", (vp_r.px_left + vp_r.px_right) * 0.5, cfg.px(280), 38, cfg, PALETTE.ACCENT_BLUE, "sans", bold=True, alpha=p4_env)


def render_scene_03_what_is_ml(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    04:30–08:00 — What Is Machine Learning?
    Single spacious full-screen coordinate plane (zero boxes):
    - Raw points appear -> staircase rule fails -> smooth curve emerges ->
      input x* projects up to curve ŷ = f(x*) and compares with true observation y.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 270.0, 480.0, 1.0, 1.0)
    vp = PlotViewport(0.0, 10.0, 0.0, 10.0, cfg.px(420), cfg.px(220), W - cfg.px(420), cfg.px(1640))
    draw_3b1b_axes(canvas, vp, cfg, alpha=env, x_ticks=[2, 4, 6, 8, 10], y_ticks=[2, 4, 6, 8, 10], x_label="Input x", y_label="Output y")

    true_fn = lambda x: 1.5 + 0.55 * x + 0.03 * (x ** 2)
    pts_x = [1.0, 1.9, 2.8, 3.7, 4.6, 5.6, 6.5, 7.4, 8.3, 9.2]
    pts_noise = [0.35, -0.45, 0.28, -0.32, 0.52, -0.40, 0.38, -0.48, 0.30, -0.25]

    stair_alpha = window_envelope(t, 270.0, 316.0, 1.0, 1.5)
    if stair_alpha > 0.01:
        stair_fn = lambda x: np.floor(x / 2.5) * 2.2 + 2.0
        draw_curve(canvas, vp, stair_fn, cfg, PALETTE.ACCENT_CORAL, 5.0, stair_alpha, progress=smoothstep(272.0, 286.0, t))

    for i, (px_val, dy) in enumerate(zip(pts_x, pts_noise)):
        pa = smoothstep(284.0 + i * 0.8, 286.0 + i * 0.8, t) * env
        if pa > 0.01:
            py_val = float(true_fn(np.array([px_val]))[0]) + dy
            sx, sy = vp.to_px(px_val, py_val)
            draw_glow_point(canvas, sx, sy, 12.0, cfg, PALETTE.WHITE, pa, glow_layers=2)

    curve_prog = smootherstep(348.0, 368.0, t)
    if curve_prog > 0.01:
        draw_curve(canvas, vp, true_fn, cfg, PALETTE.ACCENT_BLUE, 7.0, env, progress=curve_prog)
        if t >= 382.0:
            cx_lbl, cy_lbl = vp.to_px(8.2, float(true_fn(np.array([8.2]))[0]))
            draw_text(canvas, "Model: ŷ = f(x)", cx_lbl - cfg.px(40), cy_lbl - cfg.px(55), 38, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=env, anchor="midright")

    # Projection for Prediction ŷ vs Actual y at x = 6.5 (t >= 415s)
    proj_alpha = smoothstep(415.0, 422.0, t) * env
    if proj_alpha > 0.01:
        qx = pts_x[6]
        qy_true = float(true_fn(np.array([qx]))[0]) + pts_noise[6]
        qy_pred = float(true_fn(np.array([qx]))[0])
        s_axis = vp.to_px(qx, 0.0)
        s_pred = vp.to_px(qx, qy_pred)
        s_true = vp.to_px(qx, qy_true)

        draw_dashed_line(canvas, s_axis, s_pred, 3.0, cfg, PALETTE.ACCENT_BLUE, proj_alpha)
        draw_glow_point(canvas, s_pred[0], s_pred[1], 14.0, cfg, PALETTE.ACCENT_GOLD, proj_alpha, filled=False)
        draw_text(canvas, f"Prediction ŷ = {qy_pred:.2f}", s_pred[0] + cfg.px(28), s_pred[1] - cfg.px(18), 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=proj_alpha, anchor="midleft")

        if t >= 445.0:
            gap_a = smoothstep(445.0, 450.0, t) * env
            draw_aa_line(canvas, s_pred, s_true, 5.5, cfg, PALETTE.ACCENT_GOLD, gap_a)
            draw_text(canvas, f"Observed Data y = {qy_true:.2f}", s_true[0] - cfg.px(28), s_true[1] + cfg.px(32), 34, cfg, PALETTE.WHITE, "mono", bold=True, alpha=gap_a, anchor="midright")


def draw_minimal_house(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    scale_s: float,
    cfg: RenderConfig,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    if alpha <= 0.01:
        return
    w = cfg.px(64.0 * scale_s)
    h = cfg.px(46.0 * scale_s)
    roof = cfg.px(32.0 * scale_s)
    x0, x1 = cx - w * 0.5, cx + w * 0.5
    yb, ye, yp = cy + h * 0.5, cy - h * 0.5, cy - h * 0.5 - roof
    draw_aa_line(canvas, (x0, yb), (x1, yb), 3.0, cfg, color, alpha)
    draw_aa_line(canvas, (x0, yb), (x0, ye), 3.0, cfg, color, alpha)
    draw_aa_line(canvas, (x1, yb), (x1, ye), 3.0, cfg, color, alpha)
    draw_aa_line(canvas, (x0 - cfg.px(8), ye), (cx, yp), 3.2, cfg, color, alpha)
    draw_aa_line(canvas, (x1 + cfg.px(8), ye), (cx, yp), 3.2, cfg, color, alpha)


def render_scene_04_data_features_patterns(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    08:00–11:30 — Data, Features, and Patterns
    Single open coordinate plane (Area m² vs Price $k) with zero boxes:
    Minimal house wireframes appear at their coordinates and morph smoothly into
    glowing data points; orthogonal dashed lines highlight Feature x and Target y.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 480.0, 690.0, 1.0, 1.0)
    vp = PlotViewport(40.0, 230.0, 100.0, 440.0, cfg.px(420), cfg.px(220), W - cfg.px(380), cfg.px(1640))
    draw_3b1b_axes(
        canvas, vp, cfg, alpha=env,
        x_ticks=[60, 100, 140, 180, 220],
        y_ticks=[150, 200, 250, 300, 350, 400],
        x_label="Feature: Floor Area x (m²)",
        y_label="Target: Price y ($k)",
    )

    morph_to_dot = smootherstep(525.0, 542.0, t)
    highlight_idx = 4

    for i, (area, price) in enumerate(zip(HOUSE_DATA_X, HOUSE_DATA_Y)):
        ia = env * smoothstep(482.0 + i * 2.0, 485.5 + i * 2.0, t)
        if ia <= 0.01:
            continue
        px, py = vp.to_px(float(area), float(price))
        is_hi = (i == highlight_idx and 546.0 <= t <= 628.0)
        col = PALETTE.ACCENT_GOLD if is_hi else PALETTE.WHITE

        # Minimal house wireframe fades into glowing coordinate point
        if morph_to_dot < 0.98:
            draw_minimal_house(canvas, px, py - cfg.px(26), 0.55 + 0.45 * (area / 210.0), cfg, col, ia * (1.0 - morph_to_dot * 0.65))
        draw_glow_point(canvas, px, py, 15.0 if is_hi else 11.0, cfg, col, ia * clamp(0.35 + morph_to_dot, 0.0, 1.0), glow_layers=3 if is_hi else 2)

    # Feature / Target / Training Example orthogonal projections on point #5 (138 m², $278k)
    call_env = window_envelope(t, 546.0, 628.0, 1.0, 1.0) * env
    if call_env > 0.01:
        hx, hy = float(HOUSE_DATA_X[highlight_idx]), float(HOUSE_DATA_Y[highlight_idx])
        px, py = vp.to_px(hx, hy)
        _, y_bot = vp.to_px(hx, vp.y_min)
        x_left, _ = vp.to_px(vp.x_min, hy)

        draw_dashed_line(canvas, (px, py), (px, y_bot), 3.5, cfg, PALETTE.ACCENT_BLUE, call_env)
        draw_text(canvas, f"Feature x₅ = {hx:.0f} m²", px + cfg.px(20), (py + y_bot) * 0.5, 34, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=call_env, anchor="midleft")

        targ_a = smoothstep(563.0, 567.0, t) * call_env
        if targ_a > 0.01:
            draw_dashed_line(canvas, (px, py), (x_left, py), 3.5, cfg, PALETTE.ACCENT_GOLD, targ_a)
            draw_text(canvas, f"Target y₅ = ${hy:.0f}k", (px + x_left) * 0.5, py - cfg.px(32), 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=targ_a)

        ex_a = smoothstep(580.0, 584.0, t) * call_env
        if ex_a > 0.01:
            draw_text(canvas, f"Training Example (x₅, y₅) = ({hx:.0f}, {hy:.0f})", px + cfg.px(28), py - cfg.px(36), 36, cfg, PALETTE.WHITE, "mono", bold=True, alpha=ex_a, anchor="midleft")

    # Subtle trend line & noise band (t >= 630s)
    noise_a = smoothstep(630.0, 640.0, t) * env
    if noise_a > 0.01:
        draw_curve(canvas, vp, lambda x: 1.65 * x + 60.0, cfg, PALETTE.GRAY_DARK, 2.5, noise_a * 0.65)
        draw_curve(canvas, vp, lambda x: 1.65 * x + 16.0, cfg, PALETTE.GRAY_DARK, 2.5, noise_a * 0.65)
        draw_curve(canvas, vp, lambda x: 1.65 * x + 38.0, cfg, PALETTE.ACCENT_BLUE, 5.5, noise_a * 0.9)


def render_scene_05_linear_model(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    11:30–15:00 — The Mathematical Heart: y = wx + b
    Single spacious coordinate plane with floating equation y = wx + b in open black space
    (zero boxes!), live geometric slope triangle Δy/Δx = w, and y-intercept dot (0, b).
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 690.0, 900.0, 1.0, 1.0)

    if t < 783.0:
        w_val, b_val = 1.45, 45.0
    elif t < 800.0:
        w_val = lerp(0.0, 1.25, ease_in_out_sine((t - 783.0) / 17.0))
        b_val = 0.0
    elif t < 817.0:
        w_val = lerp(1.25, 2.35, ease_in_out_sine((t - 800.0) / 17.0))
        b_val = 0.0
    elif t < 833.0:
        u = (t - 817.0) / 16.0
        w_val = lerp(2.35, -0.55, ease_in_out_sine(u * 2.0)) if u < 0.5 else lerp(-0.55, 1.65, ease_in_out_sine((u - 0.5) * 2.0))
        b_val = 190.0 if w_val < 0 else lerp(0.0, 25.0, u)
    elif t < 866.0:
        w_val = 1.65
        u = (t - 833.0) / 33.0
        if u < 0.45:
            b_val = lerp(25.0, 105.0, ease_in_out_sine(u / 0.45))
        elif u < 0.80:
            b_val = lerp(105.0, -15.0, ease_in_out_sine((u - 0.45) / 0.35))
        else:
            b_val = lerp(-15.0, 38.0, ease_in_out_sine((u - 0.80) / 0.20))
    else:
        w_val, b_val = 1.65, 38.0

    vp = PlotViewport(0.0, 230.0, -40.0, 460.0, cfg.px(440), cfg.px(240), W - cfg.px(360), cfg.px(1640))
    draw_3b1b_axes(
        canvas, vp, cfg, alpha=env,
        x_ticks=[0, 50, 100, 150, 200],
        y_ticks=[0, 100, 200, 300, 400],
        x_label="Input x",
        y_label="Output y",
    )

    # Floating equation in open top-left space (no box!)
    draw_text(canvas, "y = w · x + b", cfg.px(560), cfg.px(160), 64, cfg, PALETTE.WHITE, "mono", bold=True, alpha=env, anchor="midleft")
    draw_text(
        canvas,
        f"w = {w_val:+5.2f} (slope)     b = {b_val:+5.1f} (intercept)",
        cfg.px(1220), cfg.px(160), 40, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=env, anchor="midleft",
    )

    line_fn = lambda x: w_val * x + b_val
    draw_curve(canvas, vp, line_fn, cfg, PALETTE.ACCENT_BLUE, 7.0, env, progress=smoothstep(692.0, 704.0, t))

    # Intercept dot (0, b)
    ix_b, iy_b = vp.to_px(0.0, clamp(b_val, vp.y_min, vp.y_max))
    draw_glow_point(canvas, ix_b, iy_b, 15.0, cfg, PALETTE.ACCENT_GOLD, env, glow_layers=3)
    draw_text(canvas, f"b = {b_val:+.0f}", ix_b + cfg.px(28), iy_b, 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env, anchor="midleft")

    # Slope triangle between x=80 and x=140
    x1_t, x2_t = 80.0, 140.0
    y1_t, y2_t = w_val * x1_t + b_val, w_val * x2_t + b_val
    if vp.y_min <= y1_t <= vp.y_max and vp.y_min <= y2_t <= vp.y_max:
        pa = vp.to_px(x1_t, y1_t)
        pb = vp.to_px(x2_t, y1_t)
        pc = vp.to_px(x2_t, y2_t)
        draw_dashed_line(canvas, pa, pb, 3.0, cfg, PALETTE.GRAY_LIGHT, env * 0.9)
        draw_dashed_line(canvas, pb, pc, 3.5, cfg, PALETTE.ACCENT_GOLD, env * 0.9)
        draw_text(canvas, "Δx", (pa[0] + pb[0]) * 0.5, pb[1] + cfg.px(34), 32, cfg, PALETTE.GRAY_LIGHT, "mono", alpha=env)
        draw_text(canvas, f"Δy = w·Δx ({y2_t - y1_t:+.0f})", pc[0] + cfg.px(24), (pb[1] + pc[1]) * 0.5, 32, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env, anchor="midleft")

    # Overlay the 8 house points for t >= 864s
    pts_a = smoothstep(864.0, 870.0, t) * env
    if pts_a > 0.01:
        for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
            px, py = vp.to_px(float(hx), float(hy))
            draw_glow_point(canvas, px, py, 12.0, cfg, PALETTE.WHITE, pts_a, glow_layers=2)


def render_scene_06_neural_network(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    15:00–18:30 — From a Simple Model to a Neural Network
    Pure 3Blue1Brown neural network floating in open black space (zero boxes!),
    plus an open mini coordinate graph on the right showing the non-linear activation curve σ(z).
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 900.0, 1110.0, 1.0, 1.0)

    x_in = cfg.px(540)
    x_hid = cfg.px(1480)
    x_out = cfg.px(2420)

    in_extra_a = smoothstep(948.0, 960.0, t) * env
    hid_a = smoothstep(965.0, 980.0, t) * env
    out_a = smoothstep(982.0, 995.0, t) * env

    in_ys = [cfg.px(520), cfg.px(900), cfg.px(1280)]
    hid_ys = [cfg.px(380), cfg.px(720), cfg.px(1060), cfg.px(1400)]
    out_y = cfg.px(900)

    single_a = env * (1.0 - smoothstep(965.0, 975.0, t))
    if single_a > 0.01:
        draw_aa_line(canvas, (x_in, in_ys[1]), (x_out, out_y), 5.0, cfg, PALETTE.ACCENT_BLUE, single_a)
        draw_text(canvas, "y = w · x + b", (x_in + x_out) * 0.5, in_ys[1] - cfg.px(55), 42, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=single_a)

    sig_phase = ((t - 900.0) * 0.6) % 2.0
    if hid_a > 0.01:
        for i, iy in enumerate(in_ys):
            na = env if i == 1 else in_extra_a
            if na <= 0.01:
                continue
            for j, hy in enumerate(hid_ys):
                w_mag = 0.4 + 0.6 * abs(math.sin((i + 1) * 1.7 + (j + 1) * 0.9))
                ecol = lerp_color(PALETTE.GRAY_DARK, PALETTE.ACCENT_BLUE, w_mag)
                draw_aa_line(canvas, (x_in, iy), (x_hid, hy), 2.0 + 3.2 * w_mag, cfg, ecol, hid_a * na * 0.85)
                if 0.0 <= sig_phase <= 1.0 and t >= 1000.0:
                    draw_glow_point(canvas, lerp(x_in, x_hid, sig_phase), lerp(iy, hy, sig_phase), 8.0, cfg, PALETTE.WHITE, hid_a * na, glow_layers=2)

    if out_a > 0.01:
        for j, hy in enumerate(hid_ys):
            w_mag = 0.5 + 0.5 * abs(math.cos((j + 1) * 1.3))
            ecol = lerp_color(PALETTE.GRAY_MID, PALETTE.WHITE, w_mag)
            draw_aa_line(canvas, (x_hid, hy), (x_out, out_y), 2.5 + 3.5 * w_mag, cfg, ecol, out_a * 0.9)
            if 1.0 < sig_phase <= 2.0 and t >= 1000.0:
                u2 = sig_phase - 1.0
                draw_glow_point(canvas, lerp(x_hid, x_out, u2), lerp(hy, out_y, u2), 9.0, cfg, PALETTE.ACCENT_GOLD, out_a, glow_layers=2)

    for i, iy in enumerate(in_ys):
        na = env if i == 1 else in_extra_a
        if na > 0.01:
            draw_glow_point(canvas, x_in, iy, 42.0, cfg, PALETTE.WHITE, na, glow_layers=3, filled=False)
            draw_text(canvas, f"x{i+1}", x_in, iy, 34, cfg, PALETTE.WHITE, "mono", bold=True, alpha=na)

    if hid_a > 0.01:
        for j, hy in enumerate(hid_ys):
            draw_glow_point(canvas, x_hid, hy, 44.0, cfg, PALETTE.ACCENT_BLUE, hid_a, glow_layers=3, filled=False)
            draw_text(canvas, f"h{j+1}", x_hid, hy, 34, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=hid_a)

    draw_glow_point(canvas, x_out, out_y, 48.0, cfg, PALETTE.ACCENT_GOLD, env, glow_layers=4, filled=False)
    draw_text(canvas, "ŷ", x_out, out_y, 40, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env)

    draw_text(canvas, "Input Layer", x_in, cfg.px(230), 36, cfg, PALETTE.GRAY_LIGHT, "sans", bold=True, alpha=env)
    if hid_a > 0.01:
        draw_text(canvas, "Hidden Layer", x_hid, cfg.px(230), 36, cfg, PALETTE.ACCENT_BLUE, "sans", bold=True, alpha=hid_a)
    draw_text(canvas, "Output Layer", x_out, cfg.px(230), 36, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=env)

    # Open non-linear activation curve graph on the right (t >= 1030s, zero boxes!)
    act_a = smoothstep(1030.0, 1038.0, t) * env
    if act_a > 0.01:
        vp_act = PlotViewport(-4.0, 4.0, -0.2, 1.2, cfg.px(2820), cfg.px(560), W - cfg.px(220), cfg.px(1240))
        draw_3b1b_axes(canvas, vp_act, cfg, alpha=act_a, x_ticks=[-2, 2], y_ticks=[0.5, 1.0], x_label="z", y_label="σ(z)")
        sig_fn = lambda z: 1.0 / (1.0 + np.exp(-z))
        draw_curve(canvas, vp_act, sig_fn, cfg, PALETTE.ACCENT_GOLD, 6.0, act_a)
        draw_text(canvas, "z = Σ wᵢxᵢ + b", (vp_act.px_left + vp_act.px_right) * 0.5, cfg.px(460), 36, cfg, PALETTE.WHITE, "mono", bold=True, alpha=act_a)


def render_scene_07_how_model_learns(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    18:30–22:00 — How Does a Model Learn?
    Left: Open 2D coordinate plane with house data, line y = wx + b, and gold vertical error residuals.
    Right: True 3D Loss Surface wireframe bowl L(w, b) rotating gently in pure black space,
           with a glowing gold ball rolling down the 3D surface to the minimum! Zero boxes!
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1110.0, 1320.0, 1.0, 1.0)

    keyframes = [
        (1110.0, 0.35, 85.0),
        (1210.0, 0.35, 85.0),
        (1235.0, 0.88, 66.0),
        (1258.0, 1.38, 48.0),
        (1276.0, 2.04, 30.0),  # Slight overshoot across the valley!
        (1292.0, 1.52, 42.0),
        (1310.0, 1.65, 38.0),
        (1320.0, 1.65, 38.0),
    ]
    w_curr, b_curr = keyframes[-1][1], keyframes[-1][2]
    for (t0, w0, b0), (t1, w1, b1) in zip(keyframes[:-1], keyframes[1:]):
        if t0 <= t <= t1:
            u = ease_in_out_cubic((t - t0) / max(1e-5, t1 - t0))
            w_curr = lerp(w0, w1, u)
            b_curr = lerp(b0, b1, u)
            break

    loss_curr = compute_house_mse(w_curr, b_curr)

    # Floating formula readout at top
    draw_text(
        canvas,
        f"w = {w_curr:5.2f}     b = {b_curr:5.1f}     Loss L(w, b) = {loss_curr:7.1f}",
        W * 0.5, cfg.px(140), 44, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env,
    )

    # Left: Open 2D Data Space
    vp_data = PlotViewport(40.0, 230.0, 80.0, 450.0, cfg.px(240), cfg.px(300), cfg.px(1820), cfg.px(1600))
    draw_3b1b_axes(
        canvas, vp_data, cfg, alpha=env,
        x_ticks=[60, 100, 140, 180, 220],
        y_ticks=[100, 200, 300, 400],
        x_label="Area x",
        y_label="Price y",
    )
    draw_curve(canvas, vp_data, lambda x: w_curr * x + b_curr, cfg, PALETTE.ACCENT_BLUE, 6.5, env)

    err_a = smoothstep(1126.0, 1134.0, t) * env
    for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
        y_pred = w_curr * float(hx) + b_curr
        p_true = vp_data.to_px(float(hx), float(hy))
        p_pred = vp_data.to_px(float(hx), clamp(y_pred, vp_data.y_min, vp_data.y_max))
        if err_a > 0.01:
            draw_aa_line(canvas, p_true, p_pred, 4.5, cfg, PALETTE.ACCENT_GOLD, err_a * 0.9)
        draw_glow_point(canvas, p_true[0], p_true[1], 11.0, cfg, PALETTE.WHITE, env, glow_layers=2)

    # Right: True 3D Loss Surface Bowl L(w, b) floating in open space (t >= 1170s)
    surf_a = smoothstep(1170.0, 1180.0, t) * env
    if surf_a > 0.01:
        yaw_angle = 0.62 + 0.22 * math.sin((t - 1170.0) * 0.12)
        cam = Camera3D(cfg.px(2820), cfg.px(1020), cfg.px(250), yaw=yaw_angle, pitch=0.50)
        draw_3d_axes(canvas, cam, cfg, alpha=surf_a * 0.85, axis_len=2.2, labels=("w", "b", "Loss L"))

        # Normalized 3D paraboloid bowl centered at optimal (w=1.65, b=38) -> (u=0, v=0, z=-0.6)
        bowl_fn = lambda u, v: -0.6 + 0.52 * (u * u + 0.85 * v * v)
        draw_3d_surface_wireframe(
            canvas, cam, bowl_fn, cfg,
            u_range=(-1.6, 1.6), v_range=(-1.6, 1.6),
            grid_n=15, alpha=surf_a,
            color_low=PALETTE.ACCENT_BLUE, color_high=PALETTE.WHITE,
        )

        # Map current (w_curr, b_curr) onto the 3D bowl coordinates
        u_w = clamp((w_curr - 1.65) * 1.15, -1.6, 1.6)
        v_b = clamp((b_curr - 38.0) / 36.0, -1.6, 1.6)
        z_l = bowl_fn(u_w, v_b)
        bx, by, _ = cam.project(u_w, v_b, z_l)
        draw_glow_point(canvas, bx, by, 18.0, cfg, PALETTE.ACCENT_GOLD, surf_a, glow_layers=4)


def render_scene_08_three_approaches(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    22:00–25:00 — Three Important Learning Approaches
    Sequential, full-screen unboxed demonstrations (zero boxes!):
    - 1320..1380s: Supervised Learning on an open 2D coordinate plane.
    - 1380..1440s: Unsupervised Learning in a rotating 3D point cloud separating into 3 clusters.
    - 1440..1500s: Reinforcement Learning on an open 2D vector field with agent trajectory.
    """
    W, H = cfg.width, cfg.height

    # 1. Supervised Learning (1320..1380s)
    s1_env = window_envelope(t, 1320.0, 1380.5, 1.0, 1.2)
    if s1_env > 0.01:
        draw_text(canvas, "SUPERVISED LEARNING", W * 0.5, cfg.px(150), 50, cfg, PALETTE.ACCENT_BLUE, "sans", bold=True, alpha=s1_env)
        vp = PlotViewport(40.0, 230.0, 100.0, 440.0, cfg.px(440), cfg.px(260), W - cfg.px(440), cfg.px(1620))
        draw_3b1b_axes(canvas, vp, cfg, alpha=s1_env, x_ticks=[60, 100, 140, 180, 220], y_ticks=[150, 250, 350], x_label="Feature x", y_label="Known Target y")
        draw_curve(canvas, vp, lambda x: 1.65 * x + 38.0, cfg, PALETTE.ACCENT_BLUE, 6.5, s1_env, progress=smoothstep(1335.0, 1355.0, t))
        for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
            px, py = vp.to_px(float(hx), float(hy))
            draw_glow_point(canvas, px, py, 12.0, cfg, PALETTE.WHITE, s1_env, glow_layers=2)
            draw_text(canvas, f"({hx:.0f}, ${hy:.0f}k)", px + cfg.px(20), py - cfg.px(22), 28, cfg, PALETTE.ACCENT_GOLD, "mono", alpha=s1_env, anchor="midleft")

    # 2. Unsupervised Learning in 3D Space (1380..1440s)
    s2_env = window_envelope(t, 1380.0, 1440.5, 1.2, 1.2)
    if s2_env > 0.01:
        draw_text(canvas, "UNSUPERVISED LEARNING", W * 0.5, cfg.px(150), 50, cfg, PALETTE.WHITE, "sans", bold=True, alpha=s2_env)
        cam = Camera3D(W * 0.5, H * 0.50, cfg.px(290), yaw=0.4 + (t - 1380.0) * 0.18, pitch=0.45)
        draw_3d_axes(canvas, cam, cfg, alpha=s2_env * 0.8, axis_len=2.3, labels=("x₁", "x₂", "x₃"))

        u_sep = smootherstep(1388.0, 1416.0, t)
        centers_3d = [(-1.25, -0.9, 0.6), (1.3, -0.7, 0.4), (0.1, 1.35, -0.5)]
        cluster_cols = [PALETTE.ACCENT_BLUE, PALETTE.ACCENT_GOLD, PALETTE.WHITE]
        offsets_3d = [(-0.25, 0.18, 0.15), (0.22, -0.20, 0.12), (-0.18, -0.22, -0.18), (0.20, 0.24, -0.14), (0.0, 0.08, 0.25), (-0.12, 0.20, -0.10)]

        for c_i, (cc, col) in enumerate(zip(centers_3d, cluster_cols)):
            for ox, oy, oz in offsets_3d:
                sx0, sy0, sz0 = ox * 3.2, oy * 3.2, oz * 2.5
                cx3 = lerp(sx0, cc[0] + ox, u_sep)
                cy3 = lerp(sy0, cc[1] + oy, u_sep)
                cz3 = lerp(sz0, cc[2] + oz, u_sep)
                px, py, _ = cam.project(cx3, cy3, cz3)
                draw_glow_point(canvas, px, py, 11.0, cfg, lerp_color(PALETTE.WHITE, col, u_sep), s2_env, glow_layers=2)

    # 3. Reinforcement Learning Vector Field (1440..1500s)
    s3_env = window_envelope(t, 1440.0, 1500.0, 1.2, 1.0)
    if s3_env > 0.01:
        draw_text(canvas, "REINFORCEMENT LEARNING", W * 0.5, cfg.px(150), 50, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=s3_env)
        vp = PlotViewport(0.0, 10.0, 0.0, 6.0, cfg.px(420), cfg.px(260), W - cfg.px(420), cfg.px(1620))
        draw_3b1b_axes(canvas, vp, cfg, alpha=s3_env, x_ticks=[2, 4, 6, 8, 10], y_ticks=[1, 2, 3, 4, 5], x_label="State s₁", y_label="State s₂")

        # Vector field arrows guiding toward Goal at (8.8, 4.8)
        for gx_v in np.linspace(1.2, 8.2, 8):
            for gy_v in np.linspace(1.0, 5.0, 5):
                dx, dy = 8.8 - float(gx_v), 4.8 - float(gy_v)
                norm = max(1e-4, math.hypot(dx, dy))
                p_start = vp.to_px(float(gx_v), float(gy_v))
                p_end = vp.to_px(float(gx_v) + 0.42 * (dx / norm), float(gy_v) + 0.42 * (dy / norm))
                draw_arrow(canvas, p_start, p_end, 2.2, cfg, PALETTE.GRAY_DARK, s3_env * 0.75, 0.25)

        rl_curve = lambda x: 1.0 + 3.8 * ((x - 1.0) / 7.8) ** 0.75 + 0.45 * np.sin(math.pi * (x - 1.0) / 7.8)
        draw_curve(canvas, vp, rl_curve, cfg, PALETTE.ACCENT_BLUE, 6.0, s3_env)

        gx, gy = vp.to_px(8.8, 4.8)
        draw_glow_point(canvas, gx, gy, 18.0, cfg, PALETTE.ACCENT_GOLD, s3_env, glow_layers=4)
        draw_text(canvas, "Reward +10", gx - cfg.px(28), gy - cfg.px(36), 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=s3_env, anchor="midright")

        u_ag = ((t - 1440.0) * 0.22) % 1.0
        ax_v = lerp(1.0, 8.8, u_ag)
        ay_v = float(rl_curve(np.array([ax_v]))[0])
        apx, apy = vp.to_px(ax_v, ay_v)
        draw_glow_point(canvas, apx, apy, 16.0, cfg, PALETTE.WHITE, s3_env, glow_layers=4)
        draw_text(canvas, "Agent", apx + cfg.px(24), apy + cfg.px(28), 32, cfg, PALETTE.WHITE, "mono", bold=True, alpha=s3_env, anchor="midleft")


def render_scene_09_generalization(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    25:00–27:30 — Training vs. Generalization
    Single spacious full-screen coordinate plane (zero boxes!):
    - Solid white Training Data points.
    - Degree-7 oscillating coral polynomial (Overfitting) vs smooth blue trend line (Generalization).
    - Hollow gold Unseen Data points with vertical residual dashed lines.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1500.0, 1650.0, 1.0, 1.0)
    vp = PlotViewport(45.0, 220.0, 90.0, 440.0, cfg.px(420), cfg.px(230), W - cfg.px(380), cfg.px(1640))
    draw_3b1b_axes(
        canvas, vp, cfg, alpha=env,
        x_ticks=[60, 100, 140, 180, 210],
        y_ticks=[100, 200, 300, 400],
        x_label="House Area x (m²)",
        y_label="Sale Price y ($k)",
    )

    poly_coeffs = np.polyfit(HOUSE_DATA_X, HOUSE_DATA_Y, deg=7)
    overfit_fn = lambda x: np.polyval(poly_coeffs, x)
    general_fn = lambda x: 1.65 * x + 38.0

    over_prog = smootherstep(1522.0, 1544.0, t)
    if over_prog > 0.01:
        draw_curve(canvas, vp, overfit_fn, cfg, PALETTE.ACCENT_CORAL, 5.0, env, progress=over_prog, num_samples=320)
        draw_text(canvas, "Overfitting (Memorizes Noise)", cfg.px(520), cfg.px(160), 38, cfg, PALETTE.ACCENT_CORAL, "mono", bold=True, alpha=env * over_prog, anchor="midleft")

    gen_prog = smootherstep(1549.0, 1565.0, t)
    if gen_prog > 0.01:
        draw_curve(canvas, vp, general_fn, cfg, PALETTE.ACCENT_BLUE, 7.0, env, progress=gen_prog)
        draw_text(canvas, "Generalization (Captures Trend)", cfg.px(1880), cfg.px(160), 38, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=env * gen_prog, anchor="midleft")

    for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
        px, py = vp.to_px(float(hx), float(hy))
        draw_glow_point(canvas, px, py, 11.5, cfg, PALETTE.WHITE, env, glow_layers=2)

    unseen_a = smoothstep(1584.0, 1592.0, t) * env
    if unseen_a > 0.01:
        for ux, uy in zip(UNSEEN_DATA_X, UNSEEN_DATA_Y):
            px, py = vp.to_px(float(ux), float(uy))
            y_over = clamp(float(overfit_fn(np.array([ux]))[0]), vp.y_min, vp.y_max)
            p_over = vp.to_px(float(ux), y_over)
            draw_dashed_line(canvas, (px, py), p_over, 3.5, cfg, PALETTE.ACCENT_CORAL, unseen_a * 0.85)
            draw_glow_point(canvas, px, py, 16.0, cfg, PALETTE.ACCENT_GOLD, unseen_a, glow_layers=3, filled=False)


def render_scene_10_roadmap(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    27:30–29:00 — What Will We Learn Next?
    Open geometric constellation pathway in pure black space (zero boxes!):
    Mathematics -> Python -> Machine Learning -> Neural Networks -> Deep Learning,
    followed by a luminous PyTorch future node below.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1650.0, 1740.0, 1.0, 1.0)

    stages = [
        ("Mathematics", 1652.0),
        ("Python", 1662.0),
        ("Machine Learning", 1672.0),
        ("Neural Networks", 1682.0),
        ("Deep Learning", 1692.0),
    ]
    xs = [cfg.px(460 + i * 730) for i in range(5)]
    ys = [cfg.px(760 - 90 * math.sin(i * math.pi / 4.0)) for i in range(5)]

    for i in range(4):
        ea = env * smoothstep(stages[i + 1][1] - 2.0, stages[i + 1][1] + 2.0, t)
        if ea > 0.01:
            draw_arrow(canvas, (xs[i] + cfg.px(55), ys[i]), (xs[i + 1] - cfg.px(55), ys[i + 1]), 4.5, cfg, PALETTE.ACCENT_BLUE, ea, 0.04)

    for i, (st_name, t_un) in enumerate(stages):
        na = env * smoothstep(t_un - 1.5, t_un + 2.0, t)
        if na > 0.01:
            draw_glow_point(canvas, xs[i], ys[i], 38.0, cfg, PALETTE.ACCENT_BLUE, na, glow_layers=4, filled=False)
            draw_glow_point(canvas, xs[i], ys[i], 10.0, cfg, PALETTE.WHITE, na, glow_layers=2)
            draw_text(canvas, st_name, xs[i], ys[i] + cfg.px(88), 38, cfg, PALETTE.WHITE, "sans", bold=True, alpha=na)

    pt_a = env * smoothstep(1710.0, 1716.0, t)
    if pt_a > 0.01:
        py_y = cfg.px(1260)
        draw_glow_point(canvas, W * 0.5, py_y, 22.0, cfg, PALETTE.ACCENT_GOLD, pt_a, glow_layers=4)
        draw_text(canvas, "PyTorch — Coming in a Future Lecture", W * 0.5, py_y + cfg.px(82), 52, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=pt_a)


def render_scene_11_recap_closing(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    29:00–30:00 — Recap and Closing
    Returns to the moving white point and 3D/2D parametric curve in pure open black space
    (zero boxes!), floating the closing question and final title before fading to black at 30:00.
    """
    W, H = cfg.width, cfg.height
    master_fade = 1.0 - smoothstep(1796.2, 1799.6, t)
    if master_fade <= 0.005:
        return

    vp = PlotViewport(0.0, 10.0, 0.0, 10.0, cfg.px(420), cfg.px(220), W - cfg.px(420), cfg.px(1640))
    bg_a = master_fade * (0.75 if t < 1782.0 else 0.25)
    draw_3b1b_axes(canvas, vp, cfg, alpha=bg_a, x_ticks=[2, 4, 6, 8, 10], y_ticks=[2, 4, 6, 8, 10], x_label="x", y_label="y")

    curve_fn = lambda x: 1.8 + 0.62 * x + 1.15 * np.sin(0.58 * x)
    draw_curve(canvas, vp, curve_fn, cfg, PALETTE.ACCENT_BLUE, 6.0, bg_a)

    xt = 5.0 + 3.8 * math.sin(0.45 * (t - 1740.0))
    yt = float(curve_fn(np.array([xt]))[0])
    px, py = vp.to_px(xt, yt)
    draw_glow_point(canvas, px, py, 16.0, cfg, PALETTE.WHITE, master_fade * 0.9, glow_layers=4)

    q_env = window_envelope(t, 1768.3, 1782.5, 1.0, 1.0) * master_fade
    if q_env > 0.01:
        draw_text(canvas, "If a machine can learn a pattern from examples,", W * 0.5, cfg.px(420), 54, cfg, PALETTE.WHITE, "sans", bold=True, alpha=q_env)
        draw_text(canvas, "what else can we teach it to discover?", W * 0.5, cfg.px(520), 54, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=q_env)

    end_env = smoothstep(1782.8, 1785.0, t) * master_fade
    if end_env > 0.01:
        draw_text(canvas, "SILENT MATH", W * 0.5, cfg.px(700), 96, cfg, PALETTE.WHITE, "sans", bold=True, alpha=end_env)
        draw_text(canvas, "LECTURE #1 COMPLETE", W * 0.5, cfg.px(855), 50, cfg, PALETTE.ACCENT_BLUE, "mono", bold=True, alpha=end_env)
        draw_text(canvas, "Next: Mathematics Behind Machine Learning", W * 0.5, cfg.px(995), 42, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=end_env)


SCENE_RENDER_DISPATCH: Dict[int, Callable[[np.ndarray, float, RenderConfig], None]] = {
    1: render_scene_01_intro,
    2: render_scene_02_what_is_ai,
    3: render_scene_03_what_is_ml,
    4: render_scene_04_data_features_patterns,
    5: render_scene_05_linear_model,
    6: render_scene_06_neural_network,
    7: render_scene_07_how_model_learns,
    8: render_scene_08_three_approaches,
    9: render_scene_09_generalization,
    10: render_scene_10_roadmap,
    11: render_scene_11_recap_closing,
}


def render_frame_at_time(t: float, cfg: RenderConfig, canvas: Optional[np.ndarray] = None) -> np.ndarray:
    if canvas is None:
        canvas = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    else:
        canvas.fill(0)

    scene = get_scene_for_time(t)
    renderer = SCENE_RENDER_DISPATCH.get(scene.index)
    if renderer is not None:
        renderer(canvas, t, cfg)
    draw_floating_captions(canvas, scene, t, cfg)
    return canvas


# ==============================================================================
# 7. AUTOMATED QUALITY CONTROL & FFPROBE VALIDATOR
# ==============================================================================

def run_project_validation(cfg: RenderConfig) -> Dict[str, object]:
    issues: List[str] = []
    if abs(LECTURE_SCENES[0].start_time - 0.0) > 1e-6:
        issues.append("Scene 1 does not start at 0.0s")
    for s_prev, s_next in zip(LECTURE_SCENES[:-1], LECTURE_SCENES[1:]):
        if abs(s_prev.end_time - s_next.start_time) > 1e-6:
            issues.append(f"Discontinuity between {s_prev.slug} and {s_next.slug}")
    total_dur = sum(s.duration for s in LECTURE_SCENES)
    if abs(total_dur - 1800.0) > 1e-6:
        issues.append(f"Total duration is {total_dur}, expected 1800.0")

    total_captions = sum(len(s.captions) for s in LECTURE_SCENES)
    sample_canvas = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    for scene in LECTURE_SCENES:
        mid_t = 0.5 * (scene.start_time + scene.end_time)
        render_frame_at_time(mid_t, cfg, sample_canvas)
        if int(sample_canvas.max()) == 0:
            issues.append(f"Blank frame in {scene.slug}")

    render_frame_at_time(1799.9, cfg, sample_canvas)
    if float(sample_canvas.mean()) > 1.5:
        issues.append("Final frame at 1799.9s did not fade to black")

    return {
        "status": "PASS" if not issues else "FAIL",
        "style": "Pure 3Blue1Brown 2D/3D Open Space (Zero Boxes/Panels)",
        "total_scenes": len(LECTURE_SCENES),
        "total_duration_seconds": total_dur,
        "total_frames_at_30fps": int(round(total_dur * 30)),
        "total_animated_captions": total_captions,
        "issues": issues,
    }


def verify_mp4_with_ffprobe(mp4_path: str) -> Dict[str, object]:
    if not os.path.isfile(mp4_path):
        return {"status": "FILE_NOT_FOUND", "path": mp4_path}
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", mp4_path]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    streams = data.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    v0 = v_streams[0] if v_streams else {}
    fmt = data.get("format", {})
    return {
        "status": "VERIFIED",
        "file": mp4_path,
        "codec_name": v0.get("codec_name"),
        "width": v0.get("width"),
        "height": v0.get("height"),
        "r_frame_rate": v0.get("r_frame_rate"),
        "pix_fmt": v0.get("pix_fmt"),
        "duration_seconds": float(fmt.get("duration", 0.0)),
        "audio_streams_count": len(a_streams),
        "has_audio": len(a_streams) > 0,
    }


# ==============================================================================
# 8. FORK-SAFE SUBPROCESS RENDER ENGINE WITH 2-SECOND LIVE HEARTBEAT
# ==============================================================================

def _render_single_segment_worker(
    seg_idx: int,
    total_segs: int,
    t_start: float,
    t_end: float,
    out_file: str,
    target_w: int,
    target_h: int,
    fps: int,
    motion_fps: int,
    internal_scale: float,
    preset: str,
    crf: int,
) -> None:
    status_file = out_file + ".status.json"

    def _write_status(state: str, frame_done: int, frame_total: int, speed_fps: float, elapsed: float) -> None:
        try:
            with open(status_file, "w", encoding="utf-8") as sf:
                json.dump(
                    {
                        "seg_idx": seg_idx,
                        "state": state,
                        "frame_done": frame_done,
                        "frame_total": frame_total,
                        "speed_fps": round(speed_fps, 1),
                        "elapsed": round(elapsed, 1),
                    },
                    sf,
                )
        except Exception:
            pass

    t_seg0 = time.time()
    rw = max(16, int(round(target_w * internal_scale)) // 2 * 2)
    rh = max(16, int(round(target_h * internal_scale)) // 2 * 2)

    eff_motion_fps = max(1, min(fps, int(motion_fps)))
    cfg = RenderConfig(width=rw, height=rh, fps=fps)
    n_in_frames = max(1, int(round((t_end - t_start) * eff_motion_fps)))
    total_out_frames = max(1, int(round((t_end - t_start) * fps)))

    _write_status("starting", 0, total_out_frames, 0.0, 0.0)

    vf_chain: List[str] = ["format=yuv420p"]
    if rw != target_w or rh != target_h:
        vf_chain.append(f"scale={target_w}:{target_h}:flags=neighbor")
    if eff_motion_fps != fps:
        vf_chain.append(f"fps={fps}")
    vf_args = ["-vf", ",".join(vf_chain)]

    tmp_out = out_file + ".partial.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{rw}x{rh}", "-r", str(eff_motion_fps),
        "-i", "-", "-an",
        *vf_args,
        "-c:v", "libx264", "-threads", "1",
        "-preset", preset, "-tune", "fastdecode",
        "-x264-params", "bframes=0:ref=1:rc-lookahead=0:sync-lookahead=0",
        "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", tmp_out,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    canvas = np.zeros((rh, rw, 3), dtype=np.uint8)
    inv_mfps = 1.0 / float(eff_motion_fps)

    try:
        for f_idx in range(n_in_frames):
            t_cur = t_start + f_idx * inv_mfps
            render_frame_at_time(t_cur, cfg, canvas)
            assert proc.stdin is not None
            proc.stdin.write(canvas.tobytes())

            out_done = min(total_out_frames, int(round((f_idx + 1) * (fps / eff_motion_fps))))
            el = max(0.001, time.time() - t_seg0)
            _write_status("rendering", out_done, total_out_frames, out_done / el, el)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {proc.returncode} on {out_file}")

    shutil.move(tmp_out, out_file)
    el = time.time() - t_seg0
    _write_status("done", total_out_frames, total_out_frames, total_out_frames / max(0.001, el), el)


def render_lecture_video(
    mode: str,
    res_key: str,
    output_path: str,
    scene_index: Optional[int] = None,
    workers: int = 2,
    internal_scale: Optional[float] = None,
    motion_fps: int = 6,
    preset: str = "ultrafast",
    crf: int = 20,
    parts_dir: Optional[str] = None,
    keep_parts: bool = False,
) -> Dict[str, object]:
    target_w, target_h = RESOLUTION_PRESETS.get(res_key.lower(), (3840, 2160))
    fps = 30
    out_abs = os.path.abspath(output_path)
    out_dir = os.path.dirname(out_abs) or "."
    os.makedirs(out_dir, exist_ok=True)

    if internal_scale is None:
        internal_scale = (1.0 / 3.0) if target_w >= 3840 else 1.0

    t0 = time.time()
    segments: List[Tuple[float, float]] = []

    if mode == "preview":
        for sc in LECTURE_SCENES:
            mid = 0.5 * (sc.start_time + sc.end_time)
            segments.append((mid, min(sc.end_time, mid + 2.0)))
    elif mode == "scene":
        idx = scene_index or 1
        matching = [s for s in LECTURE_SCENES if s.index == idx]
        if not matching:
            raise ValueError(f"Scene {idx} not found (valid: 1..11)")
        sc = matching[0]
        cur = sc.start_time
        while cur < sc.end_time - 1e-6:
            nxt = min(sc.end_time, cur + 15.0)
            segments.append((cur, nxt))
            cur = nxt
    elif mode == "full":
        chunk_sec = 15.0
        for c in range(int(round(1800.0 / chunk_sec))):
            segments.append((c * chunk_sec, (c + 1) * chunk_sec))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    work_parts_dir = os.path.abspath(parts_dir or os.path.join(out_dir, f"silent_math_parts_{mode}_{res_key}"))
    os.makedirs(work_parts_dir, exist_ok=True)

    total_segs = len(segments)
    eff_workers = max(1, min(workers, total_segs, (os.cpu_count() or 2)))
    rw = max(16, int(round(target_w * internal_scale)) // 2 * 2)
    rh = max(16, int(round(target_h * internal_scale)) // 2 * 2)

    print("=" * 78, flush=True)
    print("[Silent Math] 3Blue1Brown-Style Pure Open-Space Engine Initialized", flush=True)
    print(f"  Mode             : {mode.upper()} ({total_segs} segment(s), {sum(e - s for s, e in segments):.1f}s total)", flush=True)
    print(f"  Output Target    : {out_abs}", flush=True)
    print(f"  Parts Directory  : {work_parts_dir}", flush=True)
    print(f"  Output Stream    : {target_w}x{target_h} (16:9) @ {fps} FPS | H.264 (yuv420p) | No Audio (-an)", flush=True)
    print(f"  Raster Engine    : {rw}x{rh} internal canvas (scale={internal_scale:.4f}) | motion={motion_fps}Hz -> {fps}FPS", flush=True)
    print(f"  Parallel Workers : {eff_workers} isolated subprocess worker(s)", flush=True)
    print("=" * 78, flush=True)

    def _mux_available_prefix(comp: Dict[int, str], last_muxed_cnt: int) -> int:
        k = 0
        while k in comp and os.path.isfile(comp[k]):
            k += 1
        if k <= last_muxed_cnt or k == 0:
            return last_muxed_cnt
        prefix_files = [comp[i] for i in range(k)]
        tmp_mux = out_abs + ".muxing.mp4"
        if k == 1:
            shutil.copy2(prefix_files[0], tmp_mux)
        else:
            c_txt = os.path.join(work_parts_dir, "prefix_concat.txt")
            with open(c_txt, "w", encoding="utf-8") as cf:
                for pf in prefix_files:
                    cf.write(f"file '{pf}'\n")
            subprocess.check_call([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", c_txt,
                "-an", "-c", "copy", "-movflags", "+faststart", tmp_mux,
            ])
        shutil.move(tmp_mux, out_abs)
        sz_mb = os.path.getsize(out_abs) / (1024.0 * 1024.0)
        dur_so_far = sum(segments[i][1] - segments[i][0] for i in range(k))
        print(
            f"[Silent Math] [LIVE MP4 UPDATED] {os.path.basename(out_abs)} -> "
            f"{k}/{total_segs} segments ({dur_so_far:.0f}s rendered, {sz_mb:.2f} MB on disk)",
            flush=True,
        )
        return k

    completed_parts: Dict[int, str] = {}
    pending_indices: List[int] = []

    for s_idx, (ts, te) in enumerate(segments):
        part_path = os.path.join(work_parts_dir, f"part_{s_idx:04d}.mp4")
        if os.path.isfile(part_path) and os.path.getsize(part_path) > 1024:
            completed_parts[s_idx] = part_path
        else:
            pending_indices.append(s_idx)

    last_prefix_muxed = 0
    if completed_parts:
        print(f"[Silent Math] Resuming: found {len(completed_parts)}/{total_segs} completed segments on disk.", flush=True)
        last_prefix_muxed = _mux_available_prefix(completed_parts, last_prefix_muxed)

    active_procs: Dict[int, Tuple[subprocess.Popen, str, float, float]] = {}
    script_path = os.path.abspath(__file__)
    clean_env = dict(os.environ)
    clean_env["PYTHONUNBUFFERED"] = "1"

    last_heartbeat = 0.0
    while pending_indices or active_procs:
        while pending_indices and len(active_procs) < eff_workers:
            s_idx = pending_indices.pop(0)
            ts, te = segments[s_idx]
            part_path = os.path.join(work_parts_dir, f"part_{s_idx:04d}.mp4")
            sc_obj = get_scene_for_time(0.5 * (ts + te))
            print(
                f"[Silent Math] [Seg {s_idx + 1:03d}/{total_segs:03d}] START "
                f"{ts:06.1f}s–{te:06.1f}s ({sc_obj.slug}) -> {os.path.basename(part_path)}",
                flush=True,
            )
            w_cmd = [
                sys.executable, script_path,
                "--internal-worker-seg",
                str(s_idx), str(total_segs), str(ts), str(te), part_path,
                str(target_w), str(target_h), str(fps), str(motion_fps),
                str(internal_scale), preset, str(crf),
            ]
            p = subprocess.Popen(w_cmd, env=clean_env)
            active_procs[s_idx] = (p, part_path, ts, te)

        finished_now: List[int] = []
        for s_idx, (p, part_path, ts, te) in list(active_procs.items()):
            ret = p.poll()
            if ret is not None:
                if ret != 0 or not os.path.isfile(part_path):
                    raise RuntimeError(f"Worker for segment {s_idx} exited with code {ret}")
                finished_now.append(s_idx)
                completed_parts[s_idx] = part_path
                sz_kb = os.path.getsize(part_path) / 1024.0
                print(
                    f"[Silent Math] [Seg {s_idx + 1:03d}/{total_segs:03d}] DONE "
                    f"({ts:06.1f}s–{te:06.1f}s) | {os.path.basename(part_path)} ({sz_kb:.1f} KB)",
                    flush=True,
                )

        for s_idx in finished_now:
            active_procs.pop(s_idx, None)

        if finished_now:
            last_prefix_muxed = _mux_available_prefix(completed_parts, last_prefix_muxed)
            done_n = len(completed_parts)
            el = max(0.1, time.time() - t0)
            eta = (el / max(1, done_n)) * (total_segs - done_n)
            print(
                f"[Silent Math] === Overall: {done_n}/{total_segs} segments "
                f"({(done_n / total_segs) * 100.0:.1f}%) | Elapsed: {el:.1f}s | ETA: {eta:.1f}s ===",
                flush=True,
            )

        now = time.time()
        if active_procs and (now - last_heartbeat >= 2.0):
            statuses: List[str] = []
            for s_idx in sorted(active_procs.keys()):
                _, part_path, _, _ = active_procs[s_idx]
                st_file = part_path + ".status.json"
                if os.path.isfile(st_file):
                    try:
                        with open(st_file, "r", encoding="utf-8") as sf:
                            st = json.load(sf)
                        fd, ft = st.get("frame_done", 0), max(1, st.get("frame_total", 1))
                        sp = st.get("speed_fps", 0.0)
                        statuses.append(f"Seg {s_idx + 1:03d}: {fd}/{ft}f ({(fd/ft)*100:.0f}%, {sp:.0f}fps)")
                    except Exception:
                        statuses.append(f"Seg {s_idx + 1:03d}: init")
                else:
                    statuses.append(f"Seg {s_idx + 1:03d}: starting")
            print(f"  [Live Status | {now - t0:5.1f}s] " + " | ".join(statuses), flush=True)
            last_heartbeat = now

        if active_procs:
            time.sleep(0.25)

    ordered_files = [completed_parts[i] for i in range(total_segs)]
    print(f"[Silent Math] Finalizing {len(ordered_files)} segment(s) into {out_abs} ...", flush=True)

    if len(ordered_files) == 1:
        shutil.copy2(ordered_files[0], out_abs)
    else:
        concat_txt = os.path.join(work_parts_dir, "concat_list.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for pf in ordered_files:
                f.write(f"file '{pf}'\n")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-an", "-c", "copy", "-movflags", "+faststart", out_abs,
        ])

    if not keep_parts:
        shutil.rmtree(work_parts_dir, ignore_errors=True)

    elapsed = round(time.time() - t0, 2)
    probe = verify_mp4_with_ffprobe(out_abs)
    probe["render_elapsed_seconds"] = elapsed
    probe["mode"] = mode
    print(f"[Silent Math] Render Complete in {elapsed:.2f}s -> {out_abs}", flush=True)
    return probe


# ==============================================================================
# 9. COMMAND-LINE INTERFACE (CLI) ENTRY POINT
# ==============================================================================

def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silent Math — Lecture #1: Pure 3Blue1Brown-Style 2D/3D Mathematical Video Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["full", "preview", "scene", "validate"], default="full")
    parser.add_argument("--res", choices=list(RESOLUTION_PRESETS.keys()), default="4k")
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--output", type=str, default="silent_math_lecture_01_ai_ml.mp4")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--internal-scale", type=float, default=None)
    parser.add_argument("--motion-fps", type=int, default=6)
    parser.add_argument("--preset", type=str, default="ultrafast")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--parts-dir", type=str, default=None)
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--verify-file", type=str, default="")
    parser.add_argument("--internal-worker-seg", nargs=12, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli_args(argv)

    if args.internal_worker_seg:
        w = args.internal_worker_seg
        _render_single_segment_worker(
            int(w[0]), int(w[1]), float(w[2]), float(w[3]), w[4],
            int(w[5]), int(w[6]), int(w[7]), int(w[8]), float(w[9]), w[10], int(w[11]),
        )
        return 0

    if args.verify_file:
        report = verify_mp4_with_ffprobe(args.verify_file)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report.get("status") == "VERIFIED" else 1

    if args.validate or args.mode == "validate":
        w, h = RESOLUTION_PRESETS.get(args.res.lower(), (3840, 2160))
        report = run_project_validation(RenderConfig(width=w, height=h))
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report.get("status") == "PASS" else 1

    print(
        f"[Silent Math] Starting render: mode={args.mode}, resolution={args.res}, "
        f"workers={args.workers}, output={args.output}",
        flush=True,
    )
    result = render_lecture_video(
        mode=args.mode,
        res_key=args.res,
        output_path=args.output,
        scene_index=args.scene,
        workers=args.workers,
        internal_scale=args.internal_scale,
        motion_fps=args.motion_fps,
        preset=args.preset,
        crf=args.crf,
        parts_dir=args.parts_dir,
        keep_parts=args.keep_parts,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

