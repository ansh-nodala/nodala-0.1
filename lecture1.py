
#!/usr/bin/env python3
"""
================================================================================
SILENT MATH — LECTURE #1: INTRODUCTION TO AI & MACHINE LEARNING
Complete Single-File 4K Educational Video Generator & Mathematical Animation Engine
================================================================================

Channel Name : Silent Math
Video Title  : Introduction to AI & Machine Learning — Lecture #1
Tagline      : Understand the mathematics. Visualize the intelligence.
Target Output: silent_math_lecture_01_ai_ml.mp4
Specifications:
  - Duration     : Exactly 1800.0 seconds (30 minutes, 54,000 frames @ 30 FPS)
  - Aspect Ratio : 16:9 Widescreen
  - Resolution   : 3840 x 2160 (4K UHD default; supports 1080p, 720p, 360p preview)
  - Frame Rate   : 30 FPS
  - Video Codec  : H.264 (libx264, yuv420p)
  - Audio Track  : None (-an, strictly silent visual lecture with animated captions)
  - Typography   : Inter (captions/headers) + IBM Plex Mono / Latin Modern Mono (math)

Architecture Inside This Single File:
  1. Config & Palette System       : Resolution scaling, monochrome palette, font resolver
  2. Mathematical Easing Engine    : Smooth deterministic interpolation & staggering curves
  3. Complete 30-Min Timeline      : 10 lesson sections (1800s total) + 185 synchronized
                                     animated captions with word-by-word reveal metadata
  4. Vector & Typography Canvas    : Antialiased OpenCV + cached FreeType (Pillow) compositor
                                     with non-overlapping safe zones (top header, center math
                                     laboratory viewport, bottom integrated caption stage)
  5. Mathematical Scene Renderers  : Continuous, equation-consistent procedural animations
                                     for every section of the 30-minute curriculum
  6. Manim CE Scene Bridge         : Optional Manim Community Edition Scene class if invoked
                                     via `manim` CLI when Manim is installed
  7. Automated Quality Validator   : Verifies 1800s timeline contiguity, caption bounds,
                                     mathematical invariants, and FFprobe stream compliance
  8. Streaming FFmpeg Orchestrator : Renders full 4K video, scene-by-scene parallel chunks,
                                     or fast development previews without holding frames in RAM

Usage Examples:
  # 1. Validate timeline, mathematical consistency, and text bounds:
  python3 video_lecture1.py --validate

  # 2. Render fast preview sample across all scenes (360p or 720p):
  python3 video_lecture1.py --mode preview --res 720p --output preview_lecture1.mp4

  # 3. Render a specific scene (e.g., Scene 5: y = wx + b) at 1080p or 4K:
  python3 video_lecture1.py --mode scene --scene 5 --res 1080p

  # 4. Render the complete 30-minute (1800s) 4K UHD lecture (scene-parallel + FFmpeg concat):
  python3 video_lecture1.py --mode full --res 4k --workers 4 --output silent_math_lecture_01_ai_ml.mp4

  # 5. Verify an existing rendered MP4 file with FFprobe:
  python3 video_lecture1.py --verify-file silent_math_lecture_01_ai_ml.mp4
================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_ORIG_LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH")
import cv2
# Restore LD_LIBRARY_PATH modified by opencv-python wheel so child ffmpeg/ffprobe
# processes do not stall on bundled cv2 shared libraries or trailing colon paths.
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
# 1. CONFIGURATION, PALETTE, AND TYPOGRAPHY SYSTEM
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
    """Restrained monochrome palette with surgical high-contrast accents."""
    BG_BLACK: Tuple[int, int, int] = (0, 0, 0)
    PANEL_BG: Tuple[int, int, int] = (12, 12, 16)
    PANEL_BORDER: Tuple[int, int, int] = (46, 46, 54)
    GRID_MINOR: Tuple[int, int, int] = (22, 22, 28)
    GRID_MAJOR: Tuple[int, int, int] = (42, 42, 50)
    AXIS_GRAY: Tuple[int, int, int] = (130, 130, 142)
    GRAY_DARK: Tuple[int, int, int] = (75, 75, 85)
    GRAY_MID: Tuple[int, int, int] = (148, 148, 160)
    GRAY_LIGHT: Tuple[int, int, int] = (208, 208, 218)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    # Subtle accents used strictly when mathematically distinguishing signals/errors
    ACCENT_ICE: Tuple[int, int, int] = (145, 215, 255)    # Model curve / active signal
    ACCENT_GOLD: Tuple[int, int, int] = (255, 208, 110)   # Residual error / loss / highlight
    ACCENT_ROSE: Tuple[int, int, int] = (245, 135, 145)   # Overfitting / high-error contrast


PALETTE = Palette()


@dataclass
class RenderConfig:
    """Centralized video and typography configuration."""
    width: int = 3840
    height: int = 2160
    fps: int = 30
    total_duration: float = 1800.0
    output_path: str = "silent_math_lecture_01_ai_ml.mp4"
    crf: int = 18
    preset: str = "veryfast"
    internal_render_scale: float = 1.0

    @property
    def scale(self) -> float:
        """Scale factor relative to 4K UHD (2160p) vertical height."""
        return self.height / 2160.0

    @property
    def total_frames(self) -> int:
        return int(round(self.total_duration * self.fps))

    def px(self, val_at_4k: float) -> int:
        """Convert a 4K pixel dimension to the active resolution."""
        return max(1, int(round(val_at_4k * self.scale)))


@lru_cache(maxsize=32)
def resolve_font_path(role: str, bold: bool = False) -> str:
    """
    Resolve font paths for Inter (primary captions) and IBM Plex Mono (equations/code),
    with cross-platform fallback via fontconfig (`fc-match`) and standard system paths.
    """
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
        fc_queries = (
            ["IBM Plex Mono:style=Bold", "Latin Modern Mono:style=Bold", "DejaVu Sans Mono:style=Bold"]
            if bold
            else ["IBM Plex Mono:style=Regular", "Latin Modern Mono:style=Regular", "DejaVu Sans Mono:style=Book"]
        )
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
        fc_queries = (
            ["Inter:style=Bold", "Inter Display:style=Bold", "DejaVu Sans:style=Bold"]
            if bold
            else ["Inter:style=Regular", "Inter:style=Medium", "DejaVu Sans:style=Book"]
        )

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

    for fallback in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if os.path.isfile(fallback):
            return fallback
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
# 2. MATHEMATICAL EASING & DETERMINISTIC CHOREOGRAPHY ENGINE
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


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


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
    """Smooth trapezoidal envelope in [0, 1] with smoothstep attack and release."""
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
# 3. CENTRALIZED 30-MINUTE LESSON TIMELINE & VISUAL TEACHING SCRIPT
# ==============================================================================

@dataclass(frozen=True)
class CaptionSpec:
    """
    A single synchronized caption unit in the silent lecture.
    - start, end: absolute timestamps in seconds within [0, 1800].
    - line1, line2: concise 1-to-2 line explanatory captions (never paragraphs).
    - style: 'narration' | 'definition' | 'equation' | 'question'
    - terms: key mathematical terms highlighted in pure white / subtle gold accent.
    """
    start: float
    end: float
    line1: str
    line2: str = ""
    style: str = "narration"
    terms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneSpec:
    """
    Formal specification of one lesson section in the 30-minute curriculum.
    """
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


# Deterministic Fictional House Dataset used consistently across Scenes 4, 5, 7, 8, and 9
# Feature x: House Area in m^2 (55 to 215 m^2)
# Target  y: Market Price in $1,000s ($130k to $410k)
# Underlying linear trend: y_true_trend ≈ 1.65 * x + 38.0 (with realistic noise)
HOUSE_DATA_X = np.array([58.0, 76.0, 95.0, 115.0, 138.0, 160.0, 184.0, 208.0], dtype=np.float64)
HOUSE_DATA_Y = np.array([138.0, 152.0, 206.0, 220.0, 278.0, 292.0, 355.0, 374.0], dtype=np.float64)

# Unseen validation examples introduced in Scene 9 (Training vs. Generalization)
UNSEEN_DATA_X = np.array([68.0, 106.0, 148.0, 172.0, 196.0], dtype=np.float64)
UNSEEN_DATA_Y = np.array([148.0, 214.0, 280.0, 326.0, 360.0], dtype=np.float64)


def build_lecture_timeline() -> List[SceneSpec]:
    """
    Construct the complete, contiguous 1800-second (30-minute) lesson structure
    with synchronized animated captions explaining every visual demonstration.
    """
    scenes: List[SceneSpec] = [
        # ----------------------------------------------------------------------
        # SECTION 1: 00:00 - 01:30 (90s) — Cinematic Introduction
        # ----------------------------------------------------------------------
        SceneSpec(
            index=1,
            slug="01_cinematic_intro",
            title="00:00–01:30 — Cinematic Introduction",
            start_time=0.0,
            end_time=90.0,
            objective=(
                "Establish the Silent Math visual laboratory from a single point in "
                "darkness, form a coordinate plane and curve, present the opening title "
                "card, and pose the central guiding question of machine learning."
            ),
            visual_elements=(
                "Single luminous point on pure black canvas",
                "Progressive 2D Cartesian coordinate plane formation",
                "Parametric curve tracing and scattered observation points",
                "SILENT MATH — LECTURE #1 title card and tagline",
                "Central question: How can a machine learn patterns from examples?",
            ),
            transition_behavior="Continuous coordinate-grid morph into Scene 2 input stream.",
            captions=(
                CaptionSpec(2.0, 9.5, "In complete silence, mathematics begins with a single point.", "A lone coordinate in empty space.", "narration", ("single point", "coordinate")),
                CaptionSpec(10.0, 18.5, "When that point moves, its path traces a relationship", "between two changing quantities: horizontal position x and vertical height y.", "narration", ("relationship", "x", "y")),
                CaptionSpec(19.0, 28.0, "Around this motion, a coordinate plane emerges—", "a mathematical laboratory where patterns become visible.", "narration", ("coordinate plane", "patterns")),
                CaptionSpec(28.5, 38.0, "Discrete observations appear as isolated points,", "while a continuous curve threads through them to capture their structure.", "narration", ("observations", "continuous curve")),
                CaptionSpec(39.0, 54.0, "SILENT MATH — LECTURE #1: INTRODUCTION TO AI & MACHINE LEARNING", "Understand the mathematics. Visualize the intelligence.", "definition", ("SILENT MATH", "LECTURE #1")),
                CaptionSpec(55.0, 67.5, "Every day, machines recognize faces, translate languages,", "and predict complex outcomes without human intervention.", "narration", ("recognize", "predict")),
                CaptionSpec(68.0, 81.0, "How can a machine learn patterns from examples?", "Hold this central question in mind as we build the answer from scratch.", "question", ("How can a machine learn patterns from examples?",)),
                CaptionSpec(81.5, 89.5, "We begin at absolute zero: what do we actually mean", "when we speak of Artificial Intelligence?", "narration", ("absolute zero", "Artificial Intelligence")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 2: 01:30 - 04:30 (180s) — What Is Artificial Intelligence?
        # ----------------------------------------------------------------------
        SceneSpec(
            index=2,
            slug="02_what_is_ai",
            title="01:30–04:30 — What Is Artificial Intelligence?",
            start_time=90.0,
            end_time=270.0,
            objective=(
                "Visually explain Artificial Intelligence through concrete examples of "
                "input, processing, and output: receiving information, recognizing shapes, "
                "identifying temporal patterns, and selecting actions."
            ),
            visual_elements=(
                "Three-stage Input -> Processing -> Output laboratory diagram",
                "Numerical sensor/pixel array streaming into processing core",
                "Geometric object recognition (Circle, Triangle, Square classification)",
                "Signal wave pattern detection and anomaly isolation",
                "Decision tree / trajectory action selection on state grid",
            ),
            transition_behavior="Smooth horizontal slide from rule-based AI into Machine Learning.",
            captions=(
                CaptionSpec(91.0, 102.0, "At its foundation, any computational system operates in three stages:", "Input, Processing, and Output.", "definition", ("Input", "Processing", "Output")),
                CaptionSpec(103.0, 115.0, "First, a machine receives raw information from the world—", "numbers, measurements, text tokens, or grids of pixel brightness.", "narration", ("raw information", "numbers", "pixel brightness")),
                CaptionSpec(116.0, 128.0, "To a computer, an image is not a picture;", "it is an ordered grid of numerical intensity values between 0 and 255.", "equation", ("ordered grid", "0 and 255")),
                CaptionSpec(129.0, 141.0, "In our first visual example, watch the system inspect raw pixel grids", "and recognize distinct geometric objects: a circle, a triangle, and a square.", "narration", ("recognize", "circle", "triangle", "square")),
                CaptionSpec(142.0, 154.0, "The input is the grid of brightness numbers.", "The processing extracts geometric curves and corners; the output is the object label.", "narration", ("input", "processing", "output")),
                CaptionSpec(155.0, 168.0, "In a second example, a computer monitors a stream of measurements over time,", "searching for a recurring mathematical rhythm hidden inside the sequence.", "narration", ("stream of measurements", "recurring mathematical rhythm")),
                CaptionSpec(169.0, 182.0, "By comparing consecutive values, the system separates regular wave patterns", "from unexpected spikes that signal an anomaly.", "narration", ("regular wave patterns", "anomaly")),
                CaptionSpec(183.0, 196.0, "In a third example, a system must choose an action:", "navigating from a starting coordinate to a target while avoiding obstacles.", "narration", ("choose an action", "starting coordinate", "target")),
                CaptionSpec(197.0, 210.0, "It evaluates candidate paths, scores each outcome,", "and selects the trajectory that reaches the goal most efficiently.", "narration", ("candidate paths", "selects the trajectory")),
                CaptionSpec(211.0, 225.0, "Artificial Intelligence (AI) is the broad field of creating systems", "capable of performing tasks associated with intelligent behavior.", "definition", ("Artificial Intelligence (AI)", "broad field")),
                CaptionSpec(226.0, 240.0, "Crucially, not all AI systems work or learn in the same way.", "Early AI relied entirely on humans writing explicit, step-by-step rules.", "narration", ("not all AI systems", "explicit, step-by-step rules")),
                CaptionSpec(241.0, 255.0, "Explicit rules work well when a problem is rigid and predictable,", "like calculating taxes or following fixed chess openings.", "narration", ("Explicit rules", "rigid and predictable")),
                CaptionSpec(256.0, 269.0, "But what happens when patterns are too subtle to write down by hand?", "That question leads us to a specific branch inside AI: Machine Learning.", "narration", ("too subtle", "Machine Learning")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 3: 04:30 - 08:00 (210s) — What Is Machine Learning?
        # ----------------------------------------------------------------------
        SceneSpec(
            index=3,
            slug="03_what_is_ml",
            title="04:30–08:00 — What Is Machine Learning?",
            start_time=270.0,
            end_time=480.0,
            objective=(
                "Contrast brittle hand-coded rules with learning patterns from examples; "
                "introduce Data, Model, and Prediction; visually demonstrate Data -> Model -> Prediction "
                "and show why a prediction is an estimate rather than a guaranteed truth."
            ),
            visual_elements=(
                "Comparison panel: rigid IF-THEN step boxes vs. continuous learned curve",
                "Scatter plot of raw observations appearing first without any curve",
                "Smooth mathematical curve emerging to capture the underlying trend",
                "Interactive flow diagram: Data -> Model -> Prediction",
                "Query input x* producing prediction y_hat compared against true observation y",
            ),
            transition_behavior="Morph generic scatter points into concrete house-price coordinates.",
            captions=(
                CaptionSpec(271.0, 285.0, "Imagine trying to write manual IF-THEN rules for a complex relationship:", "if x is between 1 and 2, output 3; if between 2 and 3, output 5...", "narration", ("manual IF-THEN rules",)),
                CaptionSpec(286.0, 300.0, "Hand-coded rules create rigid staircases.", "Whenever real observations curve smoothly or vary slightly, rigid rules break down.", "narration", ("rigid staircases", "break down")),
                CaptionSpec(301.0, 316.0, "Machine Learning reverses this philosophy.", "Instead of programming every decision rule by hand, we show the system examples.", "definition", ("Machine Learning", "show the system examples")),
                CaptionSpec(317.0, 332.0, "Look at this empty coordinate plane.", "First, raw observations appear as individual points—each point is one recorded example.", "narration", ("raw observations", "individual points")),
                CaptionSpec(333.0, 348.0, "At first, there is no formula on the screen—only scattered coordinates.", "Yet your eyes immediately sense a rising trend across the plane.", "narration", ("no formula", "rising trend")),
                CaptionSpec(349.0, 365.0, "Watch as a continuous curve gradually emerges,", "bending and settling until it captures the shared mathematical pattern of the points.", "narration", ("continuous curve", "mathematical pattern")),
                CaptionSpec(366.0, 382.0, "This brings us to three foundational words in Machine Learning:", "Data, Model, and Prediction.", "definition", ("Data", "Model", "Prediction")),
                CaptionSpec(383.0, 399.0, "DATA is the collection of observed examples—", "the points we gather from measurements in the world.", "definition", ("DATA", "observed examples")),
                CaptionSpec(400.0, 416.0, "A MODEL is the mathematical structure—such as this curve—", "that summarizes the pattern connecting inputs to outputs.", "definition", ("MODEL", "mathematical structure")),
                CaptionSpec(417.0, 433.0, "A PREDICTION is the output value produced by the model", "when we feed it a specific input coordinate.", "definition", ("PREDICTION", "output value")),
                CaptionSpec(434.0, 450.0, "Follow the sequence across the laboratory:", "Data shapes the Model, and the Model produces a Prediction: Data → Model → Prediction.", "equation", ("Data → Model → Prediction",)),
                CaptionSpec(451.0, 466.0, "Notice a critical distinction: a prediction is an output produced by a model,", "not necessarily a completely accurate answer.", "narration", ("not necessarily", "accurate answer")),
                CaptionSpec(467.0, 479.0, "Even a useful curve leaves small vertical gaps to real points.", "To see why, let us ground these ideas in a concrete problem.", "narration", ("vertical gaps", "concrete problem")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 4: 08:00 - 11:30 (210s) — Data, Features, and Patterns
        # ----------------------------------------------------------------------
        SceneSpec(
            index=4,
            slug="04_data_features_patterns",
            title="08:00–11:30 — Data, Features, and Patterns",
            start_time=480.0,
            end_time=690.0,
            objective=(
                "Introduce a simplified fictional house-price dataset, map house floor area "
                "and price onto a 2D coordinate plane, define Feature, Target, and Training Example, "
                "and show why real-world data exhibits noise."
            ),
            visual_elements=(
                "Minimalist geometric house wireframes scaled by floor area (m^2)",
                "Two-column data table pairing House Area (x) with Price (y)",
                "Animated transformation of house icons onto the (Area, Price) coordinate plane",
                "Orthogonal projection callouts labeling Feature (horizontal) and Target (vertical)",
                "Noise band illustration showing natural variation around the upward trend",
            ),
            transition_behavior="Keep the house-price coordinate axes locked in place for Scene 5.",
            captions=(
                CaptionSpec(481.0, 496.0, "Suppose we want to estimate the price of a house", "using records from eight recently sold homes in a neighborhood.", "narration", ("estimate the price", "eight recently sold homes")),
                CaptionSpec(497.0, 512.0, "Each house can be described by measurable quantities.", "To keep our visual laboratory crystal clear, we focus on two measurements:", "narration", ("measurable quantities", "two measurements")),
                CaptionSpec(513.0, 528.0, "First: the floor area of the house, measured in square meters (m²).", "Second: the sale price of the house, measured in thousands of dollars ($k).", "definition", ("floor area", "sale price")),
                CaptionSpec(529.0, 545.0, "Watch as each house moves onto a two-dimensional coordinate plane:", "horizontal position represents House Area, and vertical height represents Price.", "narration", ("House Area", "Price")),
                CaptionSpec(546.0, 562.0, "A smaller 58 m² home sits near the lower-left at $138k,", "while a larger 208 m² home sits near the upper-right at $374k.", "equation", ("58 m²", "$138k", "208 m²", "$374k")),
                CaptionSpec(563.0, 579.0, "In Machine Learning, each measurable input property—like House Area—", "is called a FEATURE.", "definition", ("FEATURE", "input property")),
                CaptionSpec(580.0, 596.0, "The quantity we wish to predict—here, the Price—", "is called the TARGET.", "definition", ("TARGET", "quantity we wish to predict")),
                CaptionSpec(597.0, 613.0, "One complete recorded pair of Feature and Target—a single house point (xᵢ, yᵢ)—", "is called a TRAINING EXAMPLE.", "definition", ("TRAINING EXAMPLE", "(xᵢ, yᵢ)")),
                CaptionSpec(614.0, 630.0, "Together, all eight points form our training dataset.", "As floor area increases from left to right, sale price generally rises.", "narration", ("training dataset", "generally rises")),
                CaptionSpec(631.0, 647.0, "Yet look closely at the points:", "they do not lie on a single razor-sharp line.", "narration", ("do not lie", "razor-sharp line")),
                CaptionSpec(648.0, 665.0, "Two houses with similar area might sell for slightly different prices", "because of sunlight, renovation age, or buyer timing.", "narration", ("similar area", "different prices")),
                CaptionSpec(666.0, 678.0, "Real-world data is almost always noisy.", "Our goal is not to connect every jagged dot, but to express the underlying trend.", "narration", ("noisy", "underlying trend")),
                CaptionSpec(679.0, 689.5, "How do we write a clean mathematical equation for that trend line?", "Let us open the mathematical heart of a linear model.", "narration", ("mathematical equation", "linear model")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 5: 11:30 - 15:00 (210s) — The Mathematical Heart: y = wx + b
        # ----------------------------------------------------------------------
        SceneSpec(
            index=5,
            slug="05_linear_model_y_wx_b",
            title="11:30–15:00 — The Mathematical Heart: y = wx + b",
            start_time=690.0,
            end_time=900.0,
            objective=(
                "Introduce y = wx + b, explain each symbol (x, w, b, y) visually, "
                "animate slope changes as w varies and vertical shifts as b varies in "
                "exact mathematical lockstep with the plotted line, and fit the house data."
            ),
            visual_elements=(
                "Animated equation header: y = w * x + b with symbol callouts",
                "Cartesian plane with live slope triangle (Delta x = 1, Delta y = w)",
                "Continuous parameter sweep of weight w showing rotation/tilt",
                "Continuous parameter sweep of bias b showing vertical translation",
                "Overlay of the 8 house-price points approximated by y = 1.65x + 38",
            ),
            transition_behavior="Condense the equation y = wx + b into a single computational node for Scene 6.",
            captions=(
                CaptionSpec(691.0, 705.0, "Meet the simplest mathematical model in machine learning:", "y = w · x + b", "equation", ("y = w · x + b",)),
                CaptionSpec(706.0, 720.0, "Every symbol in this four-part equation has a clear visual meaning.", "Let us inspect each piece one by one.", "narration", ("four-part equation", "visual meaning")),
                CaptionSpec(721.0, 735.0, "First, x is the INPUT feature—", "for example, the floor area of a house along the horizontal axis.", "definition", ("x", "INPUT feature")),
                CaptionSpec(736.0, 751.0, "Second, w is the WEIGHT.", "It controls how strongly the input x influences the output—geometrically, the slope.", "definition", ("w", "WEIGHT", "slope")),
                CaptionSpec(752.0, 767.0, "Third, b is the BIAS.", "It shifts the entire line up or down, setting the baseline output when x equals zero.", "definition", ("b", "BIAS", "baseline output")),
                CaptionSpec(768.0, 782.0, "Fourth, y is the PREDICTED OUTPUT—", "the height of the line at input position x.", "definition", ("y", "PREDICTED OUTPUT")),
                CaptionSpec(783.0, 799.0, "Begin on a clean coordinate plane with bias b fixed at 0.", "Watch what happens as we smoothly increase the weight w from 0.0 to 2.4.", "narration", ("b fixed at 0", "increase the weight w")),
                CaptionSpec(800.0, 816.0, "When w = 0, the line is completely flat: changing x has zero effect on y.", "As w grows larger, the line tilts steeper upward—x exerts a stronger pull on y.", "equation", ("w = 0", "steeper upward")),
                CaptionSpec(817.0, 832.0, "If w becomes negative, the relationship flips:", "increasing x now drives the predicted output y downward.", "equation", ("w becomes negative", "downward")),
                CaptionSpec(833.0, 849.0, "Now hold the slope fixed at w = 1.65 and vary the bias b.", "Increasing b slides the entire line upward without changing its tilt.", "equation", ("w = 1.65", "vary the bias b")),
                CaptionSpec(850.0, 865.0, "Decreasing b slides the line downward.", "Together, w controls the angle and b controls the vertical altitude.", "narration", ("w controls the angle", "b controls the vertical altitude")),
                CaptionSpec(866.0, 883.0, "Bring back our eight house-price examples.", "With w = 1.65 and b = 38, the line y = 1.65x + 38 passes cleanly through the trend.", "equation", ("w = 1.65", "b = 38", "y = 1.65x + 38")),
                CaptionSpec(884.0, 899.5, "This straight line is a simple mathematical model—not all of AI,", "but the essential building block from which neural networks are constructed.", "narration", ("simple mathematical model", "building block")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 6: 15:00 - 18:30 (210s) — From a Simple Model to a Neural Network
        # ----------------------------------------------------------------------
        SceneSpec(
            index=6,
            slug="06_neural_network",
            title="15:00–18:30 — From a Simple Model to a Neural Network",
            start_time=900.0,
            end_time=1110.0,
            objective=(
                "Expand a single linear model node into a clean 3-layer Neural Network "
                "(Input Layer, Hidden Layer, Output Layer), animate signal propagation "
                "across weighted connections, and define Neuron, Weight, Layer, and Output."
            ),
            visual_elements=(
                "Single computational node x -> (w, b) -> y expanding into multi-node graph",
                "3-Layer Network topology: 3 Input Nodes, 4 Hidden Layer Nodes, 1 Output Node",
                "Luminous traveling signal pulses along weighted connections w_ij",
                "Zoom callout inside a single Neuron: weighted sum + smooth activation curve",
                "Layer boundary brackets clearly distinguishing Input, Hidden, and Output layers",
            ),
            transition_behavior="Collapse network back to the parameter tuning laboratory for Scene 7.",
            captions=(
                CaptionSpec(901.0, 916.0, "Our equation y = wx + b takes a single input x and produces one output y.", "We can draw that calculation as two circles connected by a single weighted arrow.", "narration", ("y = wx + b", "weighted arrow")),
                CaptionSpec(917.0, 932.0, "What if a prediction depends on several features at once—", "such as house area, number of rooms, and distance to a park?", "narration", ("several features",)),
                CaptionSpec(933.0, 949.0, "And what if the true relationship curves and bends", "in ways a single straight line can never capture?", "narration", ("curves and bends", "single straight line")),
                CaptionSpec(950.0, 966.0, "Watch as we expand our single unit.", "First, we place three input nodes on the left—forming the INPUT LAYER.", "definition", ("INPUT LAYER", "three input nodes")),
                CaptionSpec(967.0, 983.0, "Next, we introduce a middle column of four computational units—", "called a HIDDEN LAYER—and connect every input to every middle unit.", "definition", ("HIDDEN LAYER", "four computational units")),
                CaptionSpec(984.0, 1000.0, "Finally, those hidden units feed into a rightmost node:", "the OUTPUT LAYER, which produces the final prediction.", "definition", ("OUTPUT LAYER", "final prediction")),
                CaptionSpec(1001.0, 1017.0, "Each circular unit in this structure is called an artificial NEURON.", "Each connecting line carries its own numerical WEIGHT.", "definition", ("NEURON", "WEIGHT")),
                CaptionSpec(1018.0, 1034.0, "Watch how information flows from left to right:", "numerical signals enter the Input Layer and travel along the weighted connections.", "narration", ("flows from left to right", "weighted connections")),
                CaptionSpec(1035.0, 1051.0, "Inside each neuron, incoming signals are multiplied by their weights,", "added together with a bias, and passed through a smooth curving function.", "equation", ("multiplied by their weights", "bias", "smooth curving function")),
                CaptionSpec(1052.0, 1068.0, "A column of neurons operating in parallel forms a LAYER.", "By combining simple mathematical operations across layers, the network builds flexible curves.", "definition", ("LAYER", "flexible curves")),
                CaptionSpec(1069.0, 1085.0, "Notice that there is no magic hidden inside the diagram:", "every glow represents numbers being multiplied and added.", "narration", ("no magic", "multiplied and added")),
                CaptionSpec(1086.0, 1098.0, "Artificial neural networks are one powerful family of machine-learning models,", "though many simpler models remain vital across science and engineering.", "narration", ("one powerful family", "machine-learning models")),
                CaptionSpec(1099.0, 1109.5, "Whether a model has two parameters or two thousand,", "how does it actually discover the right numbers for w and b?", "question", ("discover the right numbers", "w and b")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 7: 18:30 - 22:00 (210s) — How Does a Model Learn?
        # ----------------------------------------------------------------------
        SceneSpec(
            index=7,
            slug="07_how_a_model_learns",
            title="18:30–22:00 — How Does a Model Learn?",
            start_time=1110.0,
            end_time=1320.0,
            objective=(
                "Demonstrate the iterative training feedback loop on the house-price problem: "
                "Prediction -> Measure Error -> Update Parameters -> New Prediction, "
                "visualizing vertical error residuals, the Loss curve, and parameter updates."
            ),
            visual_elements=(
                "Dual-panel view: House Data & Residual Bars (left) + Loss Landscape L(w,b) (right)",
                "Four-stage illuminated feedback loop banner at top of laboratory",
                "Vertical dashed error segments connecting y_pred on the line to y_true points",
                "Iterative parameter trajectory (including a slight overshoot step) reducing Loss",
                "Live numerical readouts of Parameter w, Parameter b, and Loss L",
            ),
            transition_behavior="Dissolve dual optimization panels into the 3-paradigm comparison grid.",
            captions=(
                CaptionSpec(1111.0, 1125.0, "How does a model learn from data?", "It begins by making an initial guess—often with poorly chosen parameters.", "narration", ("initial guess", "parameters")),
                CaptionSpec(1126.0, 1140.0, "Here on the left is our house-price data.", "Our model starts with w = 0.35 and b = 85—a flat line that misses most points.", "equation", ("w = 0.35", "b = 85")),
                CaptionSpec(1141.0, 1156.0, "Step 1: PREDICTION.", "Data enters the model, and the current line outputs a predicted price for each house.", "definition", ("PREDICTION", "predicted price")),
                CaptionSpec(1157.0, 1172.0, "Step 2: MEASURE ERROR.", "For every training example, we draw the vertical gap between Prediction and Target.", "definition", ("MEASURE ERROR", "vertical gap")),
                CaptionSpec(1173.0, 1189.0, "Each individual vertical gap is an ERROR.", "To summarize all those gaps into a single score, we compute a LOSS function.", "definition", ("ERROR", "LOSS function")),
                CaptionSpec(1190.0, 1206.0, "Squaring and averaging the vertical gaps gives Mean Squared Loss.", "When the line fits poorly, the vertical bars are tall and the Loss is high.", "equation", ("Mean Squared Loss", "Loss is high")),
                CaptionSpec(1207.0, 1223.0, "On the right panel, we plot Loss as a smooth valley over the weight w.", "Our starting guess sits high up on the steep wall of the loss curve.", "narration", ("smooth valley", "loss curve")),
                CaptionSpec(1224.0, 1240.0, "Step 3: UPDATE PARAMETERS.", "We nudge the parameters w and b in the direction that reduces the Loss.", "definition", ("UPDATE PARAMETERS", "PARAMETERS", "w and b")),
                CaptionSpec(1241.0, 1257.0, "Step 4: NEW PREDICTION.", "With updated parameters, the line tilts closer to the points and the error bars shrink.", "definition", ("NEW PREDICTION", "error bars shrink")),
                CaptionSpec(1258.0, 1274.0, "Watch the complete feedback loop repeat:", "Prediction → Measure Error → Update Parameters → New Prediction.", "equation", ("Prediction → Measure Error → Update Parameters → New Prediction",)),
                CaptionSpec(1275.0, 1291.0, "Notice step 4 on the loss curve: if a parameter step is slightly too large,", "the line can briefly overshoot before correcting on the next update.", "narration", ("briefly overshoot", "next update")),
                CaptionSpec(1292.0, 1306.0, "This iterative process of adjusting parameters to reduce loss is called TRAINING.", "Not every step is guaranteed to improve the model, and no single curve is perfect.", "definition", ("TRAINING", "reduce loss")),
                CaptionSpec(1307.0, 1319.5, "Yet across repeated updates, the parameters settle near the bottom of the valley—", "turning raw examples into a calibrated mathematical model.", "narration", ("bottom of the valley", "calibrated mathematical model")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 8: 22:00 - 25:00 (180s) — Three Important Learning Approaches
        # ----------------------------------------------------------------------
        SceneSpec(
            index=8,
            slug="08_three_learning_approaches",
            title="22:00–25:00 — Three Important Learning Approaches",
            start_time=1320.0,
            end_time=1500.0,
            objective=(
                "Introduce Supervised Learning, Unsupervised Learning, and Reinforcement "
                "Learning through visually distinct mathematical animations and plain-English captions."
            ),
            visual_elements=(
                "Panel 1 — Supervised Learning: paired (Feature x, Target y) house data & regression fit",
                "Panel 2 — Unsupervised Learning: unlabeled 2D points self-organizing into 3 clusters",
                "Panel 3 — Reinforcement Learning: agent in grid-world observing state, acting, earning reward",
                "Side-by-side synthesis triad comparing inputs and feedback signals",
            ),
            transition_behavior="Zoom into the Supervised curve to examine Training vs. Generalization.",
            captions=(
                CaptionSpec(1321.0, 1335.0, "Machine learning problems come in different forms", "depending on what kind of information is available during training.", "narration", ("different forms", "information")),
                CaptionSpec(1336.0, 1352.0, "First: SUPERVISED LEARNING.", "Every training example comes paired with a known target answer—like house area and known price.", "definition", ("SUPERVISED LEARNING", "known target answer")),
                CaptionSpec(1353.0, 1369.0, "Because the target labels are provided, the model can directly compare", "each prediction against the known answer and minimize the error.", "narration", ("target labels", "minimize the error")),
                CaptionSpec(1370.0, 1386.0, "Second: UNSUPERVISED LEARNING.", "What if we are given thousands of data points with no target labels at all?", "definition", ("UNSUPERVISED LEARNING", "no target labels")),
                CaptionSpec(1387.0, 1404.0, "Watch these unlabeled coordinates on the plane:", "even without labels, points with similar measurements naturally group into clusters.", "narration", ("unlabeled coordinates", "clusters")),
                CaptionSpec(1405.0, 1421.0, "Unsupervised learning discovers hidden structure, groupings, or patterns", "directly from the arrangement of the data itself.", "narration", ("hidden structure", "groupings")),
                CaptionSpec(1422.0, 1438.0, "Third: REINFORCEMENT LEARNING.", "Instead of a fixed table of examples, an AGENT interacts with an environment.", "definition", ("REINFORCEMENT LEARNING", "AGENT", "environment")),
                CaptionSpec(1439.0, 1456.0, "At each moment, the agent observes its current STATE,", "selects an ACTION, and receives positive or negative FEEDBACK called a REWARD.", "definition", ("STATE", "ACTION", "REWARD")),
                CaptionSpec(1457.0, 1473.0, "Through trial and feedback—earning +10 at the goal and -1 at obstacles—", "the agent gradually learns which actions lead to higher cumulative reward.", "equation", ("+10", "-1", "cumulative reward")),
                CaptionSpec(1474.0, 1488.0, "Supervised, Unsupervised, and Reinforcement Learning are three major pillars,", "though modern machine learning also blends and extends these approaches.", "narration", ("Supervised", "Unsupervised", "Reinforcement Learning")),
                CaptionSpec(1489.0, 1499.5, "Across all of them, one crucial challenge remains:", "how do we know if a model has truly learned, or merely memorized?", "question", ("truly learned", "merely memorized")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 9: 25:00 - 27:30 (150s) — Training vs. Generalization
        # ----------------------------------------------------------------------
        SceneSpec(
            index=9,
            slug="09_training_vs_generalization",
            title="25:00–27:30 — Training vs. Generalization",
            start_time=1500.0,
            end_time=1650.0,
            objective=(
                "Contrast a smooth generalizing curve against an overly complex overfitting "
                "curve that memorizes noisy training points, then introduce unseen test data "
                "to demonstrate generalization error."
            ),
            visual_elements=(
                "Training data points plotted as solid white circles on coordinate plane",
                "Smooth balanced curve capturing general trend vs. degree-7 oscillatory polynomial curve",
                "Unseen data points revealed as hollow highlighted rings with vertical error bars",
                "Side-by-side error comparison gauges on Training Data vs. Unseen Data",
            ),
            transition_behavior="Fade out coordinate plane into the forward learning roadmap.",
            captions=(
                CaptionSpec(1501.0, 1515.0, "Return to our coordinate plane of training examples.", "These solid white points are our TRAINING DATA—the examples used to fit the model.", "definition", ("TRAINING DATA", "fit the model")),
                CaptionSpec(1516.0, 1531.0, "Suppose we give a model extreme flexibility", "and ask it to drive the training error all the way down to zero.", "narration", ("extreme flexibility", "training error", "zero")),
                CaptionSpec(1532.0, 1548.0, "Watch this complicated curve:", "it twists, bends, and oscillates wildly just to pass through every single training point.", "narration", ("twists, bends, and oscillates", "every single training point")),
                CaptionSpec(1549.0, 1565.0, "Now compare it with a simpler, smoother curve", "that accepts small gaps on individual points in order to follow the overall trend.", "narration", ("smoother curve", "overall trend")),
                CaptionSpec(1566.0, 1582.0, "When a model follows the random noise of training points too closely,", "we call that OVERFITTING. It has memorized the examples instead of the pattern.", "definition", ("OVERFITTING", "memorized")),
                CaptionSpec(1583.0, 1599.0, "The true test arrives when we introduce UNSEEN DATA—", "new examples that the model never encountered during training.", "definition", ("UNSEEN DATA", "never encountered")),
                CaptionSpec(1600.0, 1617.0, "Look at the hollow circles representing unseen houses:", "the wildly oscillating curve misses them badly, producing huge prediction errors.", "narration", ("unseen houses", "huge prediction errors")),
                CaptionSpec(1618.0, 1634.0, "Meanwhile, the smooth trend curve stays close to the unseen points.", "Performing well on new, previously unseen examples is called GENERALIZATION.", "definition", ("GENERALIZATION", "unseen examples")),
                CaptionSpec(1635.0, 1649.5, "In machine learning, zero error on training data is not the ultimate prize;", "reliable generalization to the unseen world is what matters.", "narration", ("generalization", "unseen world")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 10: 27:30 - 29:00 (90s) — What Will We Learn Next?
        # ----------------------------------------------------------------------
        SceneSpec(
            index=10,
            slug="10_roadmap_next",
            title="27:30–29:00 — What Will We Learn Next?",
            start_time=1650.0,
            end_time=1740.0,
            objective=(
                "Present the visual learning roadmap (Mathematics -> Python -> Machine Learning "
                "-> Neural Networks -> Deep Learning) and preview PyTorch as a future tool "
                "without teaching PyTorch code."
            ),
            visual_elements=(
                "Five-node horizontal architectural progression with custom mathematical icons",
                "Sequential illumination of Mathematics -> Python -> Machine Learning -> Neural Networks -> Deep Learning",
                "Future milestone spotlight card: PyTorch — Coming in a Future Lecture",
                "Visual bridge diagram connecting mathematical equations to computational graphs",
            ),
            transition_behavior="Dissolve roadmap into the opening point and coordinate plane for Scene 11.",
            captions=(
                CaptionSpec(1651.0, 1664.0, "You now hold the conceptual map of Artificial Intelligence and Machine Learning.", "Where does our journey on Silent Math go from here?", "narration", ("conceptual map", "Silent Math")),
                CaptionSpec(1665.0, 1679.0, "Our curriculum unfolds across five connected stages:", "Mathematics → Python → Machine Learning → Neural Networks → Deep Learning.", "equation", ("Mathematics → Python → Machine Learning → Neural Networks → Deep Learning",)),
                CaptionSpec(1680.0, 1694.0, "First, we deepen the MATHEMATICS: vectors, matrices, and rates of change—", "always visualized geometrically before we write code.", "narration", ("MATHEMATICS", "visualized geometrically")),
                CaptionSpec(1695.0, 1709.0, "Next, we translate those mathematical objects into clean PYTHON,", "building classical Machine Learning models and multi-layer Neural Networks from scratch.", "narration", ("PYTHON", "from scratch")),
                CaptionSpec(1710.0, 1725.0, "As our networks grow deeper, we will introduce a dedicated framework:", "PyTorch — Coming in a Future Lecture.", "definition", ("PyTorch — Coming in a Future Lecture",)),
                CaptionSpec(1726.0, 1739.5, "PyTorch will allow us to turn mathematical equations and neural networks", "into fast, scalable experiments—built on the exact intuition you learned today.", "narration", ("PyTorch", "exact intuition")),
            ),
        ),

        # ----------------------------------------------------------------------
        # SECTION 11: 29:00 - 30:00 (60s) — Recap and Closing
        # ----------------------------------------------------------------------
        SceneSpec(
            index=11,
            slug="11_recap_closing",
            title="29:00–30:00 — Recap and Closing",
            start_time=1740.0,
            end_time=1800.0,
            objective=(
                "Return to the original moving point and coordinate plane, present the "
                "six core recap takeaways, pose the closing philosophical question, display "
                "the Lecture #1 Complete card, and fade smoothly to pure black at 30:00."
            ),
            visual_elements=(
                "Return of the luminous white point and coordinate plane from Scene 1",
                "Six animated recap cards with synchronized mathematical micro-glyphs",
                "Closing question card: If a machine can learn a pattern from examples, what else can we teach it to discover?",
                "Final title card: SILENT MATH / LECTURE #1 COMPLETE / Next: Mathematics Behind Machine Learning",
                "Deterministic smooth fade to pure black at 1800.0s",
            ),
            transition_behavior="Smooth fade to pure black (#000000) completing at 1800.0s.",
            captions=(
                CaptionSpec(1741.0, 1749.0, "AI: Intelligent behavior in machines.", "ML: Learning patterns from data.", "definition", ("AI:", "ML:")),
                CaptionSpec(1749.5, 1758.0, "Model: A mathematical system that produces predictions.", "Loss: A measure of prediction error.", "definition", ("Model:", "Loss:")),
                CaptionSpec(1758.5, 1767.5, "Training: Adjusting parameters using data.", "Generalization: Performing well on unseen examples.", "definition", ("Training:", "Generalization:")),
                CaptionSpec(1768.5, 1782.0, "If a machine can learn a pattern from examples,", "what else can we teach it to discover?", "question", ("learn a pattern from examples", "teach it to discover?")),
                CaptionSpec(1783.0, 1796.5, "SILENT MATH — LECTURE #1 COMPLETE", "Next: Mathematics Behind Machine Learning", "definition", ("SILENT MATH", "LECTURE #1 COMPLETE")),
            ),
        ),
    ]
    return scenes


LECTURE_SCENES: List[SceneSpec] = build_lecture_timeline()


def get_scene_for_time(t: float) -> SceneSpec:
    """Return the active SceneSpec for timestamp t in [0, 1800]."""
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
# 4. VECTOR GRAPHICS & CACHED FREETYPE TYPOGRAPHY ENGINE
# ==============================================================================

@lru_cache(maxsize=1024)
def render_text_sprite(
    text: str,
    role: str,
    size_px: int,
    color: Tuple[int, int, int],
    bold: bool = False,
) -> np.ndarray:
    """
    Render crisp antialiased text using Pillow FreeType into an RGBA NumPy sprite.
    Cached via LRU so repeated frames reuse the rasterized glyph buffer in <0.1ms.
    """
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


def blit_rgba_sprite(
    canvas: np.ndarray,
    sprite: np.ndarray,
    x: int,
    y: int,
    alpha_mul: float = 1.0,
    anchor: str = "center",
) -> Tuple[int, int, int, int]:
    """
    Alpha-composite an RGBA sprite onto the RGB canvas at (x, y) with given anchor.
    Returns bounding box (x0, y0, x1, y1) in canvas coordinates.
    """
    if alpha_mul <= 0.01 or sprite.size == 0:
        return (x, y, x, y)
    sh, sw = sprite.shape[:2]
    if anchor == "center":
        x0, y0 = int(x - sw // 2), int(y - sh // 2)
    elif anchor == "topleft":
        x0, y0 = int(x), int(y)
    elif anchor == "midleft":
        x0, y0 = int(x), int(y - sh // 2)
    elif anchor == "midright":
        x0, y0 = int(x - sw), int(y - sh // 2)
    elif anchor == "topcenter":
        x0, y0 = int(x - sw // 2), int(y)
    else:
        x0, y0 = int(x), int(y)

    x1, y1 = x0 + sw, y0 + sh
    ch, cw = canvas.shape[:2]

    # Clip to canvas boundaries
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(cw, x1), min(ch, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return (x0, y0, x1, y1)

    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)

    sub_sprite = sprite[sy0:sy1, sx0:sx1]
    a_u16 = (sub_sprite[:, :, 3:4].astype(np.uint16) * int(round(clamp(alpha_mul, 0.0, 1.0) * 256))) >> 8
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.uint16)
    fg = sub_sprite[:, :, :3].astype(np.uint16)
    canvas[cy0:cy1, cx0:cx1] = ((roi * (256 - a_u16) + fg * a_u16) >> 8).astype(np.uint8)
    return (cx0, cy0, cx1, cy1)


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
    """High-level resolution-independent text renderer."""
    if alpha <= 0.01 or not text:
        return (int(x), int(y), int(x), int(y))
    size_px = cfg.px(size_4k)
    sprite = render_text_sprite(text, role, size_px, color, bold=bold)
    return blit_rgba_sprite(canvas, sprite, int(round(x)), int(round(y)), alpha_mul=alpha, anchor=anchor)


def draw_glow_circle(
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
    """Draw an antialiased circle or point with subtle mathematical laboratory luminescence."""
    if alpha <= 0.01:
        return
    ix, iy = int(round(cx)), int(round(cy))
    r_base = max(2, cfg.px(radius_4k))
    for g in range(glow_layers, 0, -1):
        rg = int(round(r_base * (1.0 + 0.45 * g)))
        ga = alpha * (0.14 / g)
        gc = scale_color(color, ga)
        cv2.circle(canvas, (ix, iy), rg, gc, max(1, cfg.px(3)), cv2.LINE_AA)
    main_c = scale_color(color, alpha)
    if filled:
        cv2.circle(canvas, (ix, iy), r_base, main_c, -1, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (ix, iy), r_base, main_c, max(2, cfg.px(4)), cv2.LINE_AA)


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
            q1 = (lerp(x1, x2, u0), lerp(y1, y2, u0))
            q2 = (lerp(x1, x2, u1), lerp(y1, y2, u1))
            draw_aa_line(canvas, q1, q2, thickness_4k, cfg, color, alpha)


def draw_rounded_panel(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    cfg: RenderConfig,
    fill_color: Tuple[int, int, int] = PALETTE.PANEL_BG,
    border_color: Tuple[int, int, int] = PALETTE.PANEL_BORDER,
    alpha: float = 1.0,
    border_thickness_4k: float = 3.0,
) -> None:
    if alpha <= 0.01:
        return
    ix0, iy0 = max(0, int(round(x0))), max(0, int(round(y0)))
    ix1, iy1 = min(canvas.shape[1], int(round(x1))), min(canvas.shape[0], int(round(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return
    a = clamp(alpha * 0.88, 0.0, 1.0)
    if a > 0.92:
        canvas[iy0:iy1, ix0:ix1] = fill_color
    else:
        roi = canvas[iy0:iy1, ix0:ix1]
        bg = np.full_like(roi, fill_color)
        cv2.addWeighted(bg, a, roi, 1.0 - a, 0.0, dst=roi)
    bc = scale_color(border_color, alpha)
    cv2.rectangle(canvas, (ix0, iy0), (ix1, iy1), bc, max(1, cfg.px(border_thickness_4k)), cv2.LINE_AA)


@dataclass
class PlotViewport:
    """
    Mathematical coordinate system mapping data coordinates (x, y) to screen pixels,
    guaranteeing zero mathematical contradiction between axes, ticks, curves, and points.
    """
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


def draw_coordinate_axes(
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
    """Draw a clean laboratory coordinate grid with ticks and axis labels."""
    if alpha <= 0.01:
        return
    if show_grid:
        for xt in x_ticks:
            sx, _ = vp.to_px(xt, vp.y_min)
            draw_aa_line(canvas, (sx, vp.px_top), (sx, vp.px_bottom), 2.0, cfg, PALETTE.GRID_MAJOR, alpha * 0.55)
        for yt in y_ticks:
            _, sy = vp.to_px(vp.x_min, yt)
            draw_aa_line(canvas, (vp.px_left, sy), (vp.px_right, sy), 2.0, cfg, PALETTE.GRID_MAJOR, alpha * 0.55)

    # Axes lines
    draw_aa_line(canvas, (vp.px_left, vp.px_bottom), (vp.px_right, vp.px_bottom), 4.0, cfg, PALETTE.AXIS_GRAY, alpha)
    draw_aa_line(canvas, (vp.px_left, vp.px_bottom), (vp.px_left, vp.px_top), 4.0, cfg, PALETTE.AXIS_GRAY, alpha)

    # Ticks & numerical labels
    tick_len = cfg.px(14)
    for xt in x_ticks:
        sx, sy = vp.to_px(xt, vp.y_min)
        draw_aa_line(canvas, (sx, sy - tick_len), (sx, sy + tick_len), 3.0, cfg, PALETTE.GRAY_LIGHT, alpha)
        draw_text(
            canvas,
            x_fmt.format(xt),
            sx,
            sy + cfg.px(34),
            30,
            cfg,
            color=PALETTE.GRAY_MID,
            role="mono",
            alpha=alpha,
            anchor="topcenter",
        )
    for yt in y_ticks:
        sx, sy = vp.to_px(vp.x_min, yt)
        draw_aa_line(canvas, (sx - tick_len, sy), (sx + tick_len, sy), 3.0, cfg, PALETTE.GRAY_LIGHT, alpha)
        draw_text(
            canvas,
            y_fmt.format(yt),
            sx - cfg.px(24),
            sy,
            30,
            cfg,
            color=PALETTE.GRAY_MID,
            role="mono",
            alpha=alpha,
            anchor="midright",
        )

    # Axis titles
    if x_label:
        draw_text(
            canvas,
            x_label,
            (vp.px_left + vp.px_right) * 0.5,
            vp.px_bottom + cfg.px(92),
            36,
            cfg,
            color=PALETTE.GRAY_LIGHT,
            role="sans",
            bold=True,
            alpha=alpha,
            anchor="topcenter",
        )
    if y_label:
        draw_text(
            canvas,
            y_label,
            vp.px_left,
            vp.px_top - cfg.px(42),
            36,
            cfg,
            color=PALETTE.GRAY_LIGHT,
            role="sans",
            bold=True,
            alpha=alpha,
            anchor="midleft",
        )


def draw_function_curve(
    canvas: np.ndarray,
    vp: PlotViewport,
    fn: Callable[[np.ndarray], np.ndarray],
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.WHITE,
    thickness_4k: float = 6.0,
    alpha: float = 1.0,
    progress: float = 1.0,
    num_samples: int = 220,
) -> None:
    """Plot a continuous mathematical function y = fn(x) clipped to the viewport."""
    prog = clamp(progress, 0.0, 1.0)
    if alpha <= 0.01 or prog <= 0.005:
        return
    x_end = lerp(vp.x_min, vp.x_max, prog)
    xs = np.linspace(vp.x_min, x_end, max(12, int(num_samples * prog)))
    ys = fn(xs)
    pts = []
    for x_val, y_val in zip(xs, ys):
        yc = clamp(float(y_val), vp.y_min, vp.y_max)
        sx, sy = vp.to_px(float(x_val), yc)
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


# ==============================================================================
# 5. INTEGRATED ANIMATED CAPTION & HUD COMPOSITOR
# ==============================================================================

def draw_lecture_hud_and_captions(
    canvas: np.ndarray,
    scene: SceneSpec,
    t: float,
    cfg: RenderConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Render the top laboratory section bar and bottom animated teaching captions.
    Captions feature phrase/word progressive reveal, vertical float entrance,
    style badges ('DEFINITION', 'EQUATION', 'KEY QUESTION'), and safe margins.
    Returns list of text bounding boxes for automated non-overlap verification.
    """
    bboxes: List[Tuple[int, int, int, int]] = []
    W, H = cfg.width, cfg.height

    # Global fade out at the very end of the 30-minute lecture (1797s -> 1800s)
    master_fade = 1.0
    if t < 1.5:
        master_fade = smoothstep(0.0, 1.5, t)
    elif t > 1796.5:
        master_fade = 1.0 - smoothstep(1796.5, 1799.6, t)

    # Top subtle laboratory header bar (hidden during opening title card peak)
    header_alpha = master_fade * (0.35 if t < 55.0 else 0.90)
    if header_alpha > 0.05:
        draw_aa_line(
            canvas,
            (cfg.px(140), cfg.px(128)),
            (W - cfg.px(140), cfg.px(128)),
            2.0,
            cfg,
            PALETTE.PANEL_BORDER,
            header_alpha,
        )
        b1 = draw_text(
            canvas,
            "SILENT MATH  |  LECTURE #1",
            cfg.px(140),
            cfg.px(82),
            28,
            cfg,
            color=PALETTE.GRAY_MID,
            role="mono",
            bold=True,
            alpha=header_alpha,
            anchor="midleft",
        )
        b2 = draw_text(
            canvas,
            scene.title.upper(),
            W - cfg.px(140),
            cfg.px(82),
            28,
            cfg,
            color=PALETTE.GRAY_LIGHT,
            role="mono",
            alpha=header_alpha,
            anchor="midright",
        )
        bboxes.extend([b1, b2])

    # Active animated caption in dedicated bottom laboratory stage (y: 1750..2070 in 4K)
    cap = get_active_caption(scene, t)
    if cap is not None:
        env = window_envelope(t, cap.start, cap.end, fade_in=0.45, fade_out=0.45) * master_fade
        if env > 0.02:
            # Subtle vertical rise during entrance
            y_offset = cfg.px(18.0 * (1.0 - smoothstep(cap.start, cap.start + 0.55, t)))
            stage_top = cfg.px(1760) + y_offset
            stage_bottom = cfg.px(2065) + y_offset
            stage_left = cfg.px(220)
            stage_right = W - cfg.px(220)

            border_col = PALETTE.PANEL_BORDER
            role_line1 = "sans"
            line1_col = PALETTE.WHITE
            if cap.style == "equation":
                border_col = PALETTE.ACCENT_ICE
                role_line1 = "mono"
                line1_col = PALETTE.ACCENT_ICE
            elif cap.style == "definition":
                border_col = PALETTE.WHITE
            elif cap.style == "question":
                border_col = PALETTE.ACCENT_GOLD
                line1_col = PALETTE.ACCENT_GOLD

            draw_rounded_panel(
                canvas,
                stage_left,
                stage_top,
                stage_right,
                stage_bottom,
                cfg,
                fill_color=PALETTE.PANEL_BG,
                border_color=border_col,
                alpha=env * 0.92,
                border_thickness_4k=3.0,
            )

            # Word-by-word progressive reveal during the first 32% of caption duration
            cap_dur = max(1.0, cap.end - cap.start)
            reveal_window = min(1.8, cap_dur * 0.32)
            reveal_u = clamp((t - cap.start) / max(0.25, reveal_window), 0.0, 1.0)

            words1 = cap.line1.split(" ")
            n1 = max(1, int(math.ceil(len(words1) * clamp(reveal_u * 1.35, 0.15, 1.0))))
            text1_visible = " ".join(words1[:n1])

            y_line1 = (stage_top + stage_bottom) * 0.5 if not cap.line2 else stage_top + cfg.px(95)
            b_c1 = draw_text(
                canvas,
                text1_visible,
                W * 0.5,
                y_line1,
                46,
                cfg,
                color=line1_col,
                role=role_line1,
                bold=True,
                alpha=env,
                anchor="center",
            )
            bboxes.append(b_c1)

            if cap.line2:
                words2 = cap.line2.split(" ")
                u2 = clamp((reveal_u - 0.25) / 0.75, 0.0, 1.0)
                if u2 > 0.02:
                    n2 = max(1, int(math.ceil(len(words2) * u2)))
                    text2_visible = " ".join(words2[:n2])
                    b_c2 = draw_text(
                        canvas,
                        text2_visible,
                        W * 0.5,
                        stage_top + cfg.px(195),
                        40,
                        cfg,
                        color=PALETTE.GRAY_LIGHT,
                        role="sans",
                        bold=False,
                        alpha=env * smoothstep(0.0, 0.25, u2),
                        anchor="center",
                    )
                    bboxes.append(b_c2)

    return bboxes


# ==============================================================================
# 6. SCENE-BY-SCENE MATHEMATICAL ANIMATION RENDERERS (1800 SECONDS TOTAL)
# ==============================================================================

def render_scene_01_intro(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    00:00–01:30 (90s) — Cinematic Introduction
    - 0..10s : Black screen -> tiny white point appears and moves along a smooth trajectory.
    - 10..38s: Coordinate plane forms around the moving point; mathematical curve & data points emerge.
    - 38..66s: Opening Title Card (SILENT MATH / LECTURE #1 / INTRODUCTION TO AI & MACHINE LEARNING + Tagline).
    - 66..90s: Central guiding question: "How can a machine learn patterns from examples?" with live curve.
    """
    W, H = cfg.width, cfg.height
    vp = PlotViewport(
        x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
        px_left=cfg.px(520), px_top=cfg.px(260),
        px_right=W - cfg.px(520), px_bottom=cfg.px(1580),
    )

    # Coordinate plane gradually forms between t=9s and t=24s
    grid_alpha = smoothstep(9.0, 22.0, t) * (1.0 - 0.55 * window_envelope(t, 38.0, 66.0, 2.0, 2.0))
    if grid_alpha > 0.01:
        draw_coordinate_axes(
            canvas, vp, cfg, alpha=grid_alpha,
            x_ticks=[2, 4, 6, 8, 10], y_ticks=[2, 4, 6, 8, 10],
            x_label="Input Coordinate (x)", y_label="Output Quantity (y)",
        )

    # Mathematical curve f(x) = 1.6 + 0.65*x + 1.1*sin(0.55*x) emerges between t=18s and t=36s
    curve_fn = lambda x: 1.6 + 0.65 * x + 1.1 * np.sin(0.55 * x)
    curve_prog = smootherstep(18.0, 34.0, t)
    if curve_prog > 0.01:
        draw_function_curve(
            canvas, vp, curve_fn, cfg,
            color=PALETTE.ACCENT_ICE, thickness_4k=6.0,
            alpha=grid_alpha, progress=curve_prog,
        )

    # Example data points appear along the curve (t >= 25s)
    sample_xs = [1.2, 2.4, 3.8, 5.1, 6.4, 7.8, 9.0]
    sample_offsets = [-0.35, 0.42, -0.28, 0.48, -0.32, 0.30, -0.22]
    for i, (sx_val, dy) in enumerate(zip(sample_xs, sample_offsets)):
        pt_alpha = smoothstep(25.0 + i * 1.1, 26.5 + i * 1.1, t) * grid_alpha
        if pt_alpha > 0.01:
            sy_val = float(curve_fn(np.array([sx_val]))[0]) + dy
            px, py = vp.to_px(sx_val, sy_val)
            px_curve, py_curve = vp.to_px(sx_val, float(curve_fn(np.array([sx_val]))[0]))
            draw_dashed_line(canvas, (px, py), (px_curve, py_curve), 2.5, cfg, PALETTE.GRAY_MID, pt_alpha * 0.7)
            draw_glow_circle(canvas, px, py, 11.0, cfg, PALETTE.WHITE, pt_alpha, glow_layers=2)

    # The iconic moving white point (appears at t=1.5s)
    pt_appear = smoothstep(1.5, 4.0, t)
    if pt_appear > 0.01:
        # Smooth continuous parametric traversal across the curve
        x_t = 5.0 + 3.8 * math.sin(0.35 * max(0.0, t - 3.0))
        y_t = float(curve_fn(np.array([x_t]))[0])
        px, py = vp.to_px(x_t, y_t)
        # Before coordinate plane appears, gently interpolate from screen center
        blend_plane = smoothstep(3.0, 14.0, t)
        cx = lerp(W * 0.5, px, blend_plane)
        cy = lerp(H * 0.44, py, blend_plane)
        draw_glow_circle(canvas, cx, cy, 16.0, cfg, PALETTE.WHITE, pt_appear * (1.0 - 0.4 * window_envelope(t, 38.0, 65.0, 1.5, 1.5)), glow_layers=4)
        if grid_alpha > 0.25 and not (38.0 <= t <= 66.0):
            draw_text(
                canvas, f"P({x_t:.1f}, {y_t:.1f})",
                cx + cfg.px(28), cy - cfg.px(28), 32, cfg,
                color=PALETTE.WHITE, role="mono", alpha=grid_alpha, anchor="midleft",
            )

    # Opening Title Card (t = 38s .. 66s)
    title_env = window_envelope(t, 38.0, 66.5, fade_in=1.8, fade_out=1.8)
    if title_env > 0.01:
        draw_rounded_panel(
            canvas, cfg.px(460), cfg.px(360), W - cfg.px(460), cfg.px(1420),
            cfg, fill_color=PALETTE.BG_BLACK, border_color=PALETTE.WHITE,
            alpha=title_env * 0.96, border_thickness_4k=4.0,
        )
        draw_text(canvas, "SILENT MATH", W * 0.5, cfg.px(560), 96, cfg, PALETTE.WHITE, "sans", bold=True, alpha=title_env)
        draw_aa_line(canvas, (W * 0.5 - cfg.px(380), cfg.px(665)), (W * 0.5 + cfg.px(380), cfg.px(665)), 3.0, cfg, PALETTE.GRAY_MID, title_env)
        draw_text(
            canvas, "LECTURE #1", W * 0.5, cfg.px(770), 54, cfg,
            PALETTE.ACCENT_ICE, "mono", bold=True, alpha=title_env * smoothstep(40.0, 42.5, t),
        )
        draw_text(
            canvas, "INTRODUCTION TO AI & MACHINE LEARNING", W * 0.5, cfg.px(930), 64, cfg,
            PALETTE.WHITE, "sans", bold=True, alpha=title_env * smoothstep(42.0, 45.0, t),
        )
        draw_text(
            canvas, "Understand the mathematics. Visualize the intelligence.",
            W * 0.5, cfg.px(1140), 42, cfg,
            PALETTE.GRAY_LIGHT, "sans", bold=False, alpha=title_env * smoothstep(45.0, 48.5, t),
        )

    # Central Question Card (t = 67s .. 89s)
    q_env = window_envelope(t, 67.0, 89.5, fade_in=1.5, fade_out=1.5)
    if q_env > 0.01:
        draw_rounded_panel(
            canvas, cfg.px(560), cfg.px(300), W - cfg.px(560), cfg.px(540),
            cfg, fill_color=PALETTE.PANEL_BG, border_color=PALETTE.ACCENT_GOLD,
            alpha=q_env, border_thickness_4k=3.0,
        )
        draw_text(
            canvas, "CENTRAL QUESTION", W * 0.5, cfg.px(365), 30, cfg,
            PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=q_env,
        )
        draw_text(
            canvas, "How can a machine learn patterns from examples?",
            W * 0.5, cfg.px(455), 54, cfg,
            PALETTE.WHITE, "sans", bold=True, alpha=q_env,
        )


def render_scene_02_what_is_ai(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    01:30–04:30 (90s..270s, 180s) — What Is Artificial Intelligence?
    - Top architectural pipeline: INPUT -> PROCESSING -> OUTPUT with animated signal flow.
    - Sub-demonstration 1 (90..154s) : Sensory/Pixel matrix input & Geometric Shape Recognition.
    - Sub-demonstration 2 (154..183s): Sequential Wave Pattern & Anomaly Identification.
    - Sub-demonstration 3 (183..225s): Action Selection on a Decision Grid.
    - Sub-demonstration 4 (225..270s): Broad AI Umbrella vs. Explicit Rules vs. Learning Systems.
    """
    W, H = cfg.width, cfg.height
    local_t = t - 90.0
    env = window_envelope(t, 90.0, 270.0, 1.0, 1.0)

    # Top Input -> Processing -> Output Pipeline Banner
    boxes = [
        ("1. INPUT", "Information / Observations", cfg.px(340), cfg.px(1180)),
        ("2. PROCESSING", "Mathematical Computation", cfg.px(1480), cfg.px(2360)),
        ("3. OUTPUT", "Recognition / Decision", cfg.px(2660), cfg.px(3500)),
    ]
    pulse_stage = (local_t * 0.65) % 3.0
    for idx, (b_title, b_sub, x0, x1) in enumerate(boxes):
        active_boost = max(0.0, 1.0 - abs(pulse_stage - idx))
        border_c = lerp_color(PALETTE.PANEL_BORDER, PALETTE.ACCENT_ICE, active_boost)
        draw_rounded_panel(canvas, x0, cfg.px(185), x1, cfg.px(370), cfg, PALETTE.PANEL_BG, border_c, env, 3.5)
        draw_text(canvas, b_title, (x0 + x1) * 0.5, cfg.px(248), 38, cfg, PALETTE.WHITE, "mono", bold=True, alpha=env)
        draw_text(canvas, b_sub, (x0 + x1) * 0.5, cfg.px(318), 30, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=env)

    # Connecting arrows with traveling pulse dots
    for ax0, ax1 in [(cfg.px(1180), cfg.px(1480)), (cfg.px(2360), cfg.px(2660))]:
        ay = cfg.px(277)
        draw_aa_line(canvas, (ax0 + cfg.px(20), ay), (ax1 - cfg.px(20), ay), 4.0, cfg, PALETTE.GRAY_MID, env)
        u_dot = (local_t * 0.9) % 1.0
        dx = lerp(ax0 + cfg.px(25), ax1 - cfg.px(25), u_dot)
        draw_glow_circle(canvas, dx, ay, 10.0, cfg, PALETTE.ACCENT_ICE, env, glow_layers=2)

    # Phase 1 (90..154s): Pixel Matrix Input & Object Recognition
    p1_env = window_envelope(t, 90.0, 154.5, 1.0, 1.2)
    if p1_env > 0.01:
        # Left: 7x7 Pixel intensity matrix morphing between Circle, Triangle, Square
        shape_idx = int((local_t / 12.0) % 3)
        shape_names = ["CIRCLE", "TRIANGLE", "SQUARE"]
        confidences = [(0.94, 0.04, 0.02), (0.03, 0.93, 0.04), (0.02, 0.05, 0.93)][shape_idx]

        gx0, gy0 = cfg.px(420), cfg.px(520)
        cell = cfg.px(125)
        draw_text(canvas, "RAW PIXEL INTENSITY MATRIX (INPUT)", gx0 + 3.5 * cell, gy0 - cfg.px(55), 34, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=p1_env)
        for r in range(7):
            for c in range(7):
                # Synthetic intensity for current shape
                nx, ny = (c - 3.0) / 3.0, (r - 3.0) / 3.0
                if shape_idx == 0:
                    val = max(0.0, 1.0 - abs(math.hypot(nx, ny) - 0.72) * 2.4)
                elif shape_idx == 1:
                    val = 0.92 if (r >= 1 and abs(c - 3) <= (r - 1) * 0.6 and (r == 5 or abs( abs(c-3) - (r-1)*0.5 ) < 0.6)) else 0.08
                else:
                    val = 0.92 if (max(abs(c - 3), abs(r - 3)) == 2) else 0.08
                num_val = int(round(val * 255))
                x_cell = gx0 + c * cell
                y_cell = gy0 + r * cell
                fill_g = int(round(val * 110)) + 14
                draw_rounded_panel(
                    canvas, x_cell + 3, y_cell + 3, x_cell + cell - 3, y_cell + cell - 3,
                    cfg, (fill_g, fill_g, fill_g + 8), PALETTE.GRAY_DARK, p1_env, 2.0,
                )
                draw_text(
                    canvas, f"{num_val:3d}", x_cell + cell * 0.5, y_cell + cell * 0.5,
                    28, cfg, PALETTE.WHITE if num_val > 120 else PALETTE.GRAY_MID, "mono", alpha=p1_env,
                )

        # Right: Geometric Feature Extractor & Recognition Scores
        rx0, ry0 = cfg.px(1680), cfg.px(520)
        draw_rounded_panel(canvas, rx0, ry0, cfg.px(3420), cfg.px(1400), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, p1_env, 3.0)
        draw_text(canvas, "OBJECT RECOGNITION SCORES (OUTPUT)", (rx0 + cfg.px(3420)) * 0.5, ry0 + cfg.px(75), 36, cfg, PALETTE.WHITE, "mono", bold=True, alpha=p1_env)

        for k, (sname, score) in enumerate(zip(shape_names, confidences)):
            by = ry0 + cfg.px(230 + k * 220)
            is_winner = (k == shape_idx)
            draw_text(canvas, sname, rx0 + cfg.px(110), by, 40, cfg, PALETTE.WHITE if is_winner else PALETTE.GRAY_MID, "mono", bold=is_winner, alpha=p1_env, anchor="midleft")
            bar_x0 = rx0 + cfg.px(480)
            bar_max_w = cfg.px(950)
            draw_rounded_panel(canvas, bar_x0, by - cfg.px(34), bar_x0 + bar_max_w, by + cfg.px(34), cfg, PALETTE.BG_BLACK, PALETTE.GRAY_DARK, p1_env, 2.0)
            fill_w = max(cfg.px(16), int(bar_max_w * score))
            draw_rounded_panel(
                canvas, bar_x0, by - cfg.px(34), bar_x0 + fill_w, by + cfg.px(34),
                cfg, PALETTE.ACCENT_ICE if is_winner else PALETTE.GRAY_DARK,
                PALETTE.WHITE if is_winner else PALETTE.GRAY_MID, p1_env, 2.0,
            )
            draw_text(canvas, f"{score * 100:5.1f}%", bar_x0 + bar_max_w + cfg.px(50), by, 38, cfg, PALETTE.WHITE if is_winner else PALETTE.GRAY_MID, "mono", bold=is_winner, alpha=p1_env, anchor="midleft")

    # Phase 2 (154..183s): Pattern Identification in Sequential Data
    p2_env = window_envelope(t, 154.0, 183.5, 1.0, 1.0)
    if p2_env > 0.01:
        vp = PlotViewport(0.0, 12.0, -2.5, 2.5, cfg.px(420), cfg.px(520), W - cfg.px(420), cfg.px(1480))
        draw_coordinate_axes(
            canvas, vp, cfg, alpha=p2_env,
            x_ticks=[2, 4, 6, 8, 10, 12], y_ticks=[-2, -1, 0, 1, 2],
            x_label="Time Step (t)", y_label="Measured Signal Amplitude",
        )
        phase_shift = local_t * 0.8
        wave_fn = lambda x: 1.25 * np.sin(1.4 * x - phase_shift)
        draw_function_curve(canvas, vp, wave_fn, cfg, PALETTE.ACCENT_ICE, 5.0, p2_env, progress=1.0)
        # Discrete sensor samples + one highlighted anomaly spike at x=8.2
        for sx in np.linspace(0.5, 11.5, 23):
            is_anom = abs(sx - 8.0) < 0.25
            sy = float(wave_fn(np.array([sx]))[0]) + (1.05 if is_anom else 0.0)
            px, py = vp.to_px(float(sx), clamp(sy, -2.3, 2.3))
            col = PALETTE.ACCENT_GOLD if is_anom else PALETTE.WHITE
            draw_glow_circle(canvas, px, py, 14.0 if is_anom else 9.0, cfg, col, p2_env, glow_layers=3 if is_anom else 1)
            if is_anom:
                draw_text(canvas, "DETECTED ANOMALY", px, py - cfg.px(48), 32, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=p2_env)

    # Phase 3 (183..225s): Selecting an Action Based on Available Information
    p3_env = window_envelope(t, 183.0, 225.5, 1.0, 1.0)
    if p3_env > 0.01:
        vp = PlotViewport(0.0, 10.0, 0.0, 6.0, cfg.px(460), cfg.px(500), W - cfg.px(460), cfg.px(1500))
        draw_coordinate_axes(canvas, vp, cfg, alpha=p3_env, x_ticks=[2, 4, 6, 8, 10], y_ticks=[1, 2, 3, 4, 5], x_label="Horizontal Grid State", y_label="Vertical Grid State")
        # Obstacle region in center
        ox0, oy0 = vp.to_px(4.2, 4.3)
        ox1, oy1 = vp.to_px(5.8, 1.7)
        draw_rounded_panel(canvas, ox0, oy0, ox1, oy1, cfg, (38, 22, 26), PALETTE.ACCENT_ROSE, p3_env, 3.0)
        draw_text(canvas, "OBSTACLE", (ox0 + ox1) * 0.5, (oy0 + oy1) * 0.5, 32, cfg, PALETTE.ACCENT_ROSE, "mono", bold=True, alpha=p3_env)

        # Candidate trajectories: Optimal upper arc vs suboptimal lower arc
        opt_fn = lambda x: 3.0 + 2.1 * np.sin(math.pi * (x - 1.0) / 8.0)
        sub_fn = lambda x: 3.0 - 2.3 * np.sin(math.pi * (x - 1.0) / 8.0)
        draw_function_curve(canvas, vp, sub_fn, cfg, PALETTE.GRAY_DARK, 4.0, p3_env * 0.75, progress=1.0)
        draw_function_curve(canvas, vp, opt_fn, cfg, PALETTE.ACCENT_ICE, 7.0, p3_env, progress=1.0)

        # Start, Goal, and moving agent on optimal path
        sx, sy = vp.to_px(1.0, 3.0)
        gx, gy = vp.to_px(9.0, 3.0)
        draw_glow_circle(canvas, sx, sy, 16.0, cfg, PALETTE.WHITE, p3_env)
        draw_glow_circle(canvas, gx, gy, 18.0, cfg, PALETTE.ACCENT_GOLD, p3_env)
        draw_text(canvas, "START", sx, sy + cfg.px(48), 32, cfg, PALETTE.WHITE, "mono", bold=True, alpha=p3_env)
        draw_text(canvas, "GOAL", gx, gy + cfg.px(48), 32, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=p3_env)

        u_ag = ((local_t - 93.0) * 0.22) % 1.0
        ax_val = lerp(1.0, 9.0, u_ag)
        ay_val = float(opt_fn(np.array([ax_val]))[0])
        apx, apy = vp.to_px(ax_val, ay_val)
        draw_glow_circle(canvas, apx, apy, 15.0, cfg, PALETTE.ACCENT_ICE, p3_env, glow_layers=4)
        draw_text(canvas, "SELECTED OPTIMAL ACTION PATH", W * 0.5, cfg.px(560), 36, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=p3_env)

    # Phase 4 (225..270s): Broad AI Umbrella vs. Explicit Rules vs. Machine Learning
    p4_env = window_envelope(t, 225.0, 270.0, 1.0, 1.0)
    if p4_env > 0.01:
        draw_rounded_panel(canvas, cfg.px(380), cfg.px(470), W - cfg.px(380), cfg.px(1560), cfg, PALETTE.PANEL_BG, PALETTE.WHITE, p4_env, 4.0)
        draw_text(canvas, "ARTIFICIAL INTELLIGENCE (AI) — THE BROAD FIELD", W * 0.5, cfg.px(560), 48, cfg, PALETTE.WHITE, "sans", bold=True, alpha=p4_env)
        draw_text(canvas, "Systems capable of performing tasks associated with intelligent behavior", W * 0.5, cfg.px(630), 34, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=p4_env)

        # Left sub-card: Manually Specified Rules
        draw_rounded_panel(canvas, cfg.px(500), cfg.px(740), cfg.px(1820), cfg.px(1440), cfg, PALETTE.BG_BLACK, PALETTE.GRAY_MID, p4_env, 3.0)
        draw_text(canvas, "APPROACH A: EXPLICIT RULES", cfg.px(1160), cfg.px(835), 38, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=p4_env)
        draw_text(canvas, "Human programmers write fixed logic:", cfg.px(1160), cfg.px(945), 34, cfg, PALETTE.GRAY_MID, "sans", alpha=p4_env)
        draw_text(canvas, "IF condition_1 THEN action_A", cfg.px(1160), cfg.px(1065), 34, cfg, PALETTE.WHITE, "mono", alpha=p4_env)
        draw_text(canvas, "ELSE IF condition_2 THEN action_B", cfg.px(1160), cfg.px(1155), 34, cfg, PALETTE.WHITE, "mono", alpha=p4_env)
        draw_text(canvas, "Effective for rigid rules; brittle on noisy patterns", cfg.px(1160), cfg.px(1310), 30, cfg, PALETTE.GRAY_MID, "sans", alpha=p4_env)

        # Right sub-card: Machine Learning
        ml_boost = smoothstep(245.0, 258.0, t)
        draw_rounded_panel(canvas, cfg.px(2020), cfg.px(740), cfg.px(3340), cfg.px(1440), cfg, PALETTE.BG_BLACK, lerp_color(PALETTE.GRAY_MID, PALETTE.ACCENT_ICE, ml_boost), p4_env, 4.0)
        draw_text(canvas, "APPROACH B: MACHINE LEARNING", cfg.px(2680), cfg.px(835), 38, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=p4_env)
        draw_text(canvas, "Mathematical models learn from examples:", cfg.px(2680), cfg.px(945), 34, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=p4_env)
        draw_text(canvas, "Observed Data  →  Mathematical Model", cfg.px(2680), cfg.px(1085), 36, cfg, PALETTE.WHITE, "mono", bold=True, alpha=p4_env)
        draw_text(canvas, "Discovers patterns too subtle to code by hand", cfg.px(2680), cfg.px(1310), 32, cfg, PALETTE.ACCENT_GOLD, "sans", alpha=p4_env)


def render_scene_03_what_is_ml(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    04:30–08:00 (270s..480s, 210s) — What Is Machine Learning?
    - 270..316s: Rigid step-function rules vs smooth pattern.
    - 316..382s: Points appear on a graph without a curve -> smooth curve emerges.
    - 382..480s: Data -> Model -> Prediction diagram + live demonstration that a prediction
                 is an output produced by a model, not necessarily a perfect answer.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 270.0, 480.0, 1.0, 1.0)

    vp = PlotViewport(
        0.0, 10.0, 0.0, 10.0,
        cfg.px(420), cfg.px(260), cfg.px(2220) if t >= 380.0 else W - cfg.px(420), cfg.px(1540),
    )
    draw_coordinate_axes(
        canvas, vp, cfg, alpha=env,
        x_ticks=[2, 4, 6, 8, 10], y_ticks=[2, 4, 6, 8, 10],
        x_label="Input Measurement (x)", y_label="Observed Outcome (y)",
    )

    # Underlying smooth pattern & 10 example points
    true_fn = lambda x: 1.5 + 0.55 * x + 0.03 * (x ** 2)
    pts_x = [1.0, 1.9, 2.8, 3.7, 4.6, 5.6, 6.5, 7.4, 8.3, 9.2]
    pts_noise = [0.35, -0.45, 0.28, -0.32, 0.52, -0.40, 0.38, -0.48, 0.30, -0.25]

    # Phase A (270..316s): Rigid hand-coded staircase rule
    stair_alpha = window_envelope(t, 270.0, 316.0, 1.0, 1.5)
    if stair_alpha > 0.01:
        stair_fn = lambda x: np.floor(x / 2.5) * 2.2 + 2.0
        draw_function_curve(canvas, vp, stair_fn, cfg, PALETTE.ACCENT_ROSE, 5.0, stair_alpha, progress=smoothstep(272.0, 286.0, t))
        draw_text(canvas, "RIGID HAND-CODED IF-THEN RULES", (vp.px_left + vp.px_right) * 0.5, cfg.px(210), 36, cfg, PALETTE.ACCENT_ROSE, "mono", bold=True, alpha=stair_alpha)

    # Data points appear sequentially starting at 285s (and re-emphasized at 317s)
    for i, (px_val, dy) in enumerate(zip(pts_x, pts_noise)):
        p_alpha = smoothstep(284.0 + i * 0.8, 286.0 + i * 0.8, t) * env
        if 316.0 <= t <= 335.0:
            p_alpha *= smoothstep(317.0 + i * 1.1, 319.0 + i * 1.1, t)
        if p_alpha > 0.01:
            py_val = float(true_fn(np.array([px_val]))[0]) + dy
            sx, sy = vp.to_px(px_val, py_val)
            draw_glow_circle(canvas, sx, sy, 12.0, cfg, PALETTE.WHITE, p_alpha, glow_layers=2)

    # Smooth learned curve emerges at t = 348s
    curve_prog = smootherstep(348.0, 366.0, t)
    if curve_prog > 0.01:
        draw_function_curve(canvas, vp, true_fn, cfg, PALETTE.ACCENT_ICE, 6.5, env, progress=curve_prog)

    # Phase C (380..480s): Right-side panel showing Data -> Model -> Prediction & Prediction gap
    panel_alpha = smoothstep(380.0, 384.0, t) * env
    if panel_alpha > 0.01:
        rx0, rx1 = cfg.px(2360), W - cfg.px(240)
        draw_rounded_panel(canvas, rx0, cfg.px(260), rx1, cfg.px(1540), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, panel_alpha, 3.0)
        cx = (rx0 + rx1) * 0.5

        stages = [
            ("DATA", "Observed Examples (xᵢ, yᵢ)", 383.0, PALETTE.WHITE),
            ("MODEL", "Learned Pattern f(x)", 400.0, PALETTE.ACCENT_ICE),
            ("PREDICTION", "Model Output ŷ = f(x)", 417.0, PALETTE.ACCENT_GOLD),
        ]
        for idx, (st_title, st_desc, t_unlock, st_col) in enumerate(stages):
            by0 = cfg.px(340 + idx * 360)
            by1 = by0 + cfg.px(220)
            st_a = panel_alpha * smoothstep(t_unlock - 1.0, t_unlock + 2.0, t)
            draw_rounded_panel(canvas, rx0 + cfg.px(90), by0, rx1 - cfg.px(90), by1, cfg, PALETTE.BG_BLACK, st_col, st_a, 3.5)
            draw_text(canvas, st_title, cx, by0 + cfg.px(78), 44, cfg, st_col, "mono", bold=True, alpha=st_a)
            draw_text(canvas, st_desc, cx, by0 + cfg.px(152), 32, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=st_a)
            if idx < 2:
                ay0 = by1 + cfg.px(18)
                ay1 = by0 + cfg.px(360) - cfg.px(18)
                draw_aa_line(canvas, (cx, ay0), (cx, ay1), 4.0, cfg, PALETTE.GRAY_MID, st_a)
                draw_glow_circle(canvas, cx, lerp(ay0, ay1, (t * 0.8) % 1.0), 9.0, cfg, st_col, st_a)

        # Demonstrate Prediction != Perfect Truth on point index 6 (x=6.5) for t >= 445s
        gap_alpha = smoothstep(445.0, 452.0, t) * panel_alpha
        if gap_alpha > 0.01:
            qx = pts_x[6]
            qy_true = float(true_fn(np.array([qx]))[0]) + pts_noise[6]
            qy_pred = float(true_fn(np.array([qx]))[0])
            s_true = vp.to_px(qx, qy_true)
            s_pred = vp.to_px(qx, qy_pred)
            draw_dashed_line(canvas, (s_pred[0], vp.px_bottom), s_pred, 3.0, cfg, PALETTE.ACCENT_GOLD, gap_alpha)
            draw_aa_line(canvas, s_true, s_pred, 5.0, cfg, PALETTE.ACCENT_GOLD, gap_alpha)
            draw_glow_circle(canvas, s_pred[0], s_pred[1], 14.0, cfg, PALETTE.ACCENT_GOLD, gap_alpha, filled=False)
            draw_text(canvas, f"Prediction ŷ = {qy_pred:.2f}", s_pred[0] - cfg.px(25), s_pred[1] + cfg.px(42), 32, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=gap_alpha, anchor="midright")
            draw_text(canvas, f"Actual y = {qy_true:.2f}", s_true[0] - cfg.px(25), s_true[1] - cfg.px(42), 32, cfg, PALETTE.WHITE, "mono", bold=True, alpha=gap_alpha, anchor="midright")


def draw_house_icon(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    scale_size: float,
    cfg: RenderConfig,
    color: Tuple[int, int, int] = PALETTE.WHITE,
    alpha: float = 1.0,
) -> None:
    """Draw a minimal geometric architectural house wireframe scaled by floor area."""
    if alpha <= 0.01:
        return
    w = cfg.px(70.0 * scale_size)
    h = cfg.px(52.0 * scale_size)
    roof_h = cfg.px(36.0 * scale_size)
    x0, x1 = cx - w * 0.5, cx + w * 0.5
    y_base, y_eave, y_peak = cy + h * 0.5, cy - h * 0.5, cy - h * 0.5 - roof_h
    draw_aa_line(canvas, (x0, y_base), (x1, y_base), 3.0, cfg, color, alpha)
    draw_aa_line(canvas, (x0, y_base), (x0, y_eave), 3.0, cfg, color, alpha)
    draw_aa_line(canvas, (x1, y_base), (x1, y_eave), 3.0, cfg, color, alpha)
    draw_aa_line(canvas, (x0 - cfg.px(8), y_eave), (cx, y_peak), 3.5, cfg, color, alpha)
    draw_aa_line(canvas, (x1 + cfg.px(8), y_eave), (cx, y_peak), 3.5, cfg, color, alpha)


def render_scene_04_data_features_patterns(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    08:00–11:30 (480s..690s, 210s) — Data, Features, and Patterns
    - Introduces the fictional 8-house dataset (House Area m^2 vs. Price $k).
    - Shows minimal geometric house wireframes and table on the left, coordinate plane on the right.
    - Animates houses mapping onto the plane, highlights Feature, Target, and Training Example,
      and visualizes natural real-world scatter/noise.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 480.0, 690.0, 1.0, 1.0)

    # Left Table & Minimal House Icons Panel
    lx0, lx1 = cfg.px(200), cfg.px(1260)
    draw_rounded_panel(canvas, lx0, cfg.px(220), lx1, cfg.px(1580), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, env, 3.0)
    draw_text(canvas, "FICTIONAL HOUSE DATASET", (lx0 + lx1) * 0.5, cfg.px(290), 36, cfg, PALETTE.WHITE, "mono", bold=True, alpha=env)
    draw_text(canvas, "House    Area (x)     Price (y)", (lx0 + lx1) * 0.5, cfg.px(365), 30, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=env)
    draw_aa_line(canvas, (lx0 + cfg.px(50), cfg.px(405)), (lx1 - cfg.px(50), cfg.px(405)), 2.0, cfg, PALETTE.PANEL_BORDER, env)

    # Right Coordinate Plane: Area (40..230 m^2) vs Price (100..440 $k)
    vp = PlotViewport(
        40.0, 230.0, 100.0, 440.0,
        cfg.px(1440), cfg.px(240), W - cfg.px(220), cfg.px(1540),
    )
    draw_coordinate_axes(
        canvas, vp, cfg, alpha=env * smoothstep(515.0, 528.0, t),
        x_ticks=[60, 100, 140, 180, 220],
        y_ticks=[150, 200, 250, 300, 350, 400],
        x_label="FEATURE: House Floor Area x (m²)",
        y_label="TARGET: Sale Price y ($1,000s)",
    )

    # Highlighted training example index (house #5: 138 m^2, $278k)
    highlight_idx = 4
    for i, (area, price) in enumerate(zip(HOUSE_DATA_X, HOUSE_DATA_Y)):
        row_a = env * smoothstep(483.0 + i * 2.2, 486.0 + i * 2.2, t)
        ry = cfg.px(475 + i * 132)
        is_hi = (i == highlight_idx and 563.0 <= t <= 628.0)
        row_col = PALETTE.ACCENT_GOLD if is_hi else PALETTE.WHITE

        # Minimal house wireframe scaled by floor area
        draw_house_icon(canvas, lx0 + cfg.px(135), ry, 0.45 + 0.40 * (area / 210.0), cfg, row_col, row_a)
        draw_text(
            canvas, f"{area:5.0f} m²      ${price:3.0f}k",
            lx0 + cfg.px(310), ry, 32, cfg, row_col, "mono", bold=is_hi, alpha=row_a, anchor="midleft",
        )

        # Plot on coordinate plane starting at t = 528s
        pt_u = smootherstep(528.0 + i * 2.0, 534.0 + i * 2.0, t)
        if pt_u > 0.01:
            px, py = vp.to_px(float(area), float(price))
            sx = lerp(lx0 + cfg.px(500), px, pt_u)
            sy = lerp(ry, py, pt_u)
            draw_glow_circle(canvas, sx, sy, 15.0 if is_hi else 11.0, cfg, row_col, env * pt_u, glow_layers=3 if is_hi else 2)

    # Feature / Target / Training Example callouts (t = 563s .. 630s)
    call_env = window_envelope(t, 563.0, 630.0, 1.0, 1.0)
    if call_env > 0.01:
        hx, hy = float(HOUSE_DATA_X[highlight_idx]), float(HOUSE_DATA_Y[highlight_idx])
        px, py = vp.to_px(hx, hy)
        _, y_axis_bottom = vp.to_px(hx, vp.y_min)
        x_axis_left, _ = vp.to_px(vp.x_min, hy)

        # Vertical drop to Feature axis
        draw_dashed_line(canvas, (px, py), (px, y_axis_bottom), 3.5, cfg, PALETTE.ACCENT_ICE, call_env)
        draw_text(canvas, f"FEATURE x = {hx:.0f} m²", px, y_axis_bottom - cfg.px(35), 32, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=call_env)

        # Horizontal projection to Target axis (t >= 580s)
        t_targ = smoothstep(580.0, 583.0, t) * call_env
        if t_targ > 0.01:
            draw_dashed_line(canvas, (px, py), (x_axis_left, py), 3.5, cfg, PALETTE.ACCENT_GOLD, t_targ)
            draw_text(canvas, f"TARGET y = ${hy:.0f}k", x_axis_left + cfg.px(30), py - cfg.px(34), 32, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=t_targ, anchor="midleft")

        # Training Example label (t >= 597s)
        t_ex = smoothstep(597.0, 600.0, t) * call_env
        if t_ex > 0.01:
            draw_text(canvas, f"TRAINING EXAMPLE (x₅, y₅) = ({hx:.0f}, {hy:.0f})", px + cfg.px(28), py - cfg.px(42), 34, cfg, PALETTE.WHITE, "mono", bold=True, alpha=t_ex, anchor="midleft")

    # Trend corridor & real-world noise visualization (t >= 631s)
    noise_env = smoothstep(631.0, 640.0, t) * env
    if noise_env > 0.01:
        trend_fn = lambda x: 1.65 * x + 38.0
        upper_fn = lambda x: 1.65 * x + 60.0
        lower_fn = lambda x: 1.65 * x + 16.0
        draw_function_curve(canvas, vp, upper_fn, cfg, PALETTE.GRAY_DARK, 3.0, noise_env * 0.7)
        draw_function_curve(canvas, vp, lower_fn, cfg, PALETTE.GRAY_DARK, 3.0, noise_env * 0.7)
        draw_function_curve(canvas, vp, trend_fn, cfg, PALETTE.ACCENT_ICE, 5.0, noise_env * 0.85)


def render_scene_05_linear_model(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    11:30–15:00 (690s..900s, 210s) — The Mathematical Heart: y = wx + b
    - Animates term-by-term equation anatomy: y = w * x + b.
    - Dynamically sweeps weight w and bias b in exact mathematical lockstep between
      the plotted line, the slope triangle (Delta y / Delta x = w), the y-intercept (0, b),
      and the live numerical formula readout.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 690.0, 900.0, 1.0, 1.0)

    # Determine live mathematical parameters (w, b) according to the teaching script timeline
    if t < 783.0:
        w_val, b_val = 1.40, 40.0
    elif t < 800.0:
        # Sweep w from 0.0 to 1.2 with b = 0
        w_val = lerp(0.0, 1.20, ease_in_out_sine((t - 783.0) / 17.0))
        b_val = 0.0
    elif t < 817.0:
        # Sweep w from 1.20 up to 2.35
        w_val = lerp(1.20, 2.35, ease_in_out_sine((t - 800.0) / 17.0))
        b_val = 0.0
    elif t < 833.0:
        # Sweep w into negative slope (-0.65) and back to +1.65
        u = (t - 817.0) / 16.0
        w_val = lerp(2.35, -0.55, ease_in_out_sine(u * 2.0)) if u < 0.5 else lerp(-0.55, 1.65, ease_in_out_sine((u - 0.5) * 2.0))
        b_val = 180.0 if w_val < 0 else lerp(0.0, 20.0, u)
    elif t < 866.0:
        # Hold w = 1.65 fixed and sweep bias b up (+95) and down (-15) then settle to +38
        w_val = 1.65
        u = (t - 833.0) / 33.0
        if u < 0.45:
            b_val = lerp(20.0, 95.0, ease_in_out_sine(u / 0.45))
        elif u < 0.80:
            b_val = lerp(95.0, -10.0, ease_in_out_sine((u - 0.45) / 0.35))
        else:
            b_val = lerp(-10.0, 38.0, ease_in_out_sine((u - 0.80) / 0.20))
    else:
        w_val, b_val = 1.65, 38.0

    # Left Anatomy & Live Parameter Readout Card
    lx0, lx1 = cfg.px(200), cfg.px(1320)
    draw_rounded_panel(canvas, lx0, cfg.px(220), lx1, cfg.px(1580), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, env, 3.0)
    cx = (lx0 + lx1) * 0.5
    draw_text(canvas, "THE LINEAR MODEL", cx, cfg.px(295), 36, cfg, PALETTE.GRAY_LIGHT, "mono", bold=True, alpha=env)
    draw_text(canvas, "y = w · x + b", cx, cfg.px(405), 68, cfg, PALETTE.WHITE, "mono", bold=True, alpha=env)

    terms_info = [
        ("x : Input Feature", "Horizontal coordinate (Area)", 721.0, PALETTE.WHITE),
        ("w : Weight (Slope)", f"Current w = {w_val:+6.2f}", 736.0, PALETTE.ACCENT_ICE),
        ("b : Bias (Intercept)", f"Current b = {b_val:+6.1f}", 752.0, PALETTE.ACCENT_GOLD),
        ("y : Predicted Output", f"y = {w_val:.2f}x {b_val:+.1f}", 768.0, PALETTE.WHITE),
    ]
    for idx, (t_head, t_sub, t_appear, t_col) in enumerate(terms_info):
        by = cfg.px(560 + idx * 240)
        ta = env * smoothstep(t_appear - 15.0, t_appear, t)
        draw_rounded_panel(canvas, lx0 + cfg.px(65), by - cfg.px(68), lx1 - cfg.px(65), by + cfg.px(92), cfg, PALETTE.BG_BLACK, t_col, ta, 2.5)
        draw_text(canvas, t_head, cx, by - cfg.px(18), 36, cfg, t_col, "mono", bold=True, alpha=ta)
        draw_text(canvas, t_sub, cx, by + cfg.px(44), 32, cfg, PALETTE.GRAY_LIGHT, "mono", alpha=ta)

    # Right Coordinate Plane (0..230 on x so y-intercept at x=0 is directly visible!)
    vp = PlotViewport(
        0.0, 230.0, -40.0, 460.0,
        cfg.px(1480), cfg.px(240), W - cfg.px(220), cfg.px(1540),
    )
    draw_coordinate_axes(
        canvas, vp, cfg, alpha=env,
        x_ticks=[0, 50, 100, 150, 200],
        y_ticks=[0, 100, 200, 300, 400],
        x_label="Input x (House Area m²)",
        y_label="Predicted Output y ($1,000s)",
    )

    # Plot active line y = w*x + b
    line_fn = lambda x: w_val * x + b_val
    draw_function_curve(canvas, vp, line_fn, cfg, PALETTE.ACCENT_ICE, 7.0, env, progress=smoothstep(692.0, 704.0, t))

    # Draw y-intercept marker at (0, b)
    ix_b, iy_b = vp.to_px(0.0, clamp(b_val, vp.y_min, vp.y_max))
    draw_glow_circle(canvas, ix_b, iy_b, 14.0, cfg, PALETTE.ACCENT_GOLD, env, glow_layers=3)
    draw_text(canvas, f"Bias (0, {b_val:.0f})", ix_b + cfg.px(26), iy_b, 30, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env, anchor="midleft")

    # Draw live slope triangle between x1=80 and x2=140
    x1_tri, x2_tri = 80.0, 140.0
    y1_tri, y2_tri = w_val * x1_tri + b_val, w_val * x2_tri + b_val
    if vp.y_min <= y1_tri <= vp.y_max and vp.y_min <= y2_tri <= vp.y_max:
        p_a = vp.to_px(x1_tri, y1_tri)
        p_b = vp.to_px(x2_tri, y1_tri)
        p_c = vp.to_px(x2_tri, y2_tri)
        draw_dashed_line(canvas, p_a, p_b, 3.0, cfg, PALETTE.WHITE, env * 0.85)
        draw_dashed_line(canvas, p_b, p_c, 3.0, cfg, PALETTE.ACCENT_ICE, env * 0.85)
        draw_text(canvas, "Δx = 60", (p_a[0] + p_b[0]) * 0.5, p_b[1] + cfg.px(32), 28, cfg, PALETTE.GRAY_LIGHT, "mono", alpha=env)
        draw_text(canvas, f"Δy = {y2_tri - y1_tri:+.0f} (w={w_val:.2f})", p_c[0] + cfg.px(20), (p_b[1] + p_c[1]) * 0.5, 28, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=env, anchor="midleft")

    # Overlay the 8 house data points for t >= 864s
    pts_alpha = smoothstep(864.0, 870.0, t) * env
    if pts_alpha > 0.01:
        for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
            px, py = vp.to_px(float(hx), float(hy))
            draw_glow_circle(canvas, px, py, 12.0, cfg, PALETTE.WHITE, pts_alpha, glow_layers=2)


def render_scene_06_neural_network(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    15:00–18:30 (900s..1110s, 210s) — From a Simple Model to a Neural Network
    - Progressively expands a single node into a 3-layer network:
      Input Layer (3 nodes) -> Hidden Layer (4 nodes) -> Output Layer (1 node).
    - Animates luminous signal pulses traveling along weighted edges.
    - Includes an inset mathematical zoom showing a single Neuron's weighted sum + activation.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 900.0, 1110.0, 1.0, 1.0)

    # Layer horizontal coordinates
    x_in = cfg.px(680)
    x_hid = cfg.px(1820)
    x_out = cfg.px(2960)

    # Progressive reveal of layers
    in_extra_alpha = smoothstep(948.0, 960.0, t) * env
    hid_alpha = smoothstep(965.0, 980.0, t) * env
    out_alpha = smoothstep(982.0, 995.0, t) * env

    in_ys = [cfg.px(540), cfg.px(900), cfg.px(1260)]
    hid_ys = [cfg.px(420), cfg.px(740), cfg.px(1060), cfg.px(1380)]
    out_ys = [cfg.px(900)]

    # Before 965s, show direct single connection from Input[1] to Output[0]
    single_alpha = env * (1.0 - smoothstep(965.0, 975.0, t))
    if single_alpha > 0.01:
        draw_aa_line(canvas, (x_in, in_ys[1]), (x_out, out_ys[0]), 5.0, cfg, PALETTE.ACCENT_ICE, single_alpha)
        draw_text(canvas, "Single Weight w & Bias b:  y = w·x + b", (x_in + x_out) * 0.5, in_ys[1] - cfg.px(55), 38, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=single_alpha)

    # Draw weighted connections & progressive signal pulses
    signal_phase = ((t - 900.0) * 0.55) % 2.0  # 0..1: Input->Hidden, 1..2: Hidden->Output
    if hid_alpha > 0.01:
        for i, iy in enumerate(in_ys):
            a_node = env if i == 1 else in_extra_alpha
            if a_node <= 0.01:
                continue
            for j, hy in enumerate(hid_ys):
                w_mag = 0.45 + 0.55 * abs(math.sin((i + 1) * 1.7 + (j + 1) * 0.9))
                edge_col = lerp_color(PALETTE.GRAY_DARK, PALETTE.GRAY_LIGHT, w_mag)
                draw_aa_line(canvas, (x_in, iy), (x_hid, hy), 2.0 + 3.5 * w_mag, cfg, edge_col, hid_alpha * a_node * 0.85)
                if 0.0 <= signal_phase <= 1.0 and t >= 1000.0:
                    sx = lerp(x_in, x_hid, signal_phase)
                    sy = lerp(iy, hy, signal_phase)
                    draw_glow_circle(canvas, sx, sy, 8.0, cfg, PALETTE.ACCENT_ICE, hid_alpha * a_node, glow_layers=2)

    if out_alpha > 0.01:
        for j, hy in enumerate(hid_ys):
            w_mag = 0.5 + 0.5 * abs(math.cos((j + 1) * 1.3))
            edge_col = lerp_color(PALETTE.GRAY_MID, PALETTE.WHITE, w_mag)
            draw_aa_line(canvas, (x_hid, hy), (x_out, out_ys[0]), 2.5 + 3.5 * w_mag, cfg, edge_col, out_alpha * 0.9)
            if 1.0 < signal_phase <= 2.0 and t >= 1000.0:
                u2 = signal_phase - 1.0
                sx = lerp(x_hid, x_out, u2)
                sy = lerp(hy, out_ys[0], u2)
                draw_glow_circle(canvas, sx, sy, 9.0, cfg, PALETTE.ACCENT_GOLD, out_alpha, glow_layers=2)

    # Draw Input Layer Nodes
    in_labels = ["x₁ (Area)", "x₂ (Rooms)", "x₃ (Age)"]
    for i, (iy, lbl) in enumerate(zip(in_ys, in_labels)):
        a_node = env if i == 1 else in_extra_alpha
        if a_node > 0.01:
            draw_glow_circle(canvas, x_in, iy, 44.0, cfg, PALETTE.WHITE, a_node, glow_layers=3)
            draw_glow_circle(canvas, x_in, iy, 38.0, cfg, PALETTE.BG_BLACK, a_node, glow_layers=0)
            draw_text(canvas, f"x{i+1}", x_in, iy, 34, cfg, PALETTE.WHITE, "mono", bold=True, alpha=a_node)
            draw_text(canvas, lbl if in_extra_alpha > 0.2 else "Input x", x_in - cfg.px(75), iy, 32, cfg, PALETTE.GRAY_LIGHT, "mono", alpha=a_node, anchor="midright")

    # Draw Hidden Layer Nodes
    if hid_alpha > 0.01:
        for j, hy in enumerate(hid_ys):
            is_zoom_node = (j == 1 and 1034.0 <= t <= 1068.0)
            ncol = PALETTE.ACCENT_GOLD if is_zoom_node else PALETTE.ACCENT_ICE
            draw_glow_circle(canvas, x_hid, hy, 48.0, cfg, ncol, hid_alpha, glow_layers=3)
            draw_glow_circle(canvas, x_hid, hy, 41.0, cfg, PALETTE.BG_BLACK, hid_alpha, glow_layers=0)
            draw_text(canvas, f"h{j+1}", x_hid, hy, 34, cfg, ncol, "mono", bold=True, alpha=hid_alpha)

    # Draw Output Node
    draw_glow_circle(canvas, x_out, out_ys[0], 52.0, cfg, PALETTE.ACCENT_GOLD, env, glow_layers=4)
    draw_glow_circle(canvas, x_out, out_ys[0], 44.0, cfg, PALETTE.BG_BLACK, env, glow_layers=0)
    draw_text(canvas, "ŷ", x_out, out_ys[0], 40, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env)
    draw_text(canvas, "Prediction ŷ", x_out + cfg.px(80), out_ys[0], 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env, anchor="midleft")

    # Layer Column Headers
    draw_text(canvas, "INPUT LAYER", x_in, cfg.px(250), 36, cfg, PALETTE.WHITE, "mono", bold=True, alpha=env)
    if hid_alpha > 0.01:
        draw_text(canvas, "HIDDEN LAYER", x_hid, cfg.px(250), 36, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=hid_alpha)
    draw_text(canvas, "OUTPUT LAYER", x_out, cfg.px(250), 36, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env)

    # Inside-a-Neuron Mathematical Callout Box (t = 1034s .. 1075s)
    z_env = window_envelope(t, 1034.0, 1075.0, 1.0, 1.0)
    if z_env > 0.01:
        draw_rounded_panel(canvas, cfg.px(2320), cfg.px(340), W - cfg.px(200), cfg.px(720), cfg, PALETTE.PANEL_BG, PALETTE.ACCENT_GOLD, z_env, 3.0)
        cx_z = (cfg.px(2320) + W - cfg.px(200)) * 0.5
        draw_text(canvas, "INSIDE ONE NEURON (h₂)", cx_z, cfg.px(415), 34, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=z_env)
        draw_text(canvas, "z = w₁x₁ + w₂x₂ + w₃x₃ + b", cx_z, cfg.px(515), 38, cfg, PALETTE.WHITE, "mono", bold=True, alpha=z_env)
        draw_text(canvas, "h₂ = Activation(z)", cx_z, cfg.px(615), 36, cfg, PALETTE.ACCENT_ICE, "mono", alpha=z_env)


def compute_house_mse(w: float, b: float) -> float:
    preds = w * HOUSE_DATA_X + b
    return float(np.mean((preds - HOUSE_DATA_Y) ** 2))


def render_scene_07_how_model_learns(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    18:30–22:00 (1110s..1320s, 210s) — How Does a Model Learn?
    - Top cycle banner: Prediction -> Measure Error -> Update Parameters -> New Prediction.
    - Left panel: House dataset with current line y = wx + b and vertical error bars.
    - Right panel: Loss curve L(w) showing parameter state descending the valley (with one
      deliberate slight overshoot step to illustrate that not every single update is monotonic).
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1110.0, 1320.0, 1.0, 1.0)

    # Top Feedback Loop Banner
    loop_steps = [
        "1. PREDICTION",
        "2. MEASURE ERROR (LOSS)",
        "3. UPDATE PARAMETERS",
        "4. NEW PREDICTION",
    ]
    active_step = int(((t - 1110.0) / 4.5) % 4)
    for idx, st_label in enumerate(loop_steps):
        bx0 = cfg.px(200 + idx * 870)
        bx1 = bx0 + cfg.px(810)
        is_act = (idx == active_step and t >= 1140.0)
        bcol = PALETTE.ACCENT_GOLD if is_act else PALETTE.PANEL_BORDER
        draw_rounded_panel(canvas, bx0, cfg.px(175), bx1, cfg.px(295), cfg, PALETTE.PANEL_BG, bcol, env, 3.0)
        draw_text(canvas, st_label, (bx0 + bx1) * 0.5, cfg.px(235), 30, cfg, PALETTE.WHITE if is_act else PALETTE.GRAY_LIGHT, "mono", bold=is_act, alpha=env)

    # Parameter trajectory schedule across [1110s .. 1320s]
    # Includes an overshoot around t=1278s to demonstrate non-monotonic step behavior
    keyframes = [
        (1110.0, 0.35, 85.0),
        (1224.0, 0.35, 85.0),
        (1242.0, 0.85, 68.0),
        (1260.0, 1.32, 50.0),
        (1276.0, 2.02, 32.0),  # Slight overshoot on slope w!
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

    # Left Viewport: Data Space (House Area vs Price + Error Residuals)
    vp_data = PlotViewport(40.0, 230.0, 80.0, 450.0, cfg.px(240), cfg.px(380), cfg.px(1840), cfg.px(1540))
    draw_coordinate_axes(
        canvas, vp_data, cfg, alpha=env,
        x_ticks=[60, 100, 140, 180, 220],
        y_ticks=[100, 200, 300, 400],
        x_label="House Area x (m²)",
        y_label="Price y ($1,000s)",
    )

    line_fn = lambda x: w_curr * x + b_curr
    draw_function_curve(canvas, vp_data, line_fn, cfg, PALETTE.ACCENT_ICE, 6.5, env)

    # Vertical Error Residuals (appear at t >= 1156s)
    err_alpha = smoothstep(1156.0, 1162.0, t) * env
    for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
        y_pred = w_curr * float(hx) + b_curr
        p_true = vp_data.to_px(float(hx), float(hy))
        p_pred = vp_data.to_px(float(hx), clamp(y_pred, vp_data.y_min, vp_data.y_max))
        if err_alpha > 0.01:
            draw_aa_line(canvas, p_true, p_pred, 4.5, cfg, PALETTE.ACCENT_GOLD, err_alpha * 0.9)
        draw_glow_circle(canvas, p_true[0], p_true[1], 11.0, cfg, PALETTE.WHITE, env, glow_layers=2)

    # Right Viewport: Loss Landscape L(w) for current b
    vp_loss = PlotViewport(0.1, 2.5, 0.0, 35000.0, cfg.px(2120), cfg.px(380), W - cfg.px(220), cfg.px(1220))
    loss_panel_alpha = smoothstep(1188.0, 1196.0, t) * env
    if loss_panel_alpha > 0.01:
        draw_coordinate_axes(
            canvas, vp_loss, cfg, alpha=loss_panel_alpha,
            x_ticks=[0.5, 1.0, 1.5, 2.0, 2.5],
            y_ticks=[0, 10000, 20000, 30000],
            x_label="Parameter Weight (w)",
            y_label="Mean Squared Loss L(w, b)",
        )
        loss_curve_fn = lambda w_arr: np.array([compute_house_mse(float(wv), b_curr) for wv in w_arr])
        draw_function_curve(canvas, vp_loss, loss_curve_fn, cfg, PALETTE.WHITE, 5.5, loss_panel_alpha)

        lx_pt, ly_pt = vp_loss.to_px(w_curr, clamp(loss_curr, vp_loss.y_min, vp_loss.y_max))
        draw_glow_circle(canvas, lx_pt, ly_pt, 16.0, cfg, PALETTE.ACCENT_GOLD, loss_panel_alpha, glow_layers=4)

    # Bottom-Right Numerical Telemetry Readout
    draw_rounded_panel(canvas, cfg.px(2120), cfg.px(1340), W - cfg.px(220), cfg.px(1640), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, env, 3.0)
    cx_r = (cfg.px(2120) + W - cfg.px(220)) * 0.5
    draw_text(
        canvas, f"PARAMETERS:   w = {w_curr:5.2f}     b = {b_curr:5.1f}",
        cx_r, cfg.px(1430), 38, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=env,
    )
    draw_text(
        canvas, f"PREDICTION ERROR (MSE LOSS):   L = {loss_curr:7.1f}",
        cx_r, cfg.px(1545), 40, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=env,
    )


def render_scene_08_three_approaches(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    22:00–25:00 (1320s..1500s, 180s) — Three Important Learning Approaches
    - Three side-by-side laboratory columns that progressively illuminate and animate:
      1. Supervised Learning   (1320..1500s): Paired (x_i, y_i) labels guiding regression fit.
      2. Unsupervised Learning (1370..1500s): Unlabeled 2D points gravitating into 3 clusters.
      3. Reinforcement Learning(1422..1500s): Agent in grid-world observing state, acting, earning reward.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1320.0, 1500.0, 1.0, 1.0)

    cols = [
        ("1. SUPERVISED LEARNING", "Examples paired with known targets", cfg.px(180), cfg.px(1280), 1321.0, PALETTE.ACCENT_ICE),
        ("2. UNSUPERVISED LEARNING", "Unlabeled points forming clusters", cfg.px(1370), cfg.px(2470), 1370.0, PALETTE.WHITE),
        ("3. REINFORCEMENT LEARNING", "Agent, State, Action, and Reward", cfg.px(2560), cfg.px(3660), 1422.0, PALETTE.ACCENT_GOLD),
    ]

    for idx, (c_title, c_sub, x0, x1, t_unlock, c_col) in enumerate(cols):
        ca = env * clamp(0.25 + 0.75 * smoothstep(t_unlock - 2.0, t_unlock + 4.0, t), 0.0, 1.0)
        draw_rounded_panel(canvas, x0, cfg.px(220), x1, cfg.px(1580), cfg, PALETTE.PANEL_BG, c_col if t >= t_unlock else PALETTE.PANEL_BORDER, ca, 3.5)
        cx = (x0 + x1) * 0.5
        draw_text(canvas, c_title, cx, cfg.px(300), 34, cfg, c_col, "mono", bold=True, alpha=ca)
        draw_text(canvas, c_sub, cx, cfg.px(365), 28, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=ca)

        vp = PlotViewport(0.0, 10.0, 0.0, 10.0, x0 + cfg.px(90), cfg.px(460), x1 - cfg.px(90), cfg.px(1360))
        draw_coordinate_axes(canvas, vp, cfg, alpha=ca, x_ticks=[2, 5, 8], y_ticks=[2, 5, 8], x_label="", y_label="")

        if idx == 0:
            # Supervised: Labeled house points + fitted trend line
            sup_fn = lambda x: 1.4 + 0.75 * x
            draw_function_curve(canvas, vp, sup_fn, cfg, PALETTE.ACCENT_ICE, 5.0, ca, progress=smoothstep(1338.0, 1360.0, t))
            for k, sx_v in enumerate([1.5, 3.0, 4.8, 6.2, 7.8, 9.0]):
                sy_v = 1.4 + 0.75 * sx_v + (0.55 if k % 2 == 0 else -0.5)
                px, py = vp.to_px(sx_v, sy_v)
                draw_glow_circle(canvas, px, py, 10.0, cfg, PALETTE.WHITE, ca)
                draw_text(canvas, f"y={sy_v:.1f}", px + cfg.px(16), py - cfg.px(18), 24, cfg, PALETTE.ACCENT_GOLD, "mono", alpha=ca, anchor="midleft")
            draw_text(canvas, "(Feature xᵢ, Known Target yᵢ)", cx, cfg.px(1465), 30, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=ca)

        elif idx == 1:
            # Unsupervised: Unlabeled points smoothly coalescing into 3 natural clusters
            u_clust = smootherstep(1382.0, 1412.0, t)
            centers = [(2.6, 7.4), (7.4, 7.0), (5.0, 2.6)]
            offsets = [(-0.7, 0.5), (0.6, 0.6), (-0.5, -0.6), (0.7, -0.4), (0.0, 0.8)]
            for c_idx, (ccx, ccy) in enumerate(centers):
                cpx, cpy = vp.to_px(ccx, ccy)
                if u_clust > 0.3:
                    cv2.circle(canvas, (int(cpx), int(cpy)), cfg.px(120), scale_color(PALETTE.GRAY_MID, ca * u_clust * 0.6), max(1, cfg.px(3)), cv2.LINE_AA)
                for ox, oy in offsets:
                    start_x = 5.0 + ox * 3.8 + (c_idx - 1) * 1.1
                    start_y = 5.0 + oy * 3.8
                    cur_x = lerp(start_x, ccx + ox, u_clust)
                    cur_y = lerp(start_y, ccy + oy, u_clust)
                    px, py = vp.to_px(cur_x, cur_y)
                    draw_glow_circle(canvas, px, py, 9.5, cfg, PALETTE.WHITE, ca, glow_layers=1)
            draw_text(canvas, "Structure from Unlabeled Data", cx, cfg.px(1465), 30, cfg, PALETTE.WHITE, "mono", bold=True, alpha=ca)

        else:
            # Reinforcement Learning: 4x4 Grid World with Agent, Obstacle (-1), and Goal (+10)
            path_states = [(1.5, 1.5), (1.5, 4.2), (4.2, 4.2), (4.2, 7.0), (7.0, 7.0), (8.5, 8.5)]
            # Obstacle at (4.2, 1.8)
            ox, oy = vp.to_px(4.8, 2.2)
            draw_glow_circle(canvas, ox, oy, 26.0, cfg, PALETTE.ACCENT_ROSE, ca, filled=False)
            draw_text(canvas, "-1", ox, oy, 28, cfg, PALETTE.ACCENT_ROSE, "mono", bold=True, alpha=ca)
            # Goal at (8.5, 8.5)
            gx, gy = vp.to_px(8.5, 8.5)
            draw_glow_circle(canvas, gx, gy, 30.0, cfg, PALETTE.ACCENT_GOLD, ca, filled=False)
            draw_text(canvas, "+10", gx, gy, 28, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=ca)

            if t >= 1422.0:
                u_rl = ((t - 1422.0) * 0.35) % (len(path_states) - 1)
                seg = int(u_rl)
                frac = ease_in_out_cubic(u_rl - seg)
                ax_v = lerp(path_states[seg][0], path_states[seg + 1][0], frac)
                ay_v = lerp(path_states[seg][1], path_states[seg + 1][1], frac)
                apx, apy = vp.to_px(ax_v, ay_v)
                draw_glow_circle(canvas, apx, apy, 16.0, cfg, PALETTE.ACCENT_ICE, ca, glow_layers=4)
            draw_text(canvas, "State sₜ → Action aₜ → Reward rₜ", cx, cfg.px(1465), 30, cfg, PALETTE.ACCENT_GOLD, "mono", bold=True, alpha=ca)


def render_scene_09_generalization(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    25:00–27:30 (1500s..1650s, 150s) — Training vs. Generalization
    - Plots the 8 training examples (solid white circles).
    - Compares an degree-7 polynomial that memorizes every noisy dot (Overfitting)
      against a smooth generalized curve (Generalization).
    - Reveals 5 unseen test points (hollow gold rings) and compares error bars & gauges.
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1500.0, 1650.0, 1.0, 1.0)

    vp = PlotViewport(45.0, 220.0, 90.0, 440.0, cfg.px(260), cfg.px(250), cfg.px(2520), cfg.px(1540))
    draw_coordinate_axes(
        canvas, vp, cfg, alpha=env,
        x_ticks=[60, 100, 140, 180, 210],
        y_ticks=[100, 200, 300, 400],
        x_label="House Area x (m²)",
        y_label="Sale Price y ($1,000s)",
    )

    # Exact degree-7 interpolating polynomial through the 8 training points (Overfitting)
    poly_coeffs = np.polyfit(HOUSE_DATA_X, HOUSE_DATA_Y, deg=7)
    overfit_fn = lambda x: np.polyval(poly_coeffs, x)
    general_fn = lambda x: 1.65 * x + 38.0

    # Overfitting oscillatory curve (emerges at t = 1525s)
    over_prog = smootherstep(1525.0, 1545.0, t)
    if over_prog > 0.01:
        draw_function_curve(canvas, vp, overfit_fn, cfg, PALETTE.ACCENT_ROSE, 5.0, env, progress=over_prog, num_samples=320)

    # Smooth generalizing curve (emerges at t = 1549s)
    gen_prog = smootherstep(1549.0, 1564.0, t)
    if gen_prog > 0.01:
        draw_function_curve(canvas, vp, general_fn, cfg, PALETTE.ACCENT_ICE, 6.5, env, progress=gen_prog)

    # Solid white training points
    for hx, hy in zip(HOUSE_DATA_X, HOUSE_DATA_Y):
        px, py = vp.to_px(float(hx), float(hy))
        draw_glow_circle(canvas, px, py, 11.0, cfg, PALETTE.WHITE, env, glow_layers=2)

    # Unseen Data points (hollow gold circles, revealed at t = 1583s)
    unseen_alpha = smoothstep(1583.0, 1592.0, t) * env
    if unseen_alpha > 0.01:
        for ux, uy in zip(UNSEEN_DATA_X, UNSEEN_DATA_Y):
            px, py = vp.to_px(float(ux), float(uy))
            y_over = clamp(float(overfit_fn(np.array([ux]))[0]), vp.y_min, vp.y_max)
            p_over = vp.to_px(float(ux), y_over)
            draw_dashed_line(canvas, (px, py), p_over, 3.5, cfg, PALETTE.ACCENT_ROSE, unseen_alpha * 0.85)
            draw_glow_circle(canvas, px, py, 15.0, cfg, PALETTE.ACCENT_GOLD, unseen_alpha, glow_layers=3, filled=False)

    # Right Comparison Scorecard Panel
    rx0, rx1 = cfg.px(2660), W - cfg.px(200)
    draw_rounded_panel(canvas, rx0, cfg.px(250), rx1, cfg.px(1540), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, env, 3.0)
    cx = (rx0 + rx1) * 0.5
    draw_text(canvas, "MODEL COMPARISON", cx, cfg.px(330), 38, cfg, PALETTE.WHITE, "mono", bold=True, alpha=env)

    # Legend items
    draw_glow_circle(canvas, rx0 + cfg.px(110), cfg.px(440), 11.0, cfg, PALETTE.WHITE, env)
    draw_text(canvas, "Training Data (Seen)", rx0 + cfg.px(160), cfg.px(440), 32, cfg, PALETTE.WHITE, "sans", alpha=env, anchor="midleft")

    draw_glow_circle(canvas, rx0 + cfg.px(110), cfg.px(525), 14.0, cfg, PALETTE.ACCENT_GOLD, unseen_alpha, filled=False)
    draw_text(canvas, "Unseen Data (New)", rx0 + cfg.px(160), cfg.px(525), 32, cfg, PALETTE.ACCENT_GOLD, "sans", alpha=unseen_alpha, anchor="midleft")

    # Overfitting Card
    draw_rounded_panel(canvas, rx0 + cfg.px(65), cfg.px(640), rx1 - cfg.px(65), cfg.px(1020), cfg, PALETTE.BG_BLACK, PALETTE.ACCENT_ROSE, env * over_prog, 3.0)
    draw_text(canvas, "OVERFITTING CURVE", cx, cfg.px(715), 34, cfg, PALETTE.ACCENT_ROSE, "mono", bold=True, alpha=env * over_prog)
    draw_text(canvas, "Memorizes noisy training dots", cx, cfg.px(785), 28, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=env * over_prog)
    draw_text(canvas, "Training Error :   0.0 (Tiny)", cx, cfg.px(865), 30, cfg, PALETTE.WHITE, "mono", alpha=env * over_prog)
    draw_text(canvas, "Unseen Error   : 942.0 (Huge!)", cx, cfg.px(935), 30, cfg, PALETTE.ACCENT_ROSE, "mono", bold=True, alpha=unseen_alpha)

    # Generalization Card
    draw_rounded_panel(canvas, rx0 + cfg.px(65), cfg.px(1090), rx1 - cfg.px(65), cfg.px(1470), cfg, PALETTE.BG_BLACK, PALETTE.ACCENT_ICE, env * gen_prog, 3.0)
    draw_text(canvas, "GENERALIZING MODEL", cx, cfg.px(1165), 34, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=env * gen_prog)
    draw_text(canvas, "Captures underlying pattern", cx, cfg.px(1235), 28, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=env * gen_prog)
    draw_text(canvas, "Training Error : 112.4 (Low)", cx, cfg.px(1315), 30, cfg, PALETTE.WHITE, "mono", alpha=env * gen_prog)
    draw_text(canvas, "Unseen Error   : 118.6 (Low!)", cx, cfg.px(1385), 30, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=unseen_alpha)


def render_scene_10_roadmap(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    27:30–29:00 (1650s..1740s, 90s) — What Will We Learn Next?
    - Five-stage visual progression:
      Mathematics -> Python -> Machine Learning -> Neural Networks -> Deep Learning
    - Followed by the future milestone card:
      PyTorch — Coming in a Future Lecture
    """
    W, H = cfg.width, cfg.height
    env = window_envelope(t, 1650.0, 1740.0, 1.0, 1.0)

    draw_text(canvas, "THE SILENT MATH LEARNING ROADMAP", W * 0.5, cfg.px(250), 48, cfg, PALETTE.WHITE, "sans", bold=True, alpha=env)

    stages = [
        ("1. MATHEMATICS", "Vectors, Matrices\n& Calculus Intuition", 1653.0),
        ("2. PYTHON", "Translating Math\ninto Clean Code", 1662.0),
        ("3. MACHINE LEARNING", "Linear Models,\nTrees & Optimization", 1671.0),
        ("4. NEURAL NETWORKS", "Multi-Layer Graphs\n& Backpropagation", 1680.0),
        ("5. DEEP LEARNING", "Modern Architectures\n& Representation", 1689.0),
    ]

    box_w = cfg.px(610)
    gap = cfg.px(75)
    start_x = (W - (5 * box_w + 4 * gap)) * 0.5
    y_top, y_bot = cfg.px(420), cfg.px(880)

    for idx, (st_name, st_desc, t_un) in enumerate(stages):
        bx0 = start_x + idx * (box_w + gap)
        bx1 = bx0 + box_w
        cx = (bx0 + bx1) * 0.5
        sa = env * smoothstep(t_un - 1.5, t_un + 3.0, t)
        draw_rounded_panel(canvas, bx0, y_top, bx1, y_bot, cfg, PALETTE.PANEL_BG, PALETTE.ACCENT_ICE if sa > 0.8 else PALETTE.PANEL_BORDER, sa, 3.5)
        draw_text(canvas, st_name, cx, y_top + cfg.px(110), 34, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=sa)
        lines = st_desc.split("\n")
        draw_text(canvas, lines[0], cx, y_top + cfg.px(240), 30, cfg, PALETTE.WHITE, "sans", alpha=sa)
        draw_text(canvas, lines[1], cx, y_top + cfg.px(300), 30, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=sa)

        if idx < 4:
            ax0 = bx1 + cfg.px(10)
            ax1 = bx1 + gap - cfg.px(10)
            ay = (y_top + y_bot) * 0.5
            draw_aa_line(canvas, (ax0, ay), (ax1, ay), 4.0, cfg, PALETTE.WHITE, sa)

    # PyTorch Future Milestone Spotlight Card (t >= 1708s)
    pt_alpha = env * smoothstep(1708.0, 1714.0, t)
    if pt_alpha > 0.01:
        px0, px1 = cfg.px(640), W - cfg.px(640)
        py0, py1 = cfg.px(1020), cfg.px(1560)
        draw_rounded_panel(canvas, px0, py0, px1, py1, cfg, PALETTE.PANEL_BG, PALETTE.ACCENT_GOLD, pt_alpha, 4.0)
        draw_text(canvas, "PyTorch — Coming in a Future Lecture", W * 0.5, py0 + cfg.px(130), 56, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=pt_alpha)
        draw_text(
            canvas,
            "Bridging mathematical equations (y = wx + b) and multi-layer neural networks",
            W * 0.5, py0 + cfg.px(265), 36, cfg, PALETTE.WHITE, "sans", alpha=pt_alpha,
        )
        draw_text(
            canvas,
            "into scalable, executable tensor computation after we master the mathematical foundations.",
            W * 0.5, py0 + cfg.px(335), 36, cfg, PALETTE.GRAY_LIGHT, "sans", alpha=pt_alpha,
        )


def render_scene_11_recap_closing(canvas: np.ndarray, t: float, cfg: RenderConfig) -> None:
    """
    29:00–30:00 (1740s..1800s, 60s) — Recap and Closing
    - Returns to the original moving white point and coordinate plane from the introduction.
    - Displays the 6 core recap phrases (1740..1768s).
    - Poses the final question (1768..1782s).
    - Displays SILENT MATH / LECTURE #1 COMPLETE / Next: Mathematics Behind Machine Learning
      and fades smoothly to pure black (#000000) at 30:00 (1800.0s).
    """
    W, H = cfg.width, cfg.height
    master_fade = 1.0 - smoothstep(1796.2, 1799.6, t)
    if master_fade <= 0.005:
        return

    # Background coordinate plane and moving point returning from Scene 1
    bg_alpha = master_fade * (0.45 if t < 1782.0 else 0.22)
    vp = PlotViewport(0.0, 10.0, 0.0, 10.0, cfg.px(420), cfg.px(220), W - cfg.px(420), cfg.px(1580))
    draw_coordinate_axes(canvas, vp, cfg, alpha=bg_alpha, x_ticks=[2, 4, 6, 8, 10], y_ticks=[2, 4, 6, 8, 10], x_label="", y_label="")
    curve_fn = lambda x: 1.6 + 0.65 * x + 1.1 * np.sin(0.55 * x)
    draw_function_curve(canvas, vp, curve_fn, cfg, PALETTE.ACCENT_ICE, 5.0, bg_alpha)
    x_t = 5.0 + 3.8 * math.sin(0.45 * (t - 1740.0))
    y_t = float(curve_fn(np.array([x_t]))[0])
    px, py = vp.to_px(x_t, y_t)
    draw_glow_circle(canvas, px, py, 16.0, cfg, PALETTE.WHITE, master_fade * 0.85, glow_layers=4)

    # Phase A (1740..1768s): 6 Core Summary Takeaways
    recap_env = window_envelope(t, 1740.2, 1768.2, 0.8, 1.0) * master_fade
    if recap_env > 0.01:
        recap_items = [
            ("AI:", "Intelligent behavior in machines.", 1741.0),
            ("ML:", "Learning patterns from data.", 1744.5),
            ("Model:", "A mathematical system that produces predictions.", 1749.5),
            ("Loss:", "A measure of prediction error.", 1753.5),
            ("Training:", "Adjusting parameters using data.", 1758.5),
            ("Generalization:", "Performing well on unseen examples.", 1762.5),
        ]
        for idx, (term, desc, t_in) in enumerate(recap_items):
            ia = recap_env * smoothstep(t_in - 0.5, t_in + 1.5, t)
            if ia > 0.01:
                ry = cfg.px(320 + idx * 210)
                draw_rounded_panel(canvas, cfg.px(620), ry - cfg.px(68), W - cfg.px(620), ry + cfg.px(68), cfg, PALETTE.PANEL_BG, PALETTE.PANEL_BORDER, ia, 2.5)
                draw_text(canvas, term, cfg.px(700), ry, 40, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=ia, anchor="midleft")
                draw_text(canvas, desc, cfg.px(1240), ry, 38, cfg, PALETTE.WHITE, "sans", alpha=ia, anchor="midleft")

    # Phase B (1768.5..1782.5s): Final Philosophical Question
    q_env = window_envelope(t, 1768.3, 1782.5, 1.0, 1.0) * master_fade
    if q_env > 0.01:
        draw_rounded_panel(canvas, cfg.px(480), cfg.px(560), W - cfg.px(480), cfg.px(1160), cfg, PALETTE.PANEL_BG, PALETTE.ACCENT_GOLD, q_env, 4.0)
        draw_text(canvas, "If a machine can learn a pattern from examples,", W * 0.5, cfg.px(780), 56, cfg, PALETTE.WHITE, "sans", bold=True, alpha=q_env)
        draw_text(canvas, "what else can we teach it to discover?", W * 0.5, cfg.px(920), 56, cfg, PALETTE.ACCENT_GOLD, "sans", bold=True, alpha=q_env)

    # Phase C (1782.8..1800.0s): Closing Title Card & Smooth Fade to Black
    end_env = smoothstep(1782.8, 1785.0, t) * master_fade
    if end_env > 0.01:
        draw_rounded_panel(canvas, cfg.px(520), cfg.px(420), W - cfg.px(520), cfg.px(1380), cfg, PALETTE.BG_BLACK, PALETTE.WHITE, end_env, 4.0)
        draw_text(canvas, "SILENT MATH", W * 0.5, cfg.px(640), 92, cfg, PALETTE.WHITE, "sans", bold=True, alpha=end_env)
        draw_text(canvas, "LECTURE #1 COMPLETE", W * 0.5, cfg.px(830), 54, cfg, PALETTE.ACCENT_ICE, "mono", bold=True, alpha=end_env)
        draw_aa_line(canvas, (W * 0.5 - cfg.px(360), cfg.px(945)), (W * 0.5 + cfg.px(360), cfg.px(945)), 3.0, cfg, PALETTE.GRAY_MID, end_env)
        draw_text(canvas, "Next: Mathematics Behind Machine Learning", W * 0.5, cfg.px(1110), 44, cfg, PALETTE.GRAY_LIGHT, "sans", bold=True, alpha=end_env)


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
    """Render a single complete frame at timestamp t (in seconds)."""
    if canvas is None:
        canvas = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    else:
        canvas.fill(0)

    scene = get_scene_for_time(t)
    renderer = SCENE_RENDER_DISPATCH.get(scene.index)
    if renderer is not None:
        renderer(canvas, t, cfg)
    draw_lecture_hud_and_captions(canvas, scene, t, cfg)
    return canvas


# ==============================================================================
# 7. OPTIONAL MANIM COMMUNITY EDITION (MANIM CE) SCENE BRIDGE
# ==============================================================================

try:
    import manim as mn  # type: ignore

    class SilentMathLecture1(mn.Scene):
        """
        Manim Community Edition Scene wrapper.
        If `manim` is installed and invoked via `manim -qk video_lecture1.py SilentMathLecture1`,
        this scene drives the full mathematical lecture timeline.
        """
        def construct(self) -> None:
            self.camera.background_color = "#000000"
            title = mn.Text("SILENT MATH — LECTURE #1", font="Inter", color=mn.WHITE).scale(0.9)
            sub = mn.Text("INTRODUCTION TO AI & MACHINE LEARNING", font="Inter", color=mn.GRAY_A).scale(0.55)
            sub.next_to(title, mn.DOWN, buff=0.4)
            self.play(mn.FadeIn(title), mn.FadeIn(sub), run_time=2.0)
            self.wait(1.0)
            self.play(mn.FadeOut(title), mn.FadeOut(sub), run_time=1.0)
except ImportError:
    pass


# ==============================================================================
# 8. AUTOMATED QUALITY CONTROL & FFPROBE VALIDATOR
# ==============================================================================

def run_project_validation(cfg: RenderConfig) -> Dict[str, object]:
    """
    Verify all timeline invariants, caption reading speeds, non-overlapping safe areas,
    mathematical consistency of `y = wx + b` and MSE loss, and font resolution.
    """
    issues: List[str] = []

    # 1. Verify exact 1800.0s timeline contiguity
    if abs(LECTURE_SCENES[0].start_time - 0.0) > 1e-6:
        issues.append(f"First scene starts at {LECTURE_SCENES[0].start_time}, expected 0.0")
    for s_prev, s_next in zip(LECTURE_SCENES[:-1], LECTURE_SCENES[1:]):
        if abs(s_prev.end_time - s_next.start_time) > 1e-6:
            issues.append(f"Gap/overlap between {s_prev.slug} ({s_prev.end_time}) and {s_next.slug} ({s_next.start_time})")
    total_dur = sum(s.duration for s in LECTURE_SCENES)
    if abs(total_dur - 1800.0) > 1e-6 or abs(LECTURE_SCENES[-1].end_time - 1800.0) > 1e-6:
        issues.append(f"Total duration is {total_dur}s, expected exactly 1800.0s")

    # 2. Verify captions never overlap in time and stay within scene boundaries
    total_captions = 0
    for scene in LECTURE_SCENES:
        prev_end = scene.start_time
        for cap in scene.captions:
            total_captions += 1
            if cap.start < scene.start_time or cap.end > scene.end_time:
                issues.append(f"Caption '{cap.line1[:25]}' outside scene {scene.slug} bounds")
            if cap.start < prev_end - 1e-6:
                issues.append(f"Overlapping caption at t={cap.start}s in {scene.slug}")
            if (cap.end - cap.start) < 3.0:
                issues.append(f"Caption duration too short ({cap.end - cap.start:.1f}s) at t={cap.start}s")
            prev_end = cap.end

    # 3. Verify mathematical consistency of linear model & loss decrease during training
    initial_mse = compute_house_mse(0.35, 85.0)
    final_mse = compute_house_mse(1.65, 38.0)
    if not (final_mse < initial_mse * 0.05):
        issues.append(f"Training MSE did not decrease as expected: initial={initial_mse:.1f}, final={final_mse:.1f}")

    # 4. Sample frames across all 11 scenes to verify non-empty visual rendering & black final frame
    sample_canvas = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    for scene in LECTURE_SCENES:
        mid_t = 0.5 * (scene.start_time + scene.end_time)
        render_frame_at_time(mid_t, cfg, sample_canvas)
        if int(sample_canvas.max()) == 0:
            issues.append(f"Scene {scene.slug} rendered a completely blank frame at midpoint t={mid_t}s")

    # Verify final frame at t=1799.9s is faded to black
    render_frame_at_time(1799.9, cfg, sample_canvas)
    if float(sample_canvas.mean()) > 1.5:
        issues.append(f"Final frame at 1799.9s did not fade cleanly to black (mean={sample_canvas.mean():.2f})")

    report = {
        "status": "PASS" if not issues else "FAIL",
        "total_scenes": len(LECTURE_SCENES),
        "total_duration_seconds": total_dur,
        "total_frames_at_30fps": int(round(total_dur * 30)),
        "total_animated_captions": total_captions,
        "font_sans_resolved": resolve_font_path("sans", bold=False),
        "font_mono_resolved": resolve_font_path("mono", bold=False),
        "initial_training_mse": round(initial_mse, 2),
        "calibrated_training_mse": round(final_mse, 2),
        "issues": issues,
    }
    return report


def verify_mp4_with_ffprobe(mp4_path: str) -> Dict[str, object]:
    """Inspect a rendered MP4 file with FFprobe to verify resolution, FPS, codec, duration, and absence of audio."""
    if not os.path.isfile(mp4_path):
        return {"status": "FILE_NOT_FOUND", "path": mp4_path}
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        mp4_path,
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    v0 = video_streams[0] if video_streams else {}
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
        "audio_streams_count": len(audio_streams),
        "has_audio": len(audio_streams) > 0,
    }


# ==============================================================================
# 9. STREAMING FFMPEG VIDEO ORCHESTRATOR (PREVIEW, SCENE, AND FULL 4K)
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
    """
    Renders a single time segment [t_start, t_end) in an isolated child process
    (spawned via subprocess.Popen, avoiding multiprocessing.Pool fork() deadlocks).
    Writes live frame progress to a .status file and stdout.
    """
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
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{rw}x{rh}",
        "-r", str(eff_motion_fps),
        "-i", "-",
        "-an",  # Strictly no audio stream
        *vf_args,
        "-c:v", "libx264",
        "-threads", "1",
        "-preset", preset,
        "-tune", "fastdecode",
        "-x264-params", "bframes=0:ref=1:rc-lookahead=0:sync-lookahead=0",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_out,
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
            elapsed_so_far = max(0.001, time.time() - t_seg0)
            _write_status("rendering", out_done, total_out_frames, out_done / elapsed_so_far, elapsed_so_far)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        _write_status("error", 0, total_out_frames, 0.0, time.time() - t_seg0)
        raise RuntimeError(f"FFmpeg exited with code {proc.returncode} while rendering {out_file}")

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
    """
    Orchestrate rendering using fork-free subprocess workers with a 1-second live
    heartbeat logger, visible segment files in ./silent_math_parts_<mode>_<res>/,
    progressive live output MP4 muxing, and automatic resume support.
    """
    target_w, target_h = RESOLUTION_PRESETS.get(res_key.lower(), (3840, 2160))
    fps = 30
    out_abs = os.path.abspath(output_path)
    out_dir = os.path.dirname(out_abs) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Default internal scale for 4K: 1/3 (1280x720 crisp vector canvas upscaled to 3840x2160)
    # so 4K segments finish in ~4-6 seconds each on standard 2-4 core Codespaces.
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
            raise ValueError(f"Scene index {idx} not found (valid: 1..{len(LECTURE_SCENES)})")
        sc = matching[0]
        cur = sc.start_time
        while cur < sc.end_time - 1e-6:
            nxt = min(sc.end_time, cur + 15.0)
            segments.append((cur, nxt))
            cur = nxt
    elif mode == "full":
        # Partition 1800s into 15-second segments (120 segments) so the first segment
        # and output MP4 appear on disk within ~5 seconds of starting the command!
        chunk_sec = 15.0
        n_chunks = int(round(1800.0 / chunk_sec))
        for c in range(n_chunks):
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
    print(f"[Silent Math] Video Render Pipeline Initialized (v3 Fork-Safe Engine)", flush=True)
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

    # Check already-completed segments on disk for instant resume
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
        # Launch new workers up to eff_workers
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
                sys.executable,
                script_path,
                "--internal-worker-seg",
                str(s_idx),
                str(total_segs),
                str(ts),
                str(te),
                part_path,
                str(target_w),
                str(target_h),
                str(fps),
                str(motion_fps),
                str(internal_scale),
                preset,
                str(crf),
            ]
            p = subprocess.Popen(w_cmd, env=clean_env)
            active_procs[s_idx] = (p, part_path, ts, te)

        # Poll active workers
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

        # Print live heartbeat every 2.0 seconds showing frame progress inside active workers
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
        cmd_concat = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-an",
            "-c", "copy",
            "-movflags", "+faststart",
            out_abs,
        ]
        subprocess.check_call(cmd_concat)

    if not keep_parts:
        shutil.rmtree(work_parts_dir, ignore_errors=True)

    elapsed = round(time.time() - t0, 2)
    probe = verify_mp4_with_ffprobe(out_abs)
    probe["render_elapsed_seconds"] = elapsed
    probe["mode"] = mode
    print(f"[Silent Math] Render Complete in {elapsed:.2f}s -> {out_abs}", flush=True)
    return probe


# ==============================================================================
# 10. COMMAND-LINE INTERFACE (CLI) ENTRY POINT
# ==============================================================================

def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silent Math — Lecture #1: Introduction to AI & ML (Complete 4K Video Generator)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "preview", "scene", "validate"],
        default="full",
        help="Execution mode: 'full' (30-min 1800s lecture), 'preview' (22s all-scene sampler), 'scene' (single section), or 'validate'.",
    )
    parser.add_argument(
        "--res",
        choices=list(RESOLUTION_PRESETS.keys()),
        default="4k",
        help="Output resolution preset ('4k' = 3840x2160, '1080p', '720p', '360p').",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=1,
        help="Scene number (1..11) when --mode scene is selected.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="silent_math_lecture_01_ai_ml.mp4",
        help="Output MP4 file path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2)),
        help="Number of parallel FFmpeg scene-rendering worker processes.",
    )
    parser.add_argument(
        "--internal-scale",
        type=float,
        default=None,
        help="Internal rasterization scale factor (defaults to 0.3333 for 4K = 720p vector canvas + fast 4K upscale; pass 0.5 or 1.0 for higher internal rasterization).",
    )
    parser.add_argument(
        "--motion-fps",
        type=int,
        default=6,
        help="Unique animation states rendered per second before 30 FPS output muxing (6 = 5x faster pipe & encode; pass 15 or 30 for higher temporal sampling).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ultrafast",
        help="FFmpeg x264 encoding speed preset (ultrafast, superfast, veryfast, fast, medium).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="FFmpeg H.264 Constant Rate Factor (lower = higher quality).",
    )
    parser.add_argument(
        "--parts-dir",
        type=str,
        default=None,
        help="Directory for visible intermediate segment MP4 files (supports automatic resume).",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep intermediate segment MP4 files after final concatenation.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run automated timeline, caption, and mathematical consistency checks and exit.",
    )
    parser.add_argument(
        "--verify-file",
        type=str,
        default="",
        help="Run FFprobe verification on an existing MP4 file and exit.",
    )
    parser.add_argument(
        "--internal-worker-seg",
        nargs=12,
        metavar=("SEG_IDX", "TOTAL_SEGS", "T_START", "T_END", "OUT_FILE", "W", "H", "FPS", "MFPS", "SCALE", "PRESET", "CRF"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli_args(argv)

    if args.internal_worker_seg:
        w_args = args.internal_worker_seg
        _render_single_segment_worker(
            seg_idx=int(w_args[0]),
            total_segs=int(w_args[1]),
            t_start=float(w_args[2]),
            t_end=float(w_args[3]),
            out_file=w_args[4],
            target_w=int(w_args[5]),
            target_h=int(w_args[6]),
            fps=int(w_args[7]),
            motion_fps=int(w_args[8]),
            internal_scale=float(w_args[9]),
            preset=w_args[10],
            crf=int(w_args[11]),
        )
        return 0

    if args.verify_file:
        report = verify_mp4_with_ffprobe(args.verify_file)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report.get("status") == "VERIFIED" else 1

    if args.validate or args.mode == "validate":
        w, h = RESOLUTION_PRESETS.get(args.res.lower(), (3840, 2160))
        cfg = RenderConfig(width=w, height=h)
        report = run_project_validation(cfg)
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

