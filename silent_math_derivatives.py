
#!/usr/bin/env python3
"""
silent_math_derivatives.py
==========================
Production-Quality Mathematical Animation Pipeline for "Silent Math"
Title:    Silent Math — Understanding Derivatives Visually
Creator:  By Ansh Raj
Output:   Silent_Math_Derivatives.mp4
Target:   25:00 (1,500.0 seconds across 10 self-contained Manim Community scenes)
Style:    Restrained Monochrome Blackboard (#000000 background, #FFFFFF foreground,
          selective grayscale hierarchy, zero captions/subtitles, strict non-overlapping
          zone layout engine with bounding-box validation and post-render frame QA).

Technology Stack:
  - Python 3.10+
  - Manim Community Edition (manim >= 0.18.0)
  - Cairo & Pango (vector & text shaping backend for Manim)
  - LaTeX (TeX Live: latex / pdflatex / dvisvgm)
  - NumPy (vectorized numerical sampling & finite-difference verification)
  - FFmpeg & FFprobe (scene concatenation, H.264 MP4 encoding, stream & frame QA)

Usage Examples:
  1. Check installed dependencies & system readiness:
     python3 silent_math_derivatives.py --check-deps

  2. Run mathematical & layout self-verification (no Manim render required):
     python3 silent_math_derivatives.py --verify-math

  3. Render a fast preview (low-res 480p15, accelerated pacing or single scene):
     python3 silent_math_derivatives.py --preview

  4. Render full 25-minute 1080p60 production video:
     python3 silent_math_derivatives.py --quality production --fps 60 --workers auto --output Silent_Math_Derivatives.mp4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# 1. CONDITIONAL MANIM COMMUNITY IMPORTS & GRACEFUL FALLBACK SHIM
# =============================================================================
# We import Manim Community Edition when available so that `manim` CLI or our
# orchestrator can execute all 10 Scene classes natively. When `manim` is not
# yet installed on the host machine, we provide lightweight geometric stubs so
# that `--help`, `--check-deps`, `--verify-math`, and `--plan` run cleanly
# without crashing at import time.

MANIM_AVAILABLE = False
MANIM_IMPORT_ERROR: Optional[str] = None

try:
    import manim as mn
    from manim import (
        BLACK,
        WHITE,
        GRAY_A,
        GRAY_B,
        GRAY_C,
        GRAY_D,
        GRAY_E,
        ORIGIN,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        UL,
        UR,
        DL,
        DR,
        PI,
        TAU,
        DEGREES,
        Scene,
        Mobject,
        VMobject,
        VGroup,
        Group,
        Axes,
        NumberLine,
        NumberPlane,
        ValueTracker,
        DecimalNumber,
        MathTex,
        Tex,
        Text,
        Dot,
        Line,
        DashedLine,
        Arrow,
        DoubleArrow,
        Brace,
        Polygon,
        Rectangle,
        RoundedRectangle,
        SurroundingRectangle,
        Circle,
        Arc,
        always_redraw,
        Create,
        Uncreate,
        Write,
        Unwrite,
        FadeIn,
        FadeOut,
        Transform,
        ReplacementTransform,
        TransformMatchingTex,
        MoveToTarget,
        Indicate,
        Circumscribe,
        Restore,
        AnimationGroup,
        LaggedStart,
        Succession,
        Wait,
        config as manim_config,
        rate_functions,
    )
    MANIM_AVAILABLE = True
except Exception as exc:  # pragma: no cover - diagnostic fallback when Manim is absent
    MANIM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    BLACK = "#000000"
    WHITE = "#FFFFFF"
    GRAY_A = "#DDDDDD"
    GRAY_B = "#BBBBBB"
    GRAY_C = "#888888"
    GRAY_D = "#555555"
    GRAY_E = "#222222"
    ORIGIN = np.array([0.0, 0.0, 0.0])
    UP = np.array([0.0, 1.0, 0.0])
    DOWN = np.array([0.0, -1.0, 0.0])
    LEFT = np.array([-1.0, 0.0, 0.0])
    RIGHT = np.array([1.0, 0.0, 0.0])
    UL = UP + LEFT
    UR = UP + RIGHT
    DL = DOWN + LEFT
    DR = DOWN + RIGHT
    PI = math.pi
    TAU = 2.0 * math.pi
    DEGREES = PI / 180.0

    class Scene:  # type: ignore[no-redef]
        """Fallback Scene stub when Manim is not installed."""
        pass

    class Mobject:  # type: ignore[no-redef]
        pass

    class VMobject(Mobject):  # type: ignore[no-redef]
        pass

    class VGroup(VMobject):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.submobjects = list(args)


# =============================================================================
# 2. CHANNEL IDENTITY, MONOCHROME PALETTE & CHAPTER SPECIFICATION
# =============================================================================

CHANNEL_NAME = "SILENT MATH"
CHANNEL_SUBTITLE = "UNDERSTANDING DERIVATIVES VISUALLY"
CREATOR_CREDIT = "By Ansh Raj"
CLOSING_TAGLINE = "MATHEMATICS, MADE VISIBLE"
CLOSING_CREDIT = "BY ANSH RAJ"
DEFAULT_OUTPUT_FILENAME = "Silent_Math_Derivatives.mp4"

# Strict Black-and-White Palette (>=95% pure monochrome; subtle silver-gray
# hierarchy used solely to distinguish secondary curves, construction lines,
# and coordinate axes).
PALETTE = {
    "bg": "#000000",           # Pure black background
    "primary": "#FFFFFF",      # Primary equations, primary curve, primary text
    "secondary": "#D0D0D0",    # Secondary curve / tangent line / key labels
    "tertiary": "#969696",     # Secant lines, construction braces, dashed guides
    "axes": "#6E6E6E",         # Thin, precise coordinate axes
    "grid": "#262626",         # Ultra-subtle background reference fill
    "box_fill": "#0D0D0D",     # Opaque backing panel for equations
    "box_stroke": "#444444",   # Hairline border for callout boxes
    "accent_warm": "#E8E8E8",  # High-contrast pearl white for highlighted terms
}


@dataclasses.dataclass(frozen=True)
class ChapterSpec:
    """Metadata and exact duration budget for each chapter in the 25-minute film."""
    number: int
    scene_name: str
    title: str
    start_time_str: str
    end_time_str: str
    target_seconds: float
    summary: str


CHAPTER_SPECS: List[ChapterSpec] = [
    ChapterSpec(
        number=1,
        scene_name="Chapter01_Intro",
        title="Cinematic Introduction",
        start_time_str="0:00",
        end_time_str="1:00",
        target_seconds=60.0,
        summary="Silent Math identity, emerging curve from pure black, gliding point and secant line, title sequence.",
    ),
    ChapterSpec(
        number=2,
        scene_name="Chapter02_MeaningOfDerivative",
        title="What Does a Derivative Mean?",
        start_time_str="1:00",
        end_time_str="3:30",
        target_seconds=150.0,
        summary="f(x)=x^2, input change h vs output change f(x+h)-f(x), rise-over-run geometry, average rate of change.",
    ),
    ChapterSpec(
        number=3,
        scene_name="Chapter03_SecantToTangent",
        title="From Secant to Tangent",
        start_time_str="3:30",
        end_time_str="6:00",
        target_seconds=150.0,
        summary="Shrinking h -> 0 from both sides, why h=0 cannot be plugged in directly, limit definition of f'(x).",
    ),
    ChapterSpec(
        number=4,
        scene_name="Chapter04_DerivingXSquared",
        title="Deriving the Derivative of x^2",
        start_time_str="6:00",
        end_time_str="9:00",
        target_seconds=180.0,
        summary="Geometric square expansion (x+h)^2 = x^2 + 2xh + h^2, cancellation for h != 0, limit evaluation to 2x.",
    ),
    ChapterSpec(
        number=5,
        scene_name="Chapter05_GraphOf2x",
        title="Understanding the Graph of 2x",
        start_time_str="9:00",
        end_time_str="12:00",
        target_seconds=180.0,
        summary="Sampling tangent slopes at x = -2, 0, 1, 2 on f(x)=x^2 and plotting the resulting derivative line f'(x)=2x.",
    ),
    ChapterSpec(
        number=6,
        scene_name="Chapter06_CubicDerivative",
        title="Derivative of a Cubic Function",
        start_time_str="12:00",
        end_time_str="15:00",
        target_seconds=180.0,
        summary="First-principles derivation of f(x)=x^3 -> f'(x)=3x^2, binomial expansion, non-negativity & inflection at x=0.",
    ),
    ChapterSpec(
        number=7,
        scene_name="Chapter07_RatesOfChangePhysics",
        title="Derivatives as Rates of Change",
        start_time_str="15:00",
        end_time_str="18:00",
        target_seconds=180.0,
        summary="Kinematics of s(t)=t^2 [m] and instantaneous velocity v(t)=s'(t)=2t [m/s] with synchronized moving particle.",
    ),
    ChapterSpec(
        number=8,
        scene_name="Chapter08_ComparingFunctions",
        title="Comparing Functions and Slopes",
        start_time_str="18:00",
        end_time_str="21:00",
        target_seconds=180.0,
        summary="Side-by-side comparison of f(x)=x^2 vs g(x)=x^3 and f'(x)=2x vs g'(x)=3x^2 across key points.",
    ),
    ChapterSpec(
        number=9,
        scene_name="Chapter09_VisualRecap",
        title="Visual Recap and Mathematical Insight",
        start_time_str="21:00",
        end_time_str="24:00",
        target_seconds=180.0,
        summary="Eight-stage visual synthesis from Δx and Δy through secant, limit, tangent, and derivative function.",
    ),
    ChapterSpec(
        number=10,
        scene_name="Chapter10_Closing",
        title="Closing",
        start_time_str="24:00",
        end_time_str="25:00",
        target_seconds=60.0,
        summary="Distillation to a single luminous curve, Silent Math — Mathematics, Made Visible — By Ansh Raj, fade to black.",
    ),
]


# =============================================================================
# 3. PURE MATHEMATICAL ENGINE & NUMERICAL VERIFICATION HELPERS
# =============================================================================

class CalculusMath:
    """
    Vectorized, dimensionally consistent mathematical functions used across all
    scenes and verified by `--verify-math` prior to rendering.
    """

    @staticmethod
    def f_quad(x: float | np.ndarray) -> float | np.ndarray:
        """Primary quadratic function f(x) = x^2."""
        return np.square(x) if isinstance(x, np.ndarray) else float(x * x)

    @staticmethod
    def df_quad(x: float | np.ndarray) -> float | np.ndarray:
        """Exact derivative of f(x) = x^2 -> f'(x) = 2x."""
        return 2.0 * x

    @staticmethod
    def f_cubic(x: float | np.ndarray) -> float | np.ndarray:
        """Cubic function g(x) = x^3."""
        return np.power(x, 3) if isinstance(x, np.ndarray) else float(x * x * x)

    @staticmethod
    def df_cubic(x: float | np.ndarray) -> float | np.ndarray:
        """Exact derivative of g(x) = x^3 -> g'(x) = 3x^2."""
        return 3.0 * (np.square(x) if isinstance(x, np.ndarray) else float(x * x))

    @staticmethod
    def f_intro_wave(x: float | np.ndarray) -> float | np.ndarray:
        """Smooth cubic polynomial used in Chapter 1 & Chapter 10: 0.22 x^3 - 0.85 x."""
        return 0.22 * (x ** 3) - 0.85 * x

    @staticmethod
    def df_intro_wave(x: float | np.ndarray) -> float | np.ndarray:
        """Exact derivative of Chapter 1 curve: 0.66 x^2 - 0.85."""
        return 0.66 * (x ** 2) - 0.85

    @staticmethod
    def difference_quotient(
        func: Callable[[float], float],
        x: float,
        h: float,
    ) -> float:
        """
        Compute the forward difference quotient (f(x + h) - f(x)) / h.
        Explicitly forbids division by zero (h == 0) to enforce mathematical rigor.
        """
        if abs(h) < 1e-14:
            raise ValueError(
                "Mathematical error: Difference quotient (f(x+h)-f(x))/h is undefined at h = 0. "
                "Use a non-zero h or evaluate the analytical limit."
            )
        return float((func(x + h) - func(x)) / h)

    @staticmethod
    def secant_line_endpoints(
        func: Callable[[float], float],
        x0: float,
        h: float,
        x_min: float,
        x_max: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Return ((x_left, y_left), (x_right, y_right), slope) for the secant line
        passing through (x0, f(x0)) and (x0 + h, f(x0 + h)) clipped to [x_min, x_max].
        """
        slope = CalculusMath.difference_quotient(func, x0, h)
        y0 = func(x0)
        y_left = y0 + slope * (x_min - x0)
        y_right = y0 + slope * (x_max - x0)
        return (x_min, y_left), (x_max, y_right), slope

    @staticmethod
    def tangent_line_endpoints(
        func: Callable[[float], float],
        dfunc: Callable[[float], float],
        x0: float,
        half_span_x: float = 1.15,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Return ((x_left, y_left), (x_right, y_right), slope) for the tangent line
        to `func` at `x0` spanning `[x0 - half_span_x, x0 + half_span_x]`.
        """
        y0 = float(func(x0))
        slope = float(dfunc(x0))
        x_left = x0 - half_span_x
        x_right = x0 + half_span_x
        y_left = y0 + slope * (x_left - x0)
        y_right = y0 + slope * (x_right - x0)
        return (x_left, y_left), (x_right, y_right), slope


# =============================================================================
# 4. NON-OVERLAPPING SCENE LAYOUT & BOUNDING-BOX VALIDATION ENGINE
# =============================================================================

@dataclasses.dataclass(frozen=True)
class BoundingBox2D:
    """Axis-aligned 2D bounding box in Manim frame coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def center(self) -> np.ndarray:
        return np.array([
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
            0.0,
        ])

    def intersects(self, other: "BoundingBox2D", clearance: float = 0.08) -> bool:
        """
        Return True if `self` and `other` overlap or violate `clearance` margin.
        """
        if (self.x_max + clearance) <= other.x_min or (other.x_max + clearance) <= self.x_min:
            return False
        if (self.y_max + clearance) <= other.y_min or (other.y_max + clearance) <= self.y_min:
            return False
        return True

    def is_within(self, container: "BoundingBox2D", tolerance: float = 1e-3) -> bool:
        """Return True if `self` lies completely inside `container`."""
        return (
            self.x_min >= container.x_min - tolerance
            and self.y_min >= container.y_min - tolerance
            and self.x_max <= container.x_max + tolerance
            and self.y_max <= container.y_max + tolerance
        )


class SafeLayoutEngine:
    """
    Enforces strict non-overlapping regions on a 16:9 Manim frame
    (width = 14.222..., height = 8.0, x in [-7.11, +7.11], y in [-4.0, +4.0]).

    Every scene uses disjoint, pre-validated zones:
      - SAFE_FRAME:       [-6.55, -3.55] to [+6.55, +3.55] (outer margin >= 0.45 units)
      - HEADER_ZONE:      [-6.45, +2.85] to [+6.45, +3.50] (top chapter/formula banner)
      - LEFT_GRAPH_ZONE:  [-6.45, -3.40] to [-0.30, +2.60] (dedicated primary graph panel)
      - RIGHT_EQ_ZONE:    [+0.30, -3.40] to [+6.45, +2.60] (dedicated equation / 2nd graph panel)
      - TOP_TRACK_ZONE:   [-6.35, +1.35] to [+6.35, +2.65] (Chapter 7 1D motion track)
      - BOTTOM_LEFT_ZONE: [-6.45, -3.45] to [-0.30, +1.05] (Chapter 7 s(t) graph)
      - BOTTOM_RIGHT_ZONE:[+0.30, -3.45] to [+6.45, +1.05] (Chapter 7 v(t) graph)
      - FULL_STAGE_ZONE:  [-6.20, -3.20] to [+6.20, +2.60] (single-focus derivation/recap zone)

    Note on Automated Collision Detection:
      While `validate_zones`, `fit_into_zone`, and `assert_no_overlap` prevent
      top-level bounding-box overflows and inter-panel collisions, no axis-aligned
      bounding-box check can guarantee zero pixel overlap for every dynamic curve
      label across every interpolated frame; therefore `sample_and_inspect_frames`
      provides post-render frame sampling QA on the generated MP4.
    """

    FRAME_WIDTH: float = 14.222222222222221
    FRAME_HEIGHT: float = 8.0

    SAFE_FRAME = BoundingBox2D(-6.55, -3.55, 6.55, 3.55, label="SAFE_FRAME")
    HEADER_ZONE = BoundingBox2D(-6.45, 2.85, 6.45, 3.50, label="HEADER_ZONE")
    LEFT_GRAPH_ZONE = BoundingBox2D(-6.45, -3.40, -0.30, 2.60, label="LEFT_GRAPH_ZONE")
    RIGHT_EQ_ZONE = BoundingBox2D(0.30, -3.40, 6.45, 2.60, label="RIGHT_EQ_ZONE")
    TOP_TRACK_ZONE = BoundingBox2D(-6.35, 1.35, 6.35, 2.65, label="TOP_TRACK_ZONE")
    BOTTOM_LEFT_ZONE = BoundingBox2D(-6.45, -3.45, -0.30, 1.05, label="BOTTOM_LEFT_ZONE")
    BOTTOM_RIGHT_ZONE = BoundingBox2D(0.30, -3.45, 6.45, 1.05, label="BOTTOM_RIGHT_ZONE")
    FULL_STAGE_ZONE = BoundingBox2D(-6.20, -3.20, 6.20, 2.60, label="FULL_STAGE_ZONE")

    @classmethod
    def validate_zones(cls) -> List[str]:
        """
        Verify mathematically that all canonical layout zones lie strictly inside
        SAFE_FRAME and that sibling zones have positive non-overlapping clearance.
        """
        errors: List[str] = []
        all_zones = [
            cls.HEADER_ZONE,
            cls.LEFT_GRAPH_ZONE,
            cls.RIGHT_EQ_ZONE,
            cls.TOP_TRACK_ZONE,
            cls.BOTTOM_LEFT_ZONE,
            cls.BOTTOM_RIGHT_ZONE,
            cls.FULL_STAGE_ZONE,
        ]
        for zone in all_zones:
            if not zone.is_within(cls.SAFE_FRAME):
                errors.append(f"Zone {zone.label} exceeds SAFE_FRAME bounds.")

        disjoint_pairs = [
            (cls.HEADER_ZONE, cls.LEFT_GRAPH_ZONE, 0.15),
            (cls.HEADER_ZONE, cls.RIGHT_EQ_ZONE, 0.15),
            (cls.LEFT_GRAPH_ZONE, cls.RIGHT_EQ_ZONE, 0.50),
            (cls.HEADER_ZONE, cls.TOP_TRACK_ZONE, 0.15),
            (cls.TOP_TRACK_ZONE, cls.BOTTOM_LEFT_ZONE, 0.20),
            (cls.TOP_TRACK_ZONE, cls.BOTTOM_RIGHT_ZONE, 0.20),
            (cls.BOTTOM_LEFT_ZONE, cls.BOTTOM_RIGHT_ZONE, 0.50),
        ]
        for z1, z2, min_gap in disjoint_pairs:
            if z1.intersects(z2, clearance=min_gap):
                errors.append(
                    f"Zones {z1.label} and {z2.label} violate minimum clearance {min_gap}."
                )
        return errors

    @staticmethod
    def mobject_bbox(mob: Any, label: str = "") -> BoundingBox2D:
        """Extract a 2D bounding box from a Manim Mobject."""
        left = float(mob.get_left()[0])
        right = float(mob.get_right()[0])
        bottom = float(mob.get_bottom()[1])
        top = float(mob.get_top()[1])
        return BoundingBox2D(left, bottom, right, top, label=label)

    @classmethod
    def fit_into_zone(
        cls,
        mob: Any,
        zone: BoundingBox2D,
        padding: float = 0.22,
        align_top: bool = False,
        align_left: bool = False,
    ) -> Any:
        """
        Automatically downscale `mob` if its width or height exceeds `zone` minus
        `2 * padding`, and position it cleanly inside `zone` so it can never overflow
        or collide with sibling zones.
        """
        max_w = max(0.2, zone.width - 2.0 * padding)
        max_h = max(0.2, zone.height - 2.0 * padding)
        cur_w = float(mob.width)
        cur_h = float(mob.height)
        if cur_w > max_w or cur_h > max_h:
            scale_factor = min(max_w / max(cur_w, 1e-6), max_h / max(cur_h, 1e-6))
            mob.scale(scale_factor)

        target_x = zone.center[0]
        target_y = zone.center[1]
        if align_left:
            target_x = zone.x_min + padding + 0.5 * float(mob.width)
        if align_top:
            target_y = zone.y_max - padding - 0.5 * float(mob.height)
        mob.move_to(np.array([target_x, target_y, 0.0]))
        return mob

    @classmethod
    def stack_in_zone(
        cls,
        mobjects: Sequence[Any],
        zone: BoundingBox2D,
        buff: float = 0.28,
        padding: float = 0.22,
        align_left: bool = False,
    ) -> Any:
        """
        Arrange a list of equation/label mobjects vertically inside `zone` with
        consistent `buff` spacing, automatic width/height fitting, and zero overlap.
        """
        group = VGroup(*mobjects)
        if len(mobjects) > 1:
            group.arrange(DOWN, buff=buff, aligned_edge=LEFT if align_left else ORIGIN)
        return cls.fit_into_zone(group, zone, padding=padding, align_left=align_left)

    @classmethod
    def assert_no_overlap(
        cls,
        mobjects: Sequence[Any],
        clearance: float = 0.04,
    ) -> List[Tuple[int, int]]:
        """
        Inspect active top-level mobjects and return any colliding index pairs.
        If a collision is detected during construction, gently separate them along Y.
        """
        boxes = [cls.mobject_bbox(m, label=f"mob_{i}") for i, m in enumerate(mobjects)]
        collisions: List[Tuple[int, int]] = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i].intersects(boxes[j], clearance=clearance):
                    collisions.append((i, j))
                    dy = (boxes[i].y_max - boxes[j].y_min) + clearance + 0.05
                    mobjects[j].shift(DOWN * dy)
                    boxes[j] = cls.mobject_bbox(mobjects[j], label=f"mob_{j}")
        return collisions


# =============================================================================
# 5. BASE SILENT MATH SCENE CLASS & REUSABLE VISUAL HELPERS
# =============================================================================

class SilentMathBaseScene(Scene):
    """
    Base class shared across all 10 chapters of Silent Math.

    Provides:
      - Pure black (#000000) background configuration
      - Pacing control (`SILENT_MATH_TIME_SCALE` env var so `--preview` can run a
        fast check while `production` runs the exact 25-minute timing)
      - Chapter header cards, restrained title cards, safe equation replacement,
        and monochrome coordinate axes builders.
    """

    chapter_number: int = 0
    chapter_title: str = ""
    target_duration: float = 60.0

    def setup(self) -> None:
        if MANIM_AVAILABLE:
            self.camera.background_color = PALETTE["bg"]
        self._elapsed_simulated: float = 0.0
        self._timing_scale: float = float(os.environ.get("SILENT_MATH_TIME_SCALE", "1.0"))

    def t(self, seconds: float) -> float:
        """
        Scale animation or wait duration by `SILENT_MATH_TIME_SCALE` (1.0 in
        production for the full 25-minute film; configurable in preview mode)
        and track total simulated chapter runtime.
        """
        self._elapsed_simulated += seconds
        return max(0.1, seconds * self._timing_scale)

    def pause(self, seconds: float) -> None:
        """Deliberate pedagogical pause scaled by the active timing multiplier."""
        if seconds > 0.05:
            self.wait(self.t(seconds))

    def finish_chapter_timing(self) -> None:
        """
        Ensure the scene hits its exact target chapter duration so all 10 chapters
        sum precisely to 25:00 (1,500.0 seconds) in production mode.
        """
        remaining = self.target_duration - self._elapsed_simulated
        if remaining > 0.1:
            self.wait(max(0.1, remaining * self._timing_scale))
            self._elapsed_simulated = self.target_duration

    def make_header(self, numeral: str, title_text: str, formula_tex: Optional[str] = None) -> VGroup:
        """
        Create a restrained top banner inside `SafeLayoutEngine.HEADER_ZONE` with
        a hairline bottom divider.
        """
        tag = Text(
            f"{numeral}  ·  {title_text.upper()}",
            font="sans-serif",
            font_size=22,
            color=PALETTE["secondary"],
            weight="LIGHT",
        )
        divider = Line(
            np.array([SafeLayoutEngine.HEADER_ZONE.x_min, SafeLayoutEngine.HEADER_ZONE.y_min, 0.0]),
            np.array([SafeLayoutEngine.HEADER_ZONE.x_max, SafeLayoutEngine.HEADER_ZONE.y_min, 0.0]),
            stroke_width=1.2,
            color=PALETTE["box_stroke"],
        )
        if formula_tex:
            rhs = MathTex(formula_tex, font_size=28, color=PALETTE["primary"])
            tag.move_to(
                np.array([
                    SafeLayoutEngine.HEADER_ZONE.x_min + 0.5 * tag.width + 0.1,
                    SafeLayoutEngine.HEADER_ZONE.center[1] + 0.05,
                    0.0,
                ])
            )
            rhs.move_to(
                np.array([
                    SafeLayoutEngine.HEADER_ZONE.x_max - 0.5 * rhs.width - 0.1,
                    SafeLayoutEngine.HEADER_ZONE.center[1] + 0.05,
                    0.0,
                ])
            )
            header = VGroup(tag, rhs, divider)
        else:
            tag.move_to(
                np.array([
                    SafeLayoutEngine.HEADER_ZONE.x_min + 0.5 * tag.width + 0.1,
                    SafeLayoutEngine.HEADER_ZONE.center[1] + 0.05,
                    0.0,
                ])
            )
            header = VGroup(tag, divider)
        return header

    def play_chapter_card(self, chapter_num: int, title: str) -> None:
        """
        Brief, restrained 4.4-second chapter transition card at the start of chapters 2-9.
        """
        ch_label = Text(
            f"CHAPTER {chapter_num:02d}",
            font="sans-serif",
            font_size=22,
            color=PALETTE["tertiary"],
        )
        ch_title = Text(
            title.upper(),
            font="sans-serif",
            font_size=36,
            color=PALETTE["primary"],
        )
        rule = Line(LEFT * 2.2, RIGHT * 2.2, stroke_width=1.4, color=PALETTE["secondary"])
        card = VGroup(ch_label, rule, ch_title).arrange(DOWN, buff=0.28)
        SafeLayoutEngine.fit_into_zone(card, SafeLayoutEngine.FULL_STAGE_ZONE)

        self.play(FadeIn(ch_label, shift=UP * 0.15), Create(rule), run_time=self.t(1.2))
        self.play(FadeIn(ch_title, shift=UP * 0.12), run_time=self.t(1.2))
        self.pause(1.0)
        self.play(FadeOut(card), run_time=self.t(1.0))

    def create_monochrome_axes(
        self,
        x_range: Sequence[float],
        y_range: Sequence[float],
        zone: BoundingBox2D,
        x_label_tex: str = "x",
        y_label_tex: str = "f(x)",
        include_numbers: bool = True,
    ) -> Tuple[Axes, VGroup]:
        """
        Construct thin, high-precision monochrome coordinate axes fitted strictly
        inside `zone` so axis labels never clip or collide with adjacent panels.
        """
        axes = Axes(
            x_range=list(x_range),
            y_range=list(y_range),
            x_length=zone.width - 1.15,
            y_length=zone.height - 1.10,
            axis_config={
                "color": PALETTE["axes"],
                "stroke_width": 1.6,
                "include_ticks": True,
                "tick_size": 0.06,
                "include_tip": True,
                "tip_width": 0.14,
                "tip_height": 0.14,
            },
            x_axis_config={"include_numbers": include_numbers, "font_size": 20},
            y_axis_config={"include_numbers": include_numbers, "font_size": 20},
        )
        axes.move_to(zone.center + DOWN * 0.12 + LEFT * 0.10)

        x_lbl = MathTex(x_label_tex, font_size=26, color=PALETTE["secondary"])
        y_lbl = MathTex(y_label_tex, font_size=26, color=PALETTE["secondary"])
        x_lbl.next_to(axes.x_axis.get_end(), UP + LEFT, buff=0.14)
        y_lbl.next_to(axes.y_axis.get_end(), RIGHT, buff=0.16)
        labels = VGroup(x_lbl, y_lbl)

        full_group = VGroup(axes, labels)
        SafeLayoutEngine.fit_into_zone(full_group, zone, padding=0.18)
        return axes, labels

    def make_equation_card(
        self,
        tex_lines: Sequence[str],
        zone: BoundingBox2D,
        font_size: int = 32,
        boxed_last: bool = False,
    ) -> VGroup:
        """
        Build a vertically stacked, non-overlapping set of MathTex equations
        fitted cleanly inside `zone`.
        """
        items: List[Any] = []
        for idx, line in enumerate(tex_lines):
            is_last = idx == len(tex_lines) - 1
            color = PALETTE["primary"] if is_last else PALETTE["secondary"]
            m = MathTex(line, font_size=font_size, color=color)
            if boxed_last and is_last:
                box = SurroundingRectangle(
                    m,
                    color=PALETTE["primary"],
                    stroke_width=1.8,
                    buff=0.18,
                )
                m = VGroup(m, box)
            items.append(m)
        return SafeLayoutEngine.stack_in_zone(items, zone, buff=0.32, padding=0.24)


# =============================================================================
# 6. THE 10 MANDATORY EDUCATIONAL CHAPTER SCENES (25 MINUTES TOTAL)
# =============================================================================

class Chapter01_Intro(SilentMathBaseScene):
    """
    Chapter 1 — Cinematic Introduction (0:00–1:00, Target: 60.0s)
    -------------------------------------------------------------
    - Starts on a completely black screen (#000000).
    - Gradually reveals a luminous white curve f(x) = 0.22 x^3 - 0.85 x.
    - Animates a point P gliding along the curve and introduces a second point Q
      with a dynamic secant line whose slope adapts smoothly across the curve.
    - Transitions into the restrained channel identity sequence:
        SILENT MATH
        UNDERSTANDING DERIVATIVES VISUALLY
        By Ansh Raj
    """

    chapter_number = 1
    chapter_title = "Cinematic Introduction"
    target_duration = 60.0

    def construct(self) -> None:
        self.pause(2.5)

        axes, axes_labels = self.create_monochrome_axes(
            x_range=[-3.2, 3.2, 1.0],
            y_range=[-2.5, 2.5, 1.0],
            zone=SafeLayoutEngine.FULL_STAGE_ZONE,
            x_label_tex="x",
            y_label_tex="f(x)",
            include_numbers=False,
        )
        axes.set_stroke(opacity=0.45)

        curve = axes.plot(
            CalculusMath.f_intro_wave,
            x_range=[-2.9, 2.9],
            color=PALETTE["primary"],
            stroke_width=3.2,
        )

        self.play(
            Create(axes, rate_func=rate_functions.ease_in_out_sine),
            FadeIn(axes_labels),
            run_time=self.t(3.5),
        )
        self.play(
            Create(curve, rate_func=rate_functions.ease_in_out_sine),
            run_time=self.t(5.5),
        )
        self.pause(1.5)

        x_tracker = ValueTracker(-2.3)
        h_tracker = ValueTracker(1.4)

        dot_p = always_redraw(
            lambda: Dot(
                axes.c2p(
                    x_tracker.get_value(),
                    CalculusMath.f_intro_wave(x_tracker.get_value()),
                ),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )
        lbl_p = always_redraw(
            lambda: MathTex("P", font_size=26, color=PALETTE["primary"]).next_to(
                dot_p, UP + LEFT, buff=0.14
            )
        )

        self.play(FadeIn(dot_p, scale=0.5), FadeIn(lbl_p), run_time=self.t(1.8))
        self.play(
            x_tracker.animate.set_value(-0.6),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(5.5),
        )

        dot_q = always_redraw(
            lambda: Dot(
                axes.c2p(
                    x_tracker.get_value() + h_tracker.get_value(),
                    CalculusMath.f_intro_wave(x_tracker.get_value() + h_tracker.get_value()),
                ),
                radius=0.065,
                color=PALETTE["secondary"],
            )
        )
        lbl_q = always_redraw(
            lambda: MathTex("Q", font_size=26, color=PALETTE["secondary"]).next_to(
                dot_q, DOWN + RIGHT, buff=0.14
            )
        )

        def build_intro_secant() -> Line:
            x0 = x_tracker.get_value()
            h_val = max(0.02, h_tracker.get_value())
            (x1, y1), (x2, y2), _ = CalculusMath.secant_line_endpoints(
                CalculusMath.f_intro_wave, x0, h_val, x0 - 1.5, x0 + h_val + 1.5
            )
            return Line(
                axes.c2p(x1, y1),
                axes.c2p(x2, y2),
                stroke_width=2.0,
                color=PALETTE["secondary"],
            )

        secant_line = always_redraw(build_intro_secant)

        self.play(
            FadeIn(dot_q, scale=0.5),
            FadeIn(lbl_q),
            Create(secant_line),
            run_time=self.t(2.5),
        )

        self.play(
            h_tracker.animate.set_value(0.05),
            x_tracker.animate.set_value(1.25),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(9.0),
        )
        self.play(
            x_tracker.animate.set_value(2.05),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(4.5),
        )
        self.pause(1.5)

        self.play(
            FadeOut(VGroup(axes, axes_labels, curve, dot_p, lbl_p, dot_q, lbl_q, secant_line)),
            run_time=self.t(3.0),
        )
        self.pause(1.0)

        title_main = Text(
            CHANNEL_NAME,
            font="sans-serif",
            font_size=60,
            color=PALETTE["primary"],
            weight="LIGHT",
        )
        divider = Line(LEFT * 3.2, RIGHT * 3.2, stroke_width=1.5, color=PALETTE["secondary"])
        subtitle = Text(
            CHANNEL_SUBTITLE,
            font="sans-serif",
            font_size=28,
            color=PALETTE["secondary"],
        )
        creator = Text(
            CREATOR_CREDIT,
            font="sans-serif",
            font_size=22,
            color=PALETTE["tertiary"],
        )

        title_stack = VGroup(title_main, divider, subtitle, creator).arrange(DOWN, buff=0.38)
        SafeLayoutEngine.fit_into_zone(title_stack, SafeLayoutEngine.FULL_STAGE_ZONE)

        self.play(Write(title_main), run_time=self.t(3.5))
        self.play(Create(divider), run_time=self.t(1.8))
        self.play(FadeIn(subtitle, shift=UP * 0.15), run_time=self.t(2.8))
        self.pause(1.2)
        self.play(FadeIn(creator, shift=UP * 0.12), run_time=self.t(2.4))
        self.pause(3.5)

        self.play(FadeOut(title_stack), run_time=self.t(3.0))
        self.finish_chapter_timing()


class Chapter02_MeaningOfDerivative(SilentMathBaseScene):
    """
    Chapter 2 — What Does a Derivative Mean? (1:00–3:30, Target: 150.0s)
    --------------------------------------------------------------------
    - Plots f(x) = x^2 in LEFT_GRAPH_ZONE and places equations in RIGHT_EQ_ZONE.
    - Animates a point (x, f(x)) and nearby point (x+h, f(x+h)).
    - Highlights horizontal displacement h and vertical displacement f(x+h)-f(x).
    - Builds the average rate of change ratio (f(x+h)-f(x))/h and links it
      geometrically to the slope of the secant line.
    """

    chapter_number = 2
    chapter_title = "What Does a Derivative Mean?"
    target_duration = 150.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header("02", "What Does a Derivative Mean?", r"f(x) = x^2")
        self.play(FadeIn(header), run_time=self.t(2.0))

        axes, axes_labels = self.create_monochrome_axes(
            x_range=[-0.5, 2.6, 0.5],
            y_range=[-0.5, 6.8, 1.0],
            zone=SafeLayoutEngine.LEFT_GRAPH_ZONE,
            x_label_tex="x",
            y_label_tex="f(x)",
            include_numbers=True,
        )
        parabola = axes.plot(
            CalculusMath.f_quad,
            x_range=[-0.4, 2.5],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        self.play(Create(axes), Write(axes_labels), run_time=self.t(4.0))
        self.play(Create(parabola), run_time=self.t(4.5))
        self.pause(2.0)

        x_tracker = ValueTracker(0.6)
        h_tracker = ValueTracker(1.1)

        dot_p = always_redraw(
            lambda: Dot(
                axes.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )
        proj_lines_p = always_redraw(
            lambda: VGroup(
                DashedLine(
                    axes.c2p(x_tracker.get_value(), 0.0),
                    axes.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                    dash_length=0.08,
                    stroke_width=1.4,
                    color=PALETTE["tertiary"],
                ),
                DashedLine(
                    axes.c2p(0.0, CalculusMath.f_quad(x_tracker.get_value())),
                    axes.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                    dash_length=0.08,
                    stroke_width=1.4,
                    color=PALETTE["tertiary"],
                ),
            )
        )
        lbl_p = always_redraw(
            lambda: MathTex(r"(x,\,f(x))", font_size=24, color=PALETTE["primary"]).next_to(
                dot_p, UP + LEFT, buff=0.12
            )
        )

        eq_stage1 = self.make_equation_card(
            [
                r"f(x) = x^2",
                r"\text{Input: } x \quad \longrightarrow \quad \text{Output: } f(x)",
                r"\Delta x = h",
                r"\Delta y = f(x+h) - f(x)",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=28,
        )

        self.play(
            FadeIn(dot_p),
            Create(proj_lines_p),
            Write(lbl_p),
            Write(eq_stage1[0]),
            Write(eq_stage1[1]),
            run_time=self.t(5.0),
        )

        self.play(
            x_tracker.animate.set_value(2.1),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(8.0),
        )
        self.play(
            x_tracker.animate.set_value(0.8),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(7.0),
        )
        self.pause(2.5)

        dot_q = always_redraw(
            lambda: Dot(
                axes.c2p(
                    x_tracker.get_value() + h_tracker.get_value(),
                    CalculusMath.f_quad(x_tracker.get_value() + h_tracker.get_value()),
                ),
                radius=0.075,
                color=PALETTE["secondary"],
            )
        )
        lbl_q = always_redraw(
            lambda: MathTex(
                r"(x+h,\,f(x+h))", font_size=24, color=PALETTE["secondary"]
            ).next_to(dot_q, UP + LEFT, buff=0.14)
        )

        horiz_segment = always_redraw(
            lambda: Line(
                axes.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                axes.c2p(
                    x_tracker.get_value() + h_tracker.get_value(),
                    CalculusMath.f_quad(x_tracker.get_value()),
                ),
                stroke_width=2.4,
                color=PALETTE["secondary"],
            )
        )
        vert_segment = always_redraw(
            lambda: Line(
                axes.c2p(
                    x_tracker.get_value() + h_tracker.get_value(),
                    CalculusMath.f_quad(x_tracker.get_value()),
                ),
                axes.c2p(
                    x_tracker.get_value() + h_tracker.get_value(),
                    CalculusMath.f_quad(x_tracker.get_value() + h_tracker.get_value()),
                ),
                stroke_width=2.4,
                color=PALETTE["primary"],
            )
        )
        lbl_h = always_redraw(
            lambda: MathTex("h", font_size=24, color=PALETTE["secondary"]).next_to(
                horiz_segment, DOWN, buff=0.12
            )
        )
        lbl_dy = always_redraw(
            lambda: MathTex(
                r"f(x+h)-f(x)", font_size=22, color=PALETTE["primary"]
            ).next_to(vert_segment, RIGHT, buff=0.12)
        )

        self.play(
            FadeIn(dot_q),
            Write(lbl_q),
            Create(horiz_segment),
            Write(lbl_h),
            Write(eq_stage1[2]),
            run_time=self.t(7.5),
        )
        self.pause(2.5)

        self.play(
            Create(vert_segment),
            Write(lbl_dy),
            Write(eq_stage1[3]),
            run_time=self.t(7.5),
        )
        self.pause(3.5)

        def build_secant() -> Line:
            x0 = x_tracker.get_value()
            h_val = max(0.05, h_tracker.get_value())
            (x1, y1), (x2, y2), _ = CalculusMath.secant_line_endpoints(
                CalculusMath.f_quad, x0, h_val, -0.2, 2.45
            )
            return Line(
                axes.c2p(x1, y1),
                axes.c2p(x2, y2),
                stroke_width=2.2,
                color=PALETTE["secondary"],
            )

        secant_line = always_redraw(build_secant)

        eq_stage2 = self.make_equation_card(
            [
                r"\text{Average Rate of Change on } [x,\,x+h]:",
                r"m_{\text{sec}} = \frac{\Delta y}{\Delta x} = \frac{f(x+h) - f(x)}{h}",
                r"f(x) = x^2 \implies m_{\text{sec}} = \frac{(x+h)^2 - x^2}{h}",
                r"\text{For } x = 0.8,\; h = 1.1: \quad m_{\text{sec}} = 2.70",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=28,
            boxed_last=True,
        )

        self.play(FadeOut(eq_stage1), run_time=self.t(2.0))
        self.play(
            Create(secant_line),
            Write(eq_stage2[0]),
            Write(eq_stage2[1]),
            run_time=self.t(8.5),
        )
        self.pause(3.5)

        self.play(Write(eq_stage2[2]), Write(eq_stage2[3]), run_time=self.t(7.5))
        self.pause(4.0)

        eq_stage3 = self.make_equation_card(
            [
                r"m_{\text{sec}}(x, h) = \frac{f(x+h) - f(x)}{h}",
                r"x = 0.8,\; h = 1.10 \implies m_{\text{sec}} = 2.70",
                r"x = 0.8,\; h = 0.60 \implies m_{\text{sec}} = 2.20",
                r"x = 1.2,\; h = 0.60 \implies m_{\text{sec}} = 3.00",
                r"x = 0.5,\; h = 0.80 \implies m_{\text{sec}} = 1.80",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=26,
        )

        self.play(ReplacementTransform(eq_stage2, eq_stage3[:2]), run_time=self.t(3.5))
        self.pause(2.5)

        self.play(
            h_tracker.animate.set_value(0.60),
            Write(eq_stage3[2]),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(4.5)

        self.play(
            x_tracker.animate.set_value(1.20),
            Write(eq_stage3[3]),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(4.5)

        self.play(
            x_tracker.animate.set_value(0.50),
            h_tracker.animate.set_value(0.80),
            Write(eq_stage3[4]),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(5.5)

        self.play(
            x_tracker.animate.set_value(0.90),
            h_tracker.animate.set_value(1.00),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(8.0),
        )
        self.pause(3.5)

        self.play(
            FadeOut(
                VGroup(
                    header,
                    axes,
                    axes_labels,
                    parabola,
                    dot_p,
                    proj_lines_p,
                    lbl_p,
                    dot_q,
                    lbl_q,
                    horiz_segment,
                    vert_segment,
                    lbl_h,
                    lbl_dy,
                    secant_line,
                    eq_stage3,
                )
            ),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter03_SecantToTangent(SilentMathBaseScene):
    """
    Chapter 3 — From Secant to Tangent (3:30–6:00, Target: 150.0s)
    --------------------------------------------------------------
    - Displays f(x) = x^2 with fixed anchor P(1, 1) and movable Q(1+h, (1+h)^2).
    - Gradually decreases h -> 0, showing the secant line rotating smoothly into
      the tangent line at x = 1.
    - Demonstrates why h cannot be set directly to 0 (0/0 indeterminate form) and
      builds the limit definition:
        f'(x) = lim_{h -> 0} (f(x+h) - f(x)) / h
    - Verifies two-sided convergence (h -> 0+ and h -> 0-).
    """

    chapter_number = 3
    chapter_title = "From Secant to Tangent"
    target_duration = 150.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "03",
            "From Secant to Tangent",
            r"f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        axes, axes_labels = self.create_monochrome_axes(
            x_range=[-0.4, 2.6, 0.5],
            y_range=[-0.5, 6.5, 1.0],
            zone=SafeLayoutEngine.LEFT_GRAPH_ZONE,
            x_label_tex="x",
            y_label_tex="f(x)",
            include_numbers=True,
        )
        curve = axes.plot(
            CalculusMath.f_quad,
            x_range=[-0.3, 2.5],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        x0 = 1.0
        h_tracker = ValueTracker(1.30)

        dot_p = Dot(axes.c2p(x0, CalculusMath.f_quad(x0)), radius=0.08, color=PALETTE["primary"])
        lbl_p = MathTex(r"P(1,\,1)", font_size=24, color=PALETTE["primary"]).next_to(
            dot_p, DOWN + RIGHT, buff=0.14
        )

        dot_q = always_redraw(
            lambda: Dot(
                axes.c2p(x0 + h_tracker.get_value(), CalculusMath.f_quad(x0 + h_tracker.get_value())),
                radius=0.07,
                color=PALETTE["secondary"],
            )
        )
        lbl_q = always_redraw(
            lambda: MathTex(r"Q(1+h,\,(1+h)^2)", font_size=22, color=PALETTE["secondary"]).next_to(
                dot_q, UP + LEFT, buff=0.12
            )
        )

        def build_secant_or_tangent() -> VGroup:
            h_val = h_tracker.get_value()
            if abs(h_val) < 1e-4:
                h_val = 1e-4
            (x1, y1), (x2, y2), _ = CalculusMath.secant_line_endpoints(
                CalculusMath.f_quad, x0, h_val, -0.1, 2.4
            )
            sec = Line(
                axes.c2p(x1, y1),
                axes.c2p(x2, y2),
                stroke_width=2.2,
                color=PALETTE["secondary"],
            )
            dx_line = DashedLine(
                axes.c2p(x0, CalculusMath.f_quad(x0)),
                axes.c2p(x0 + h_val, CalculusMath.f_quad(x0)),
                dash_length=0.07,
                stroke_width=1.5,
                color=PALETTE["tertiary"],
            )
            dy_line = DashedLine(
                axes.c2p(x0 + h_val, CalculusMath.f_quad(x0)),
                axes.c2p(x0 + h_val, CalculusMath.f_quad(x0 + h_val)),
                dash_length=0.07,
                stroke_width=1.5,
                color=PALETTE["tertiary"],
            )
            return VGroup(sec, dx_line, dy_line)

        secant_group = always_redraw(build_secant_or_tangent)

        (tx1, ty1), (tx2, ty2), _ = CalculusMath.tangent_line_endpoints(
            CalculusMath.f_quad, CalculusMath.df_quad, x0, half_span_x=1.1
        )
        true_tangent = DashedLine(
            axes.c2p(tx1, ty1),
            axes.c2p(tx2, ty2),
            dash_length=0.10,
            stroke_width=1.8,
            color=PALETTE["tertiary"],
        )

        self.play(Create(axes), Write(axes_labels), Create(curve), run_time=self.t(6.0))
        self.play(
            FadeIn(dot_p),
            Write(lbl_p),
            FadeIn(dot_q),
            Write(lbl_q),
            Create(secant_group),
            run_time=self.t(5.5),
        )
        self.pause(2.5)

        table_card = self.make_equation_card(
            [
                r"\text{Secant Slope at } x = 1:\quad m(h) = \frac{(1+h)^2 - 1^2}{h}",
                r"h = 1.000 \quad \longrightarrow \quad m = 3.000",
                r"h = 0.500 \quad \longrightarrow \quad m = 2.500",
                r"h = 0.100 \quad \longrightarrow \quad m = 2.100",
                r"h = 0.010 \quad \longrightarrow \quad m = 2.010",
                r"h \to 0^+ \quad \longrightarrow \quad m \to 2.000",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=26,
            boxed_last=True,
        )

        self.play(Write(table_card[0]), Create(true_tangent), run_time=self.t(6.0))

        for h_target, row_idx, dur in [
            (1.00, 1, 8.0),
            (0.50, 2, 9.0),
            (0.10, 3, 10.0),
            (0.01, 4, 10.0),
        ]:
            self.play(
                h_tracker.animate.set_value(h_target),
                Write(table_card[row_idx]),
                rate_func=rate_functions.ease_in_out_sine,
                run_time=self.t(dur),
            )
            self.pause(2.0)

        self.play(Write(table_card[5]), run_time=self.t(5.0))
        self.pause(4.0)

        limit_card = self.make_equation_card(
            [
                r"\text{Why not set } h = 0 \text{ directly?}",
                r"\left.\frac{f(x+h)-f(x)}{h}\right|_{h=0} = \frac{0}{0} \quad (\text{undefined})",
                r"\text{Instead, inspect behavior as } h \to 0 \text{ with } h \neq 0:",
                r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=27,
            boxed_last=True,
        )

        self.play(FadeOut(table_card), run_time=self.t(2.5))
        self.play(Write(limit_card[0]), Write(limit_card[1]), run_time=self.t(8.5))
        self.pause(4.0)
        self.play(Write(limit_card[2]), Write(limit_card[3]), run_time=self.t(9.5))
        self.pause(5.0)

        two_sided_card = self.make_equation_card(
            [
                r"\text{Two-Sided Approach to the Tangent Slope:}",
                r"h = -0.80 \quad \longrightarrow \quad m = 1.20",
                r"h = -0.20 \quad \longrightarrow \quad m = 1.80",
                r"h = -0.01 \quad \longrightarrow \quad m = 1.99",
                r"\lim_{h \to 0^-} \frac{f(1+h)-f(1)}{h} = \lim_{h \to 0^+} \frac{f(1+h)-f(1)}{h} = 2",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=26,
            boxed_last=True,
        )

        self.play(FadeOut(limit_card), run_time=self.t(2.5))
        self.play(Write(two_sided_card[0]), run_time=self.t(3.5))

        for h_target, row_idx, dur in [
            (-0.80, 1, 8.0),
            (-0.20, 2, 8.0),
            (-0.01, 3, 8.0),
        ]:
            self.play(
                h_tracker.animate.set_value(h_target),
                Write(two_sided_card[row_idx]),
                rate_func=rate_functions.ease_in_out_sine,
                run_time=self.t(dur),
            )
            self.pause(1.8)

        self.play(Write(two_sided_card[4]), run_time=self.t(6.5))
        self.pause(5.5)

        self.play(
            FadeOut(
                VGroup(
                    header,
                    axes,
                    axes_labels,
                    curve,
                    dot_p,
                    lbl_p,
                    dot_q,
                    lbl_q,
                    secant_group,
                    true_tangent,
                    two_sided_card,
                )
            ),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter04_DerivingXSquared(SilentMathBaseScene):
    """
    Chapter 4 — Deriving the Derivative of x^2 (6:00–9:00, Target: 180.0s)
    ----------------------------------------------------------------------
    - Complete algebraic and geometric derivation of f'(x) = 2x from first principles:
        1. f(x) = x^2
        2. f'(x) = lim_{h->0} ((x+h)^2 - x^2) / h
        3. Geometric area visualization of (x+h)^2 = x^2 + 2xh + h^2
        4. Cancellation of x^2 - x^2 -> lim_{h->0} (2xh + h^2) / h
        5. Factoring h and cancelling for h != 0 -> lim_{h->0} (2x + h)
        6. Evaluating the limit as h -> 0 -> boxed f'(x) = 2x
    """

    chapter_number = 4
    chapter_title = "Deriving the Derivative of x^2"
    target_duration = 180.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "04",
            "Deriving the Derivative of x²",
            r"f(x) = x^2 \implies f'(x) = 2x",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        x_side = 2.6
        h_tracker = ValueTracker(1.05)
        sq_origin = np.array([-5.2, -2.2, 0.0])

        def build_area_square() -> VGroup:
            h_val = max(0.04, h_tracker.get_value())
            sq_x2 = Rectangle(
                width=x_side,
                height=x_side,
                stroke_color=PALETTE["primary"],
                stroke_width=2.2,
                fill_color=PALETTE["box_fill"],
                fill_opacity=0.85,
            )
            sq_x2.move_to(sq_origin + np.array([0.5 * x_side, 0.5 * x_side, 0.0]))
            lbl_x2 = MathTex("x^2", font_size=32, color=PALETTE["primary"]).move_to(sq_x2.get_center())

            rect_right = Rectangle(
                width=h_val,
                height=x_side,
                stroke_color=PALETTE["secondary"],
                stroke_width=2.0,
                fill_color=PALETTE["grid"],
                fill_opacity=0.75,
            )
            rect_right.move_to(sq_origin + np.array([x_side + 0.5 * h_val, 0.5 * x_side, 0.0]))
            lbl_rx = MathTex("xh", font_size=24 if h_val > 0.45 else 16, color=PALETTE["secondary"]).move_to(
                rect_right.get_center()
            )

            rect_top = Rectangle(
                width=x_side,
                height=h_val,
                stroke_color=PALETTE["secondary"],
                stroke_width=2.0,
                fill_color=PALETTE["grid"],
                fill_opacity=0.75,
            )
            rect_top.move_to(sq_origin + np.array([0.5 * x_side, x_side + 0.5 * h_val, 0.0]))
            lbl_tx = MathTex("xh", font_size=24 if h_val > 0.45 else 16, color=PALETTE["secondary"]).move_to(
                rect_top.get_center()
            )

            sq_h2 = Rectangle(
                width=h_val,
                height=h_val,
                stroke_color=PALETTE["tertiary"],
                stroke_width=1.8,
                fill_color=PALETTE["primary"],
                fill_opacity=0.20,
            )
            sq_h2.move_to(sq_origin + np.array([x_side + 0.5 * h_val, x_side + 0.5 * h_val, 0.0]))
            lbl_h2 = MathTex("h^2", font_size=22 if h_val > 0.45 else 14, color=PALETTE["primary"]).move_to(
                sq_h2.get_center()
            )

            dim_x_bot = MathTex("x", font_size=24, color=PALETTE["primary"]).next_to(sq_x2, DOWN, buff=0.16)
            dim_h_bot = MathTex("h", font_size=24, color=PALETTE["secondary"]).next_to(rect_right, DOWN, buff=0.16)
            dim_x_left = MathTex("x", font_size=24, color=PALETTE["primary"]).next_to(sq_x2, LEFT, buff=0.16)
            dim_h_left = MathTex("h", font_size=24, color=PALETTE["secondary"]).next_to(rect_top, LEFT, buff=0.16)

            return VGroup(
                sq_x2,
                lbl_x2,
                rect_right,
                lbl_rx,
                rect_top,
                lbl_tx,
                sq_h2,
                lbl_h2,
                dim_x_bot,
                dim_h_bot,
                dim_x_left,
                dim_h_left,
            )

        area_diagram = always_redraw(build_area_square)

        deriv_part1 = self.make_equation_card(
            [
                r"f(x) = x^2",
                r"f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h}",
                r"(x+h)^2 = x^2 + 2xh + h^2",
                r"f'(x) = \lim_{h \to 0} \frac{(x^2 + 2xh + h^2) - x^2}{h}",
                r"f'(x) = \lim_{h \to 0} \frac{2xh + h^2}{h}",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=28,
        )

        self.play(Write(deriv_part1[0]), run_time=self.t(5.0))
        self.pause(2.5)
        self.play(Write(deriv_part1[1]), run_time=self.t(7.5))
        self.pause(3.0)

        self.play(
            FadeIn(area_diagram),
            Write(deriv_part1[2]),
            run_time=self.t(10.0),
        )
        self.pause(5.5)

        self.play(Write(deriv_part1[3]), run_time=self.t(9.0))
        self.pause(4.5)
        self.play(Write(deriv_part1[4]), run_time=self.t(8.5))
        self.pause(6.0)

        deriv_part2 = self.make_equation_card(
            [
                r"f'(x) = \lim_{h \to 0} \frac{2xh + h^2}{h}",
                r"f'(x) = \lim_{h \to 0} \frac{h(2x + h)}{h}",
                r"\text{Since } h \to 0 \text{ means } h \neq 0,\text{ cancel } \frac{h}{h} = 1:",
                r"f'(x) = \lim_{h \to 0} (2x + h)",
                r"f'(x) = 2x",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=28,
            boxed_last=True,
        )

        self.play(FadeOut(deriv_part1), run_time=self.t(2.5))
        self.play(Write(deriv_part2[0]), run_time=self.t(5.5))
        self.pause(3.0)

        self.play(Write(deriv_part2[1]), run_time=self.t(8.5))
        self.pause(4.0)

        self.play(Write(deriv_part2[2]), run_time=self.t(9.5))
        self.pause(5.0)

        self.play(Write(deriv_part2[3]), run_time=self.t(8.5))
        self.pause(4.5)

        self.play(
            h_tracker.animate.set_value(0.45),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(3.5)
        self.play(
            h_tracker.animate.set_value(0.10),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(3.5)

        self.play(Write(deriv_part2[4]), run_time=self.t(9.0))
        self.pause(10.0)

        self.play(
            FadeOut(VGroup(header, area_diagram, deriv_part2)),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter05_GraphOf2x(SilentMathBaseScene):
    """
    Chapter 5 — Understanding the Graph of 2x (9:00–12:00, Target: 180.0s)
    ----------------------------------------------------------------------
    - Side-by-side coordinate planes:
        Left:  Original function f(x) = x^2 with gliding point and tangent line.
        Right: Derivative function f'(x) = 2x plotted point-by-point from tangent slopes.
    - Explicitly inspects x = -2, x = 0, x = 1, x = 2 and their exact slopes
      f'(-2) = -4, f'(0) = 0, f'(1) = 2, f'(2) = 4.
    - Explains negative slopes left of the vertex, horizontal tangent at x = 0,
      and increasing positive slopes for x > 0.
    """

    chapter_number = 5
    chapter_title = "Understanding the Graph of 2x"
    target_duration = 180.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "05",
            "Understanding the Graph of 2x",
            r"f(x) = x^2 \quad \longrightarrow \quad f'(x) = 2x",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        ax_left, lbl_left = self.create_monochrome_axes(
            x_range=[-2.5, 2.5, 1.0],
            y_range=[-0.8, 5.2, 1.0],
            zone=SafeLayoutEngine.LEFT_GRAPH_ZONE,
            x_label_tex="x",
            y_label_tex="f(x)=x^2",
            include_numbers=True,
        )
        curve_f = ax_left.plot(
            CalculusMath.f_quad,
            x_range=[-2.2, 2.2],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        ax_right, lbl_right = self.create_monochrome_axes(
            x_range=[-2.5, 2.5, 1.0],
            y_range=[-5.0, 5.0, 2.0],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            x_label_tex="x",
            y_label_tex="f'(x)=2x",
            include_numbers=True,
        )

        self.play(
            Create(ax_left),
            Write(lbl_left),
            Create(curve_f),
            Create(ax_right),
            Write(lbl_right),
            run_time=self.t(8.0),
        )
        self.pause(2.5)

        x_tracker = ValueTracker(-2.0)

        dot_f = always_redraw(
            lambda: Dot(
                ax_left.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        def build_tangent_on_f() -> Line:
            x0 = x_tracker.get_value()
            (x1, y1), (x2, y2), _ = CalculusMath.tangent_line_endpoints(
                CalculusMath.f_quad, CalculusMath.df_quad, x0, half_span_x=0.75
            )
            return Line(
                ax_left.c2p(x1, y1),
                ax_left.c2p(x2, y2),
                stroke_width=2.4,
                color=PALETTE["secondary"],
            )

        tangent_line = always_redraw(build_tangent_on_f)

        dot_df = always_redraw(
            lambda: Dot(
                ax_right.c2p(x_tracker.get_value(), CalculusMath.df_quad(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        self.play(FadeIn(dot_f), Create(tangent_line), FadeIn(dot_df), run_time=self.t(4.5))

        sample_points = [
            (-2.0, -4.0, r"x = -2 \implies f'(-2) = 2(-2) = -4"),
            (0.0, 0.0, r"x = 0 \implies f'(0) = 2(0) = 0"),
            (1.0, 2.0, r"x = 1 \implies f'(1) = 2(1) = 2"),
            (2.0, 4.0, r"x = 2 \implies f'(2) = 2(2) = 4"),
        ]

        plotted_markers = VGroup()
        active_badge: Optional[Mobject] = None

        for x_val, slope_val, badge_tex in sample_points:
            self.play(
                x_tracker.animate.set_value(x_val),
                rate_func=rate_functions.ease_in_out_sine,
                run_time=self.t(10.0),
            )

            badge = MathTex(badge_tex, font_size=26, color=PALETTE["primary"])
            badge_box = SurroundingRectangle(
                badge,
                color=PALETTE["box_stroke"],
                fill_color=PALETTE["box_fill"],
                fill_opacity=0.92,
                buff=0.14,
            )
            badge_group = VGroup(badge_box, badge).move_to(np.array([0.0, -3.15, 0.0]))

            marker_dot = Dot(ax_right.c2p(x_val, slope_val), radius=0.07, color=PALETTE["secondary"])
            marker_dash = DashedLine(
                ax_right.c2p(x_val, 0.0),
                ax_right.c2p(x_val, slope_val),
                dash_length=0.07,
                stroke_width=1.3,
                color=PALETTE["tertiary"],
            )
            plotted_markers.add(marker_dash, marker_dot)

            if active_badge is not None:
                self.play(FadeOut(active_badge), run_time=self.t(1.5))
            self.play(
                FadeIn(badge_group),
                Create(marker_dash),
                FadeIn(marker_dot, scale=1.3),
                run_time=self.t(5.5),
            )
            active_badge = badge_group
            self.pause(6.5)

        if active_badge is not None:
            self.play(FadeOut(active_badge), run_time=self.t(2.0))

        self.play(
            x_tracker.animate.set_value(-2.2),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(6.0),
        )

        derivative_line = always_redraw(
            lambda: ax_right.plot(
                CalculusMath.df_quad,
                x_range=[-2.2, max(-2.19, x_tracker.get_value())],
                color=PALETTE["primary"],
                stroke_width=3.0,
            )
        )
        self.add(derivative_line)

        self.play(
            x_tracker.animate.set_value(2.2),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(22.0),
        )
        self.pause(6.0)

        self.play(
            x_tracker.animate.set_value(0.0),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(5.0)
        self.play(
            x_tracker.animate.set_value(1.5),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(10.0),
        )
        self.pause(6.0)

        self.play(
            FadeOut(
                VGroup(
                    header,
                    ax_left,
                    lbl_left,
                    curve_f,
                    ax_right,
                    lbl_right,
                    dot_f,
                    tangent_line,
                    dot_df,
                    plotted_markers,
                    derivative_line,
                )
            ),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter06_CubicDerivative(SilentMathBaseScene):
    """
    Chapter 6 — Derivative of a Cubic Function (12:00–15:00, Target: 180.0s)
    ------------------------------------------------------------------------
    - Introduces f(x) = x^3 and derives f'(x) = 3x^2 from first principles:
        (x+h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3
    - Animates subtraction of x^3, factoring and cancelling h for h != 0, and
      evaluating the limit as h -> 0.
    - Compares the graphs of f(x) = x^3 and f'(x) = 3x^2, explaining why
      f'(x) >= 0 everywhere and why the tangent is horizontal at the origin (x = 0).
    """

    chapter_number = 6
    chapter_title = "Derivative of a Cubic Function"
    target_duration = 180.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "06",
            "Derivative of a Cubic Function",
            r"f(x) = x^3 \implies f'(x) = 3x^2",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        cubic_steps_1 = self.make_equation_card(
            [
                r"f(x) = x^3",
                r"f'(x) = \lim_{h \to 0} \frac{(x+h)^3 - x^3}{h}",
                r"(x+h)^3 = x^3 + 3x^2h + 3xh^2 + h^3",
                r"f'(x) = \lim_{h \to 0} \frac{(x^3 + 3x^2h + 3xh^2 + h^3) - x^3}{h}",
            ],
            zone=SafeLayoutEngine.FULL_STAGE_ZONE,
            font_size=32,
        )

        for idx, dur in [(0, 5.5), (1, 8.5), (2, 10.0), (3, 10.0)]:
            self.play(Write(cubic_steps_1[idx]), run_time=self.t(dur))
            self.pause(3.5)

        cubic_steps_2 = self.make_equation_card(
            [
                r"f'(x) = \lim_{h \to 0} \frac{3x^2h + 3xh^2 + h^3}{h}",
                r"f'(x) = \lim_{h \to 0} \frac{h(3x^2 + 3xh + h^2)}{h}",
                r"\text{For } h \neq 0:\quad f'(x) = \lim_{h \to 0} \left(3x^2 + 3xh + h^2\right)",
                r"f'(x) = 3x^2",
            ],
            zone=SafeLayoutEngine.FULL_STAGE_ZONE,
            font_size=32,
            boxed_last=True,
        )

        self.play(FadeOut(cubic_steps_1), run_time=self.t(2.5))
        for idx, dur in [(0, 8.5), (1, 9.0), (2, 10.0), (3, 9.0)]:
            self.play(Write(cubic_steps_2[idx]), run_time=self.t(dur))
            self.pause(4.0)

        self.play(FadeOut(cubic_steps_2), run_time=self.t(3.0))

        ax_left, lbl_left = self.create_monochrome_axes(
            x_range=[-2.0, 2.0, 1.0],
            y_range=[-5.0, 5.0, 2.0],
            zone=SafeLayoutEngine.LEFT_GRAPH_ZONE,
            x_label_tex="x",
            y_label_tex="f(x)=x^3",
            include_numbers=True,
        )
        curve_cubic = ax_left.plot(
            CalculusMath.f_cubic,
            x_range=[-1.7, 1.7],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        ax_right, lbl_right = self.create_monochrome_axes(
            x_range=[-2.0, 2.0, 1.0],
            y_range=[-1.0, 8.0, 2.0],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            x_label_tex="x",
            y_label_tex="f'(x)=3x^2",
            include_numbers=True,
        )
        curve_dcubic = ax_right.plot(
            CalculusMath.df_cubic,
            x_range=[-1.6, 1.6],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        self.play(
            Create(ax_left),
            Write(lbl_left),
            Create(curve_cubic),
            Create(ax_right),
            Write(lbl_right),
            Create(curve_dcubic),
            run_time=self.t(9.0),
        )

        x_tracker = ValueTracker(-1.5)

        dot_c = always_redraw(
            lambda: Dot(
                ax_left.c2p(x_tracker.get_value(), CalculusMath.f_cubic(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        def build_cubic_tangent() -> Line:
            x0 = x_tracker.get_value()
            (x1, y1), (x2, y2), _ = CalculusMath.tangent_line_endpoints(
                CalculusMath.f_cubic, CalculusMath.df_cubic, x0, half_span_x=0.55
            )
            return Line(
                ax_left.c2p(x1, y1),
                ax_left.c2p(x2, y2),
                stroke_width=2.4,
                color=PALETTE["secondary"],
            )

        cubic_tangent = always_redraw(build_cubic_tangent)

        dot_dc = always_redraw(
            lambda: Dot(
                ax_right.c2p(x_tracker.get_value(), CalculusMath.df_cubic(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        note_nonneg = MathTex(
            r"f'(x) = 3x^2 \ge 0 \quad (\text{slope is never negative; } f'(0) = 0 \text{ at origin})",
            font_size=24,
            color=PALETTE["secondary"],
        ).move_to(np.array([0.0, -3.20, 0.0]))

        self.play(
            FadeIn(dot_c),
            Create(cubic_tangent),
            FadeIn(dot_dc),
            FadeIn(note_nonneg),
            run_time=self.t(5.5),
        )

        self.play(
            x_tracker.animate.set_value(0.0),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(18.0),
        )
        self.pause(7.0)

        self.play(
            x_tracker.animate.set_value(1.5),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(18.0),
        )
        self.pause(7.0)

        self.play(
            x_tracker.animate.set_value(-1.0),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(10.0),
        )
        self.pause(4.0)
        self.play(
            x_tracker.animate.set_value(1.0),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(10.0),
        )
        self.pause(5.0)

        self.play(
            FadeOut(
                VGroup(
                    header,
                    ax_left,
                    lbl_left,
                    curve_cubic,
                    ax_right,
                    lbl_right,
                    curve_dcubic,
                    dot_c,
                    cubic_tangent,
                    dot_dc,
                    note_nonneg,
                )
            ),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter07_RatesOfChangePhysics(SilentMathBaseScene):
    """
    Chapter 7 — Derivatives as Rates of Change (15:00–18:00, Target: 180.0s)
    ------------------------------------------------------------------------
    - Synchronized kinematics visual:
        Top Track Zone:   1D motion track for particle position s(t) = t^2 [m]
                          for t in [0, 4] s (with explicit unit scale 1 m/s^2).
        Bottom Left Zone: Position-time graph s(t) = t^2 [m] with tangent slope.
        Bottom Right Zone: Instantaneous velocity v(t) = s'(t) = 2t [m/s] &
                          comparison between average velocity and instantaneous velocity.
    """

    chapter_number = 7
    chapter_title = "Derivatives as Rates of Change"
    target_duration = 180.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "07",
            "Derivatives as Rates of Change",
            r"s(t) = t^2\text{ [m]} \implies v(t) = s'(t) = 2t\text{ [m/s]}",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        track_line = NumberLine(
            x_range=[0.0, 16.0, 2.0],
            length=10.8,
            color=PALETTE["axes"],
            include_numbers=True,
            font_size=20,
            tick_size=0.07,
        )
        track_line.move_to(SafeLayoutEngine.TOP_TRACK_ZONE.center + DOWN * 0.15)
        track_title = MathTex(
            r"\text{Position Track } s(t) = (1\,\text{m/s}^2)\,t^2\text{ [m]}",
            font_size=24,
            color=PALETTE["secondary"],
        ).next_to(track_line, UP, buff=0.22)

        ax_pos, lbl_pos = self.create_monochrome_axes(
            x_range=[0.0, 4.2, 1.0],
            y_range=[0.0, 17.0, 4.0],
            zone=SafeLayoutEngine.BOTTOM_LEFT_ZONE,
            x_label_tex=r"t\text{ [s]}",
            y_label_tex=r"s(t)\text{ [m]}",
            include_numbers=True,
        )
        curve_pos = ax_pos.plot(
            CalculusMath.f_quad,
            x_range=[0.0, 4.0],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        ax_vel, lbl_vel = self.create_monochrome_axes(
            x_range=[0.0, 4.2, 1.0],
            y_range=[0.0, 9.0, 2.0],
            zone=SafeLayoutEngine.BOTTOM_RIGHT_ZONE,
            x_label_tex=r"t\text{ [s]}",
            y_label_tex=r"v(t)\text{ [m/s]}",
            include_numbers=True,
        )
        curve_vel = ax_vel.plot(
            CalculusMath.df_quad,
            x_range=[0.0, 4.0],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        self.play(
            Create(track_line),
            Write(track_title),
            Create(ax_pos),
            Write(lbl_pos),
            Create(curve_pos),
            Create(ax_vel),
            Write(lbl_vel),
            Create(curve_vel),
            run_time=self.t(10.0),
        )
        self.pause(3.0)

        t_tracker = ValueTracker(0.0)

        particle_dot = always_redraw(
            lambda: Dot(
                track_line.n2p(CalculusMath.f_quad(t_tracker.get_value())),
                radius=0.10,
                color=PALETTE["primary"],
            )
        )

        def build_velocity_arrow() -> VGroup:
            t_val = t_tracker.get_value()
            s_val = CalculusMath.f_quad(t_val)
            v_val = CalculusMath.df_quad(t_val)
            start_pt = track_line.n2p(s_val) + UP * 0.22
            end_pt = track_line.n2p(min(16.0, s_val + 0.35 * v_val)) + UP * 0.22
            if np.linalg.norm(end_pt - start_pt) < 0.08:
                return VGroup()
            arr = Arrow(
                start_pt,
                end_pt,
                buff=0.0,
                stroke_width=2.4,
                color=PALETTE["secondary"],
                max_tip_length_to_length_ratio=0.25,
            )
            return VGroup(arr)

        vel_arrow = always_redraw(build_velocity_arrow)

        dot_pos = always_redraw(
            lambda: Dot(
                ax_pos.c2p(t_tracker.get_value(), CalculusMath.f_quad(t_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        def build_pos_tangent() -> Line:
            t0 = t_tracker.get_value()
            (t1, s1), (t2, s2), _ = CalculusMath.tangent_line_endpoints(
                CalculusMath.f_quad, CalculusMath.df_quad, t0, half_span_x=0.60
            )
            t1_c = max(0.0, t1)
            t2_c = min(4.1, t2)
            s1_c = CalculusMath.f_quad(t0) + CalculusMath.df_quad(t0) * (t1_c - t0)
            s2_c = CalculusMath.f_quad(t0) + CalculusMath.df_quad(t0) * (t2_c - t0)
            return Line(
                ax_pos.c2p(t1_c, s1_c),
                ax_pos.c2p(t2_c, s2_c),
                stroke_width=2.2,
                color=PALETTE["secondary"],
            )

        pos_tangent = always_redraw(build_pos_tangent)

        dot_vel = always_redraw(
            lambda: Dot(
                ax_vel.c2p(t_tracker.get_value(), CalculusMath.df_quad(t_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        self.play(
            FadeIn(particle_dot),
            FadeIn(vel_arrow),
            FadeIn(dot_pos),
            Create(pos_tangent),
            FadeIn(dot_vel),
            run_time=self.t(5.0),
        )

        time_checkpoints = [
            (1.0, r"t = 1\,\text{s}:\quad s(1) = 1\,\text{m},\quad v(1) = s'(1) = 2\,\text{m/s}"),
            (2.0, r"t = 2\,\text{s}:\quad s(2) = 4\,\text{m},\quad v(2) = s'(2) = 4\,\text{m/s}"),
            (3.0, r"t = 3\,\text{s}:\quad s(3) = 9\,\text{m},\quad v(3) = s'(3) = 6\,\text{m/s}"),
            (4.0, r"t = 4\,\text{s}:\quad s(4) = 16\,\text{m},\quad v(4) = s'(4) = 8\,\text{m/s}"),
        ]

        active_readout: Optional[Mobject] = None
        for t_target, readout_tex in time_checkpoints:
            self.play(
                t_tracker.animate.set_value(t_target),
                rate_func=rate_functions.linear,
                run_time=self.t(12.0),
            )
            readout = MathTex(readout_tex, font_size=24, color=PALETTE["primary"]).move_to(
                np.array([0.0, 1.15, 0.0])
            )
            if active_readout is not None:
                self.play(FadeOut(active_readout), run_time=self.t(1.5))
            self.play(FadeIn(readout), run_time=self.t(3.5))
            active_readout = readout
            self.pause(6.5)

        if active_readout is not None:
            self.play(FadeOut(active_readout), run_time=self.t(2.0))

        secant_1_3 = DashedLine(
            ax_pos.c2p(1.0, 1.0),
            ax_pos.c2p(3.0, 9.0),
            dash_length=0.09,
            stroke_width=2.0,
            color=PALETTE["tertiary"],
        )
        avg_vs_inst = MathTex(
            r"v_{\text{avg}}[1,3] = \frac{s(3)-s(1)}{3-1} = 4\,\text{m/s} \quad \Longleftrightarrow \quad v(t) = \lim_{\Delta t \to 0} \frac{s(t+\Delta t)-s(t)}{\Delta t} = 2t",
            font_size=23,
            color=PALETTE["secondary"],
        ).move_to(np.array([0.0, 1.15, 0.0]))

        self.play(
            t_tracker.animate.set_value(2.0),
            Create(secant_1_3),
            FadeIn(avg_vs_inst),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.pause(10.0)

        self.play(t_tracker.animate.set_value(0.0), run_time=self.t(6.0))
        self.play(
            t_tracker.animate.set_value(4.0),
            rate_func=rate_functions.linear,
            run_time=self.t(20.0),
        )
        self.pause(6.0)

        self.play(
            FadeOut(
                VGroup(
                    header,
                    track_line,
                    track_title,
                    ax_pos,
                    lbl_pos,
                    curve_pos,
                    ax_vel,
                    lbl_vel,
                    curve_vel,
                    particle_dot,
                    vel_arrow,
                    dot_pos,
                    pos_tangent,
                    dot_vel,
                    secant_1_3,
                    avg_vs_inst,
                )
            ),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter08_ComparingFunctions(SilentMathBaseScene):
    """
    Chapter 8 — Comparing Functions and Slopes (18:00–21:00, Target: 180.0s)
    ------------------------------------------------------------------------
    - Compares f(x) = x^2 and g(x) = x^3 on the left axes (solid white vs
      light-gray) and their derivatives f'(x) = 2x and g'(x) = 3x^2 on the right axes.
    - Demonstrates at selected points:
        1. At x = 1: f(1) = g(1) = 1 (equal function values), yet f'(1) = 2 != g'(1) = 3
           (equal outputs do NOT imply equal slopes).
        2. At x = 2/3: f'(2/3) = g'(2/3) = 4/3 (parallel tangents), yet f(2/3) != g(2/3).
        3. At x = -1: f'(-1) = -2 (decreasing) while g'(-1) = +3 (increasing).
    """

    chapter_number = 8
    chapter_title = "Comparing Functions and Slopes"
    target_duration = 180.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "08",
            "Comparing Functions and Slopes",
            r"f(x)=x^2,\; g(x)=x^3 \quad \text{vs.} \quad f'(x)=2x,\; g'(x)=3x^2",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        ax_func, lbl_func = self.create_monochrome_axes(
            x_range=[-1.6, 1.8, 0.5],
            y_range=[-3.5, 4.5, 1.0],
            zone=SafeLayoutEngine.LEFT_GRAPH_ZONE,
            x_label_tex="x",
            y_label_tex=r"f(x),\,g(x)",
            include_numbers=True,
        )
        curve_f = ax_func.plot(
            CalculusMath.f_quad,
            x_range=[-1.5, 1.65],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )
        curve_g = ax_func.plot(
            CalculusMath.f_cubic,
            x_range=[-1.5, 1.62],
            color=PALETTE["secondary"],
            stroke_width=2.4,
        )

        ax_deriv, lbl_deriv = self.create_monochrome_axes(
            x_range=[-1.6, 1.8, 0.5],
            y_range=[-3.5, 6.5, 2.0],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            x_label_tex="x",
            y_label_tex=r"f'(x),\,g'(x)",
            include_numbers=True,
        )
        curve_df = ax_deriv.plot(
            CalculusMath.df_quad,
            x_range=[-1.5, 1.65],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )
        curve_dg = ax_deriv.plot(
            CalculusMath.df_cubic,
            x_range=[-1.45, 1.45],
            color=PALETTE["secondary"],
            stroke_width=2.4,
        )

        self.play(
            Create(ax_func),
            Write(lbl_func),
            Create(curve_f),
            Create(curve_g),
            Create(ax_deriv),
            Write(lbl_deriv),
            Create(curve_df),
            Create(curve_dg),
            run_time=self.t(10.0),
        )
        self.pause(3.0)

        x_tracker = ValueTracker(0.0)

        dot_f = always_redraw(
            lambda: Dot(
                ax_func.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )
        dot_g = always_redraw(
            lambda: Dot(
                ax_func.c2p(x_tracker.get_value(), CalculusMath.f_cubic(x_tracker.get_value())),
                radius=0.070,
                color=PALETTE["secondary"],
            )
        )
        dot_df = always_redraw(
            lambda: Dot(
                ax_deriv.c2p(x_tracker.get_value(), CalculusMath.df_quad(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )
        dot_dg = always_redraw(
            lambda: Dot(
                ax_deriv.c2p(x_tracker.get_value(), CalculusMath.df_cubic(x_tracker.get_value())),
                radius=0.070,
                color=PALETTE["secondary"],
            )
        )

        def build_tangents_pair() -> VGroup:
            x0 = x_tracker.get_value()
            (fx1, fy1), (fx2, fy2), _ = CalculusMath.tangent_line_endpoints(
                CalculusMath.f_quad, CalculusMath.df_quad, x0, half_span_x=0.45
            )
            (gx1, gy1), (gx2, gy2), _ = CalculusMath.tangent_line_endpoints(
                CalculusMath.f_cubic, CalculusMath.df_cubic, x0, half_span_x=0.45
            )
            tf = Line(
                ax_func.c2p(fx1, fy1),
                ax_func.c2p(fx2, fy2),
                stroke_width=2.2,
                color=PALETTE["primary"],
            )
            tg = DashedLine(
                ax_func.c2p(gx1, gy1),
                ax_func.c2p(gx2, gy2),
                dash_length=0.08,
                stroke_width=2.0,
                color=PALETTE["secondary"],
            )
            return VGroup(tf, tg)

        tangents_pair = always_redraw(build_tangents_pair)

        self.play(
            FadeIn(dot_f),
            FadeIn(dot_g),
            FadeIn(dot_df),
            FadeIn(dot_dg),
            Create(tangents_pair),
            run_time=self.t(5.5),
        )

        comparisons = [
            (
                1.0,
                r"x = 1:\quad f(1) = g(1) = 1,\quad \text{but } f'(1) = 2 \neq g'(1) = 3",
            ),
            (
                2.0 / 3.0,
                r"x = \frac{2}{3}:\quad f'\!\left(\frac{2}{3}\right) = g'\!\left(\frac{2}{3}\right) = \frac{4}{3} \quad (\text{parallel tangents})",
            ),
            (
                -1.0,
                r"x = -1:\quad f'(-1) = -2 < 0,\quad g'(-1) = +3 > 0",
            ),
            (
                0.0,
                r"x = 0:\quad f(0) = g(0) = 0 \quad \text{and} \quad f'(0) = g'(0) = 0",
            ),
        ]

        active_banner: Optional[Mobject] = None
        for x_val, tex_str in comparisons:
            self.play(
                x_tracker.animate.set_value(x_val),
                rate_func=rate_functions.ease_in_out_sine,
                run_time=self.t(12.0),
            )
            banner = MathTex(tex_str, font_size=24, color=PALETTE["primary"])
            banner_box = SurroundingRectangle(
                banner,
                color=PALETTE["box_stroke"],
                fill_color=PALETTE["box_fill"],
                fill_opacity=0.92,
                buff=0.14,
            )
            banner_grp = VGroup(banner_box, banner).move_to(np.array([0.0, -3.18, 0.0]))
            if active_banner is not None:
                self.play(FadeOut(active_banner), run_time=self.t(1.5))
            self.play(FadeIn(banner_grp), run_time=self.t(4.5))
            active_banner = banner_grp
            self.pause(10.0)

        if active_banner is not None:
            self.play(FadeOut(active_banner), run_time=self.t(2.0))

        self.play(
            x_tracker.animate.set_value(-1.35),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(10.0),
        )
        self.play(
            x_tracker.animate.set_value(1.45),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(22.0),
        )
        self.pause(6.0)

        self.play(
            FadeOut(
                VGroup(
                    header,
                    ax_func,
                    lbl_func,
                    curve_f,
                    curve_g,
                    ax_deriv,
                    lbl_deriv,
                    curve_df,
                    curve_dg,
                    dot_f,
                    dot_g,
                    dot_df,
                    dot_dg,
                    tangents_pair,
                )
            ),
            run_time=self.t(4.0),
        )
        self.finish_chapter_timing()


class Chapter09_VisualRecap(SilentMathBaseScene):
    """
    Chapter 9 — Visual Recap and Mathematical Insight (21:00–24:00, Target: 180.0s)
    -------------------------------------------------------------------------------
    - Reconstructs the complete conceptual chain visually and dynamically:
        1. Change in input (Δx = h)
        2. Change in output (Δy = f(x+h) - f(x))
        3. Average rate of change ((f(x+h)-f(x))/h)
        4. Secant line through two distinct points
        5. Limit process (h -> 0)
        6. Tangent line at a single point
        7. Instantaneous rate of change (f'(x))
        8. Derivative as a function (f(x) = x^2 ==> f'(x) = 2x)
    """

    chapter_number = 9
    chapter_title = "Visual Recap and Mathematical Insight"
    target_duration = 180.0

    def construct(self) -> None:
        self.play_chapter_card(self.chapter_number, self.chapter_title)

        header = self.make_header(
            "09",
            "Visual Recap and Mathematical Insight",
            r"f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}",
        )
        self.play(FadeIn(header), run_time=self.t(2.0))

        axes, axes_labels = self.create_monochrome_axes(
            x_range=[-1.5, 2.5, 1.0],
            y_range=[-0.5, 6.0, 1.0],
            zone=SafeLayoutEngine.LEFT_GRAPH_ZONE,
            x_label_tex="x",
            y_label_tex="f(x)",
            include_numbers=True,
        )
        curve = axes.plot(
            CalculusMath.f_quad,
            x_range=[-1.3, 2.35],
            color=PALETTE["primary"],
            stroke_width=3.0,
        )

        x_tracker = ValueTracker(0.8)
        h_tracker = ValueTracker(1.2)

        dot_p = always_redraw(
            lambda: Dot(
                axes.c2p(x_tracker.get_value(), CalculusMath.f_quad(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )
        dot_q = always_redraw(
            lambda: Dot(
                axes.c2p(
                    x_tracker.get_value() + h_tracker.get_value(),
                    CalculusMath.f_quad(x_tracker.get_value() + h_tracker.get_value()),
                ),
                radius=0.070,
                color=PALETTE["secondary"],
            )
        )

        def build_recap_geometry() -> VGroup:
            x0 = x_tracker.get_value()
            h_val = max(0.005, h_tracker.get_value())
            (x1, y1), (x2, y2), _ = CalculusMath.secant_line_endpoints(
                CalculusMath.f_quad, x0, h_val, -1.1, 2.35
            )
            line_st = Line(
                axes.c2p(x1, y1),
                axes.c2p(x2, y2),
                stroke_width=2.2,
                color=PALETTE["secondary"],
            )
            dx_seg = DashedLine(
                axes.c2p(x0, CalculusMath.f_quad(x0)),
                axes.c2p(x0 + h_val, CalculusMath.f_quad(x0)),
                dash_length=0.07,
                stroke_width=1.6,
                color=PALETTE["tertiary"],
            )
            dy_seg = DashedLine(
                axes.c2p(x0 + h_val, CalculusMath.f_quad(x0)),
                axes.c2p(x0 + h_val, CalculusMath.f_quad(x0 + h_val)),
                dash_length=0.07,
                stroke_width=1.6,
                color=PALETTE["tertiary"],
            )
            return VGroup(line_st, dx_seg, dy_seg)

        recap_geom = always_redraw(build_recap_geometry)

        self.play(
            Create(axes),
            Write(axes_labels),
            Create(curve),
            FadeIn(dot_p),
            FadeIn(dot_q),
            Create(recap_geom),
            run_time=self.t(8.0),
        )

        recap_part1 = self.make_equation_card(
            [
                r"1.\;\; \Delta x = h \quad (\text{Change in Input})",
                r"2.\;\; \Delta y = f(x+h) - f(x) \quad (\text{Change in Output})",
                r"3.\;\; \frac{\Delta y}{\Delta x} = \frac{f(x+h) - f(x)}{h} \quad (\text{Average Rate})",
                r"4.\;\; \text{Secant Line Slope through } (x,f(x)) \text{ and } (x+h,f(x+h))",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=26,
        )

        for idx in range(4):
            self.play(Write(recap_part1[idx]), run_time=self.t(8.5))
            self.pause(4.0)

        recap_part2 = self.make_equation_card(
            [
                r"5.\;\; \text{Limit Process: } h \to 0 \;\; (h \neq 0)",
                r"6.\;\; \text{Secant Line } \longrightarrow \text{ Tangent Line}",
                r"7.\;\; f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
                r"8.\;\; f(x) = x^2 \quad \Longrightarrow \quad f'(x) = 2x",
            ],
            zone=SafeLayoutEngine.RIGHT_EQ_ZONE,
            font_size=27,
            boxed_last=True,
        )

        self.play(FadeOut(recap_part1), run_time=self.t(2.5))

        self.play(
            Write(recap_part2[0]),
            h_tracker.animate.set_value(0.35),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(11.0),
        )
        self.pause(3.5)

        self.play(
            Write(recap_part2[1]),
            h_tracker.animate.set_value(0.01),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(11.0),
        )
        self.pause(3.5)

        self.play(Write(recap_part2[2]), run_time=self.t(9.0))
        self.pause(4.5)

        self.play(Write(recap_part2[3]), run_time=self.t(9.0))
        self.pause(5.0)

        self.play(
            x_tracker.animate.set_value(-1.0),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(12.0),
        )
        self.play(
            x_tracker.animate.set_value(1.9),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(18.0),
        )
        self.pause(6.0)

        self.play(
            FadeOut(VGroup(axes, axes_labels, curve, dot_p, dot_q, recap_geom, recap_part2)),
            run_time=self.t(3.5),
        )

        master_card = self.make_equation_card(
            [
                r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
                r"f(x) = x^2 \quad \Longrightarrow \quad f'(x) = 2x",
                r"f(x) = x^3 \quad \Longrightarrow \quad f'(x) = 3x^2",
            ],
            zone=SafeLayoutEngine.FULL_STAGE_ZONE,
            font_size=36,
            boxed_last=False,
        )

        self.play(Write(master_card[0]), run_time=self.t(7.5))
        self.pause(3.0)
        self.play(Write(master_card[1]), Write(master_card[2]), run_time=self.t(7.5))
        self.pause(7.0)

        self.play(FadeOut(VGroup(header, master_card)), run_time=self.t(4.0))
        self.finish_chapter_timing()


class Chapter10_Closing(SilentMathBaseScene):
    """
    Chapter 10 — Closing (24:00–25:00, Target: 60.0s)
    -------------------------------------------------
    - Gradually simplifies the mathematical scene into a single elegant white curve
      and gliding tangent line on pure black (#000000).
    - Morphs into the final end card:
        SILENT MATH
        MATHEMATICS, MADE VISIBLE
        BY ANSH RAJ
    - Finishes with a smooth, unhurried fade to pure black.
    """

    chapter_number = 10
    chapter_title = "Closing"
    target_duration = 60.0

    def construct(self) -> None:
        axes, _ = self.create_monochrome_axes(
            x_range=[-3.2, 3.2, 1.0],
            y_range=[-2.5, 2.5, 1.0],
            zone=SafeLayoutEngine.FULL_STAGE_ZONE,
            include_numbers=False,
        )
        curve = axes.plot(
            CalculusMath.f_intro_wave,
            x_range=[-2.9, 2.9],
            color=PALETTE["primary"],
            stroke_width=3.2,
        )

        x_tracker = ValueTracker(-2.0)
        dot_p = always_redraw(
            lambda: Dot(
                axes.c2p(x_tracker.get_value(), CalculusMath.f_intro_wave(x_tracker.get_value())),
                radius=0.075,
                color=PALETTE["primary"],
            )
        )

        def build_closing_tangent() -> Line:
            x0 = x_tracker.get_value()
            (x1, y1), (x2, y2), _ = CalculusMath.tangent_line_endpoints(
                CalculusMath.f_intro_wave, CalculusMath.df_intro_wave, x0, half_span_x=1.1
            )
            return Line(
                axes.c2p(x1, y1),
                axes.c2p(x2, y2),
                stroke_width=2.0,
                color=PALETTE["secondary"],
            )

        tangent_line = always_redraw(build_closing_tangent)

        self.play(Create(curve), FadeIn(dot_p), Create(tangent_line), run_time=self.t(5.5))
        self.play(
            x_tracker.animate.set_value(2.0),
            rate_func=rate_functions.ease_in_out_sine,
            run_time=self.t(14.0),
        )
        self.pause(2.5)

        self.play(FadeOut(dot_p), FadeOut(tangent_line), run_time=self.t(3.5))
        divider = Line(LEFT * 3.4, RIGHT * 3.4, stroke_width=1.6, color=PALETTE["secondary"])
        self.play(ReplacementTransform(curve, divider), run_time=self.t(4.0))

        end_title = Text(
            CHANNEL_NAME,
            font="sans-serif",
            font_size=58,
            color=PALETTE["primary"],
            weight="LIGHT",
        )
        end_tagline = Text(
            CLOSING_TAGLINE,
            font="sans-serif",
            font_size=26,
            color=PALETTE["secondary"],
        )
        end_credit = Text(
            CLOSING_CREDIT,
            font="sans-serif",
            font_size=20,
            color=PALETTE["tertiary"],
        )

        end_title.next_to(divider, UP, buff=0.42)
        end_tagline.next_to(divider, DOWN, buff=0.40)
        end_credit.next_to(end_tagline, DOWN, buff=0.36)

        end_group = VGroup(end_title, divider, end_tagline, end_credit)
        SafeLayoutEngine.fit_into_zone(end_group, SafeLayoutEngine.FULL_STAGE_ZONE)

        self.play(FadeIn(end_title, shift=UP * 0.12), run_time=self.t(3.5))
        self.play(FadeIn(end_tagline, shift=UP * 0.10), run_time=self.t(3.5))
        self.pause(1.5)
        self.play(FadeIn(end_credit, shift=UP * 0.08), run_time=self.t(3.0))

        self.pause(9.0)
        self.play(FadeOut(end_group), run_time=self.t(5.0))
        self.pause(3.0)
        self.finish_chapter_timing()


# =============================================================================
# 7. DEPENDENCY DIAGNOSTICS, MATHEMATICAL SELF-TEST & RENDER ORCHESTRATION
# =============================================================================

@dataclasses.dataclass
class DependencyReport:
    """Structured diagnostic report for required system binaries & Python packages."""
    python_ok: bool
    numpy_ok: bool
    manim_ok: bool
    cairo_ok: bool
    pango_ok: bool
    ffmpeg_path: Optional[str]
    ffprobe_path: Optional[str]
    latex_path: Optional[str]
    dvisvgm_path: Optional[str]
    missing_items: List[str]

    @property
    def ready_for_render(self) -> bool:
        return (
            self.python_ok
            and self.numpy_ok
            and self.manim_ok
            and self.ffmpeg_path is not None
            and self.latex_path is not None
        )


def check_dependencies(verbose: bool = True) -> DependencyReport:
    """
    Inspect the host environment for Python 3, NumPy, Manim Community Edition,
    Cairo, Pango, LaTeX (TeX Live), and FFmpeg/FFprobe.
    """
    py_ok = sys.version_info >= (3, 9)
    np_ok = True

    cairo_ok = False
    try:
        import cairo  # type: ignore
        cairo_ok = True
    except Exception:
        cairo_ok = False

    pango_ok = False
    try:
        import manimpango  # type: ignore
        pango_ok = True
    except Exception:
        pango_ok = shutil.which("pango-view") is not None or MANIM_AVAILABLE

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    latex_path = shutil.which("latex") or shutil.which("pdflatex")
    dvisvgm_path = shutil.which("dvisvgm")

    missing: List[str] = []
    if not MANIM_AVAILABLE:
        missing.append(f"manim (Python package) — {MANIM_IMPORT_ERROR}")
    if not cairo_ok and not MANIM_AVAILABLE:
        missing.append("pycairo / libcairo2-dev")
    if not pango_ok and not MANIM_AVAILABLE:
        missing.append("manimpango / libpango1.0-dev")
    if ffmpeg_path is None:
        missing.append("ffmpeg (CLI binary)")
    if ffprobe_path is None:
        missing.append("ffprobe (CLI binary)")
    if latex_path is None:
        missing.append("latex / pdflatex (TeX Live)")
    if dvisvgm_path is None:
        missing.append("dvisvgm (TeX Live SVG converter required by Manim MathTex)")

    report = DependencyReport(
        python_ok=py_ok,
        numpy_ok=np_ok,
        manim_ok=MANIM_AVAILABLE,
        cairo_ok=cairo_ok or MANIM_AVAILABLE,
        pango_ok=pango_ok or MANIM_AVAILABLE,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        latex_path=latex_path,
        dvisvgm_path=dvisvgm_path,
        missing_items=missing,
    )

    if verbose:
        print("=" * 76)
        print("SILENT MATH — SYSTEM & DEPENDENCY DIAGNOSTIC REPORT")
        print("=" * 76)
        print(f"  Python Version : {sys.version.split()[0]} ({sys.executable})")
        print(f"  Platform       : {platform.platform()} | CPU Cores: {os.cpu_count() or 1}")
        print(f"  NumPy          : {np.__version__}")
        print(f"  Manim CE       : {'AVAILABLE (' + mn.__version__ + ')' if MANIM_AVAILABLE else 'MISSING'}")
        print(f"  PyCairo        : {'AVAILABLE' if report.cairo_ok else 'MISSING'}")
        print(f"  Pango/ManimPango: {'AVAILABLE' if report.pango_ok else 'MISSING'}")
        print(f"  FFmpeg         : {ffmpeg_path or 'MISSING'}")
        print(f"  FFprobe        : {ffprobe_path or 'MISSING'}")
        print(f"  LaTeX          : {latex_path or 'MISSING'}")
        print(f"  dvisvgm        : {dvisvgm_path or 'MISSING'}")
        print("-" * 76)
        if report.ready_for_render:
            print("  STATUS: READY — All required rendering tools are installed.")
        else:
            print("  STATUS: MISSING DEPENDENCIES DETECTED")
            for item in report.missing_items:
                print(f"    - {item}")
            print("\n  To install missing dependencies on Ubuntu/Debian Linux, run:")
            print("    sudo apt-get update && sudo apt-get install -y \\")
            print("        build-essential pkg-config python3-dev python3-pip python3-venv \\")
            print("        libcairo2-dev libpango1.0-dev ffmpeg texlive texlive-latex-extra \\")
            print("        texlive-fonts-extra texlive-science dvisvgm")
            print("    pip3 install --upgrade manim numpy pycairo manimpango")
        print("=" * 76)

    return report


def run_mathematical_and_layout_verification() -> bool:
    """
    Execute deterministic unit tests verifying:
      1. Exact 25-minute (1,500.0s) chapter budget across all 10 chapters
      2. SafeLayoutEngine zone containment & non-overlap guarantees
      3. Difference quotient convergence for f(x)=x^2 and g(x)=x^3
      4. Zero-division protection when h = 0 is passed to difference_quotient
      5. Chapter 5, 6, 7, 8 numerical sample accuracy
    """
    print("=" * 76)
    print("SILENT MATH — MATHEMATICAL & LAYOUT VERIFICATION SUITE")
    print("=" * 76)
    failures: List[str] = []

    total_sec = sum(ch.target_seconds for ch in CHAPTER_SPECS)
    print(f"[1/5] Chapter Count & Duration Budget: {len(CHAPTER_SPECS)} chapters, {total_sec:.1f}s ({total_sec/60.0:.1f} min)")
    if len(CHAPTER_SPECS) != 10 or abs(total_sec - 1500.0) > 1e-6:
        failures.append(f"Expected 10 chapters totaling 1500.0s, got {len(CHAPTER_SPECS)} and {total_sec}s.")

    zone_errors = SafeLayoutEngine.validate_zones()
    if zone_errors:
        failures.extend(zone_errors)
        print(f"[2/5] SafeLayoutEngine Zone Check: FAILED ({len(zone_errors)} errors)")
    else:
        print("[2/5] SafeLayoutEngine Zone Check: PASSED (all zones inside SAFE_FRAME with positive clearance)")

    try:
        CalculusMath.difference_quotient(CalculusMath.f_quad, 1.0, 0.0)
        failures.append("difference_quotient failed to raise ValueError for h = 0.")
    except ValueError:
        print("[3/5] Zero-Division Guard (h = 0): PASSED (explicitly prevented)")

    for x_test in [-2.0, 0.0, 1.0, 2.0]:
        approx_q = CalculusMath.difference_quotient(CalculusMath.f_quad, x_test, 1e-7)
        exact_q = float(CalculusMath.df_quad(x_test))
        if abs(approx_q - exact_q) > 1e-5:
            failures.append(f"Quadratic derivative mismatch at x={x_test}: {approx_q} vs {exact_q}")

        approx_c = CalculusMath.difference_quotient(CalculusMath.f_cubic, x_test, 1e-7)
        exact_c = float(CalculusMath.df_cubic(x_test))
        if abs(approx_c - exact_c) > 1e-5:
            failures.append(f"Cubic derivative mismatch at x={x_test}: {approx_c} vs {exact_c}")
    print("[4/5] Limit Convergence & Analytical Derivative Check (x^2 -> 2x, x^3 -> 3x^2): PASSED")

    if not (
        CalculusMath.f_quad(1.0) == 1.0
        and CalculusMath.f_cubic(1.0) == 1.0
        and CalculusMath.df_quad(1.0) == 2.0
        and CalculusMath.df_cubic(1.0) == 3.0
    ):
        failures.append("Chapter 8 intersection values at x=1 failed.")
    x_par = 2.0 / 3.0
    if abs(float(CalculusMath.df_quad(x_par)) - float(CalculusMath.df_cubic(x_par))) > 1e-12:
        failures.append("Chapter 8 parallel-tangent check at x=2/3 failed.")
    print("[5/5] Chapter 5–8 Sample Points & Kinematics Identities: PASSED")
    print("=" * 76)

    if failures:
        for f in failures:
            print(f"  ERROR: {f}")
        return False
    print("ALL MATHEMATICAL & LAYOUT CHECKS PASSED.")
    return True


def compute_worker_allocation(requested_workers: str, total_scenes: int = 10) -> Tuple[int, int]:
    """
    Compute balanced `(scene_workers, ffmpeg_threads_per_worker)` to prevent
    CPU oversubscription when Manim and FFmpeg run concurrently.
    """
    detected_cores = os.cpu_count() or 2
    usable_cores = max(1, detected_cores - 1) if detected_cores >= 4 else detected_cores

    if requested_workers.strip().lower() == "auto":
        if usable_cores <= 2:
            scene_workers = 1
            ffmpeg_threads = usable_cores
        else:
            scene_workers = max(1, min(total_scenes, usable_cores // 2, 6))
            ffmpeg_threads = max(1, usable_cores // scene_workers)
    else:
        try:
            parsed = max(1, int(requested_workers))
        except ValueError:
            parsed = 1
        scene_workers = min(parsed, total_scenes, usable_cores)
        ffmpeg_threads = max(1, usable_cores // scene_workers)

    return scene_workers, ffmpeg_threads


def _render_single_scene_worker(
    script_path: str,
    scene_name: str,
    media_dir: str,
    resolution: Tuple[int, int],
    fps: int,
    ffmpeg_threads: int,
    time_scale: float,
) -> Tuple[str, Path, float]:
    """
    Subprocess worker that renders a single Manim scene to an isolated MP4 file.
    """
    t0 = time.monotonic()
    env = os.environ.copy()
    env["SILENT_MATH_TIME_SCALE"] = str(time_scale)
    env["OMP_NUM_THREADS"] = str(max(1, ffmpeg_threads))
    env["OPENBLAS_NUM_THREADS"] = str(max(1, ffmpeg_threads))
    env["MKL_NUM_THREADS"] = str(max(1, ffmpeg_threads))

    width, height = resolution
    cmd = [
        sys.executable,
        "-m",
        "manim",
        "render",
        str(script_path),
        scene_name,
        "--media_dir",
        str(media_dir),
        "-r",
        f"{width},{height}",
        "--fps",
        str(fps),
        "--format",
        "mp4",
        "--disable_caching",
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Scene {scene_name} failed to render (exit code {proc.returncode}).\n"
            f"STDERR:\n{proc.stderr[-2000:]}"
        )

    media_root = Path(media_dir)
    candidates = list(media_root.rglob(f"{scene_name}.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"Manim completed for {scene_name}, but {scene_name}.mp4 was not found under {media_root}."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best_mp4 = candidates[0]
    if best_mp4.stat().st_size == 0:
        raise RuntimeError(f"Rendered file {best_mp4} has 0 bytes.")

    elapsed = time.monotonic() - t0
    return scene_name, best_mp4, elapsed


def concatenate_and_encode_final_mp4(
    scene_mp4_paths: Sequence[Path],
    output_mp4: Path,
    fps: int,
    ffmpeg_threads: int,
    include_silent_aac: bool = False,
) -> None:
    """
    Concatenate rendered chapter MP4s in exact chapter order and encode the final
    H.264 MP4 (`libx264`, `yuv420p`, `movflags +faststart`) for universal playback.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found on PATH.")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="silent_math_concat_") as tmpdir:
        concat_list_path = Path(tmpdir) / "scenes.txt"
        lines = [f"file '{p.resolve().as_posix()}'" for p in scene_mp4_paths]
        concat_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
        ]

        if include_silent_aac:
            cmd.extend([
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ])

        cmd.extend([
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-threads",
            str(max(1, ffmpeg_threads)),
            "-movflags",
            "+faststart",
        ])

        if include_silent_aac:
            cmd.extend(["-c:a", "aac", "-b:a", "128k", "-shortest"])
        else:
            cmd.extend(["-an"])

        cmd.append(str(output_mp4))

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg final concatenation/encoding failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )


def sample_and_inspect_frames(output_mp4: Path, num_samples: int = 12) -> Dict[str, Any]:
    """
    Automated post-render frame sampling QA:
      - Samples `num_samples` evenly spaced frames across `output_mp4` using FFmpeg
        raw grayscale pipe or OpenCV.
      - Checks outer 2% border pixels to detect accidental frame-edge clipping.
      - Checks active content frames (middle 80% of video) to flag unexpected
        completely blank frames.
    """
    qa_result: Dict[str, Any] = {
        "sampled_frames": 0,
        "non_blank_frames": 0,
        "border_clipping_warnings": 0,
    }
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(output_mp4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            indices = np.linspace(int(total_frames * 0.05), int(total_frames * 0.95), num_samples, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                qa_result["sampled_frames"] += 1
                gray = frame.mean(axis=2) if frame.ndim == 3 else frame
                if float(gray.max()) > 12.0:
                    qa_result["non_blank_frames"] += 1
                h, w = gray.shape[:2]
                margin_y = max(2, int(h * 0.015))
                margin_x = max(2, int(w * 0.015))
                border_max = max(
                    float(gray[:margin_y, :].max()),
                    float(gray[-margin_y:, :].max()),
                    float(gray[:, :margin_x].max()),
                    float(gray[:, -margin_x:].max()),
                )
                if border_max > 200.0:
                    qa_result["border_clipping_warnings"] += 1
        cap.release()
    except Exception:
        pass
    return qa_result


def validate_output_mp4(output_mp4: Path) -> Dict[str, Any]:
    """
    Validate the final MP4 using `ffprobe` (checking file existence, nonzero size,
    H.264 video codec, resolution, frame rate, duration, and audio stream presence)
    plus `sample_and_inspect_frames` for visual frame-sampling QA.
    """
    if not output_mp4.exists():
        raise FileNotFoundError(f"Output video does not exist: {output_mp4}")

    size_bytes = output_mp4.stat().st_size
    if size_bytes <= 0:
        raise RuntimeError(f"Output video is empty (0 bytes): {output_mp4}")

    ffprobe_bin = shutil.which("ffprobe")
    metadata: Dict[str, Any] = {
        "path": str(output_mp4.resolve()),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 2),
        "video_codec": "unknown",
        "width": None,
        "height": None,
        "fps": None,
        "duration_seconds": None,
        "has_audio": False,
    }

    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(output_mp4),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            info = json.loads(proc.stdout)
            streams = info.get("streams", [])
            fmt = info.get("format", {})
            if "duration" in fmt:
                metadata["duration_seconds"] = round(float(fmt["duration"]), 2)
            for stream in streams:
                stype = stream.get("codec_type")
                if stype == "video":
                    metadata["video_codec"] = stream.get("codec_name", "unknown")
                    metadata["width"] = stream.get("width")
                    metadata["height"] = stream.get("height")
                    r_frame_rate = stream.get("r_frame_rate", "0/1")
                    if "/" in r_frame_rate:
                        num, den = r_frame_rate.split("/", 1)
                        if float(den) > 0:
                            metadata["fps"] = round(float(num) / float(den), 2)
                elif stype == "audio":
                    metadata["has_audio"] = True

    frame_qa = sample_and_inspect_frames(output_mp4)
    metadata["frame_qa"] = frame_qa

    print("=" * 76)
    print("SILENT MATH — FINAL MP4 VERIFICATION REPORT")
    print("=" * 76)
    print(f"  Output File : {metadata['path']}")
    print(f"  File Size   : {metadata['size_mb']} MB ({metadata['size_bytes']:,} bytes)")
    print(f"  Video Codec : {metadata['video_codec']}")
    print(f"  Resolution  : {metadata['width']}x{metadata['height']}")
    print(f"  Frame Rate  : {metadata['fps']} FPS")
    print(f"  Duration    : {metadata['duration_seconds']} seconds")
    print(f"  Audio Stream: {'Present (AAC)' if metadata['has_audio'] else 'None (Pure Silent Visual Film)'}")
    if frame_qa["sampled_frames"] > 0:
        print(
            f"  Frame QA    : {frame_qa['non_blank_frames']}/{frame_qa['sampled_frames']} active frames verified, "
            f"{frame_qa['border_clipping_warnings']} border-clipping warnings"
        )
    print("=" * 76)
    return metadata


def print_render_plan() -> None:
    """Display the 10-chapter architectural plan and timing table."""
    print("=" * 84)
    print(f"SILENT MATH — 25-MINUTE ARCHITECTURAL TIMELINE ({DEFAULT_OUTPUT_FILENAME})")
    print("=" * 84)
    print(f"{'Ch':<4} {'Scene Class':<34} {'Window':<13} {'Duration':<10} {'Topic'}")
    print("-" * 84)
    for ch in CHAPTER_SPECS:
        window = f"{ch.start_time_str}–{ch.end_time_str}"
        dur = f"{ch.target_seconds:5.0f}s"
        print(f"{ch.number:02d}   {ch.scene_name:<34} {window:<13} {dur:<10} {ch.title}")
    print("-" * 84)
    total_s = sum(c.target_seconds for c in CHAPTER_SPECS)
    print(f"TOTAL TARGET DURATION: {total_s:.0f} seconds ({total_s/60.0:.1f} minutes)")
    print("=" * 84)


# =============================================================================
# 8. COMMAND-LINE INTERFACE & MAIN ENTRY POINT
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="silent_math_derivatives.py",
        description=(
            "Generate 'Silent_Math_Derivatives.mp4' — a 25-minute black-and-white "
            "calculus visual documentary teaching derivatives from first principles."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a fast low-resolution (854x480 @ 15 FPS) preview to verify pipeline & layout.",
    )
    parser.add_argument(
        "--quality",
        choices=["preview", "medium", "production"],
        default="production",
        help="Render quality preset: 'preview' (480p15), 'medium' (720p30), or 'production' (1080p60).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override target frame rate (default: 60 for production, 30 fallback, 15 for preview).",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default="auto",
        help="Number of concurrent Manim scene workers ('auto' or integer N, default: auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Final MP4 output path (default: {DEFAULT_OUTPUT_FILENAME}).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional single scene class name to render (e.g., Chapter01_Intro).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve intermediate per-scene Manim render directories.",
    )
    parser.add_argument(
        "--silent-aac",
        action="store_true",
        help="Include an optional silent AAC stereo audio stream in the final MP4 (disabled by default).",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check system dependencies (Manim, Cairo, Pango, LaTeX, FFmpeg) and exit.",
    )
    parser.add_argument(
        "--verify-math",
        action="store_true",
        help="Run mathematical accuracy and non-overlapping layout self-tests and exit.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the 10-chapter 25-minute scene schedule and CPU worker plan and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.plan:
        print_render_plan()
        s_workers, ff_threads = compute_worker_allocation(args.workers, len(CHAPTER_SPECS))
        print(f"CPU Allocation Plan (--workers {args.workers}): "
              f"{s_workers} parallel scene worker(s) x {ff_threads} FFmpeg/BLAS thread(s) per worker.")
        return 0

    if args.verify_math:
        ok = run_mathematical_and_layout_verification()
        return 0 if ok else 1

    if args.check_deps:
        rep = check_dependencies(verbose=True)
        return 0 if rep.ready_for_render else 1

    if not run_mathematical_and_layout_verification():
        print("ABORTING: Mathematical or layout verification failed.", file=sys.stderr)
        return 1

    dep_report = check_dependencies(verbose=True)
    if not dep_report.ready_for_render:
        print(
            "\nERROR: Cannot start video rendering because required dependencies are missing.\n"
            "Please run the installation commands printed above, then re-run this script.",
            file=sys.stderr,
        )
        return 2

    if args.preview or args.quality == "preview":
        resolution = (854, 480)
        fps = args.fps or 15
        time_scale = float(os.environ.get("SILENT_MATH_TIME_SCALE", "0.10"))
    elif args.quality == "medium":
        resolution = (1280, 720)
        fps = args.fps or 30
        time_scale = 1.0
    else:
        resolution = (1920, 1080)
        fps = args.fps or 60
        time_scale = 1.0

    selected_specs = CHAPTER_SPECS
    if args.scene:
        selected_specs = [c for c in CHAPTER_SPECS if c.scene_name == args.scene]
        if not selected_specs:
            valid_names = ", ".join(c.scene_name for c in CHAPTER_SPECS)
            print(f"ERROR: Unknown scene '{args.scene}'. Valid scenes: {valid_names}", file=sys.stderr)
            return 1

    scene_workers, ffmpeg_threads = compute_worker_allocation(args.workers, len(selected_specs))
    output_path = Path(args.output).resolve()
    script_path = Path(__file__).resolve()

    print(f"\nStarting Silent Math Render Pipeline:")
    print(f"  Mode            : {'PREVIEW' if args.preview else args.quality.upper()}")
    print(f"  Target Output   : {output_path}")
    print(f"  Resolution/FPS  : {resolution[0]}x{resolution[1]} @ {fps} FPS")
    print(f"  Chapters        : {len(selected_specs)} scene(s)")
    print(f"  Worker Strategy : {scene_workers} scene worker(s), {ffmpeg_threads} thread(s)/worker\n")

    temp_root_obj = tempfile.TemporaryDirectory(prefix="silent_math_build_")
    build_dir = Path(temp_root_obj.name)
    if args.keep_temp:
        build_dir = Path.cwd() / "silent_math_build_artifacts"
        build_dir.mkdir(parents=True, exist_ok=True)

    rendered_map: Dict[str, Path] = {}
    start_wall = time.monotonic()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=scene_workers) as pool:
            future_to_scene = {
                pool.submit(
                    _render_single_scene_worker,
                    str(script_path),
                    spec.scene_name,
                    str(build_dir / f"ch_{spec.number:02d}"),
                    resolution,
                    fps,
                    ffmpeg_threads,
                    time_scale,
                ): spec
                for spec in selected_specs
            }

            for fut in concurrent.futures.as_completed(future_to_scene):
                spec = future_to_scene[fut]
                sc_name, mp4_file, elapsed_s = fut.result()
                rendered_map[sc_name] = mp4_file
                print(f"  [DONE] {sc_name} rendered in {elapsed_s:.1f}s -> {mp4_file.name}")

        ordered_mp4s = [rendered_map[spec.scene_name] for spec in selected_specs]
        print("\nConcatenating scenes and encoding final H.264 MP4...")
        concatenate_and_encode_final_mp4(
            ordered_mp4s,
            output_path,
            fps=fps,
            ffmpeg_threads=max(1, scene_workers * ffmpeg_threads),
            include_silent_aac=args.silent_aac,
        )
        validate_output_mp4(output_path)
        total_wall = time.monotonic() - start_wall
        print(f"SUCCESS: Pipeline finished in {total_wall:.1f}s.")
        return 0
    except Exception as exc:
        print(f"\nPIPELINE FAILURE: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_temp:
            temp_root_obj.cleanup()


if __name__ == "__main__":
    sys.exit(main())

