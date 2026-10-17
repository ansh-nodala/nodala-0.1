#!/usr/bin/env python3
"""
4K Vertical Mathematical Animation (2160x3840, 30 FPS, 50.0 Seconds, 1500 Frames)
Visualizing how y = wx + b combined with sigmoid activations approximates nonlinear data.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from scipy.optimize import minimize


# =============================================================================
# 1. CONSTANTS & VIDEO SPECIFICATIONS
# =============================================================================

FPS: int = 30
DURATION_SEC: int = 50
TOTAL_FRAMES: int = FPS * DURATION_SEC  # 1500 frames

# Physical canvas dimensions in inches (9:16 aspect ratio)
# At DPI=300 -> 2160 x 3840 (4K UHD)
# At DPI=150 -> 1080 x 1920 (1080p FHD)
# At DPI=75  ->  540 x  960 (qHD fast preview)
FIG_WIDTH_IN: float = 7.2
FIG_HEIGHT_IN: float = 12.8

# Color Palette (3Blue1Brown-inspired dark mathematical palette)
COLOR_BG = "#000000"
COLOR_GRID = "#16181D"
COLOR_AXIS = "#5C6470"
COLOR_DATA = "#EAEEF3"
COLOR_DATA_GLOW = "#88A0B8"
COLOR_LINEAR = "#9AA5B1"
COLOR_RESIDUAL = "#4A525D"
COLOR_SIGMOID = "#38B6FF"
COLOR_NEURON_PALETTE = [
    "#2EC4B6", "#38B6FF", "#4895EF", "#4CC9F0", "#56CFE1",
    "#64DFDF", "#72EFDD", "#5E60CE", "#5390D9", "#4EA8DE",
]
COLOR_INITIAL_PRED = "#7C8594"
COLOR_FINAL_CURVE = "#FFB703"
COLOR_FINAL_GLOW = "#FB8500"
COLOR_TARGET_GHOST = "#3A404A"
COLOR_TEXT = "#F4F6F8"
COLOR_TEXT_DIM = "#8E9BAE"


# =============================================================================
# 2. SMOOTH EASING & INTERPOLATION UTILITIES
# =============================================================================

def clamp01(t: float) -> float:
    return max(0.0, min(1.0, float(t)))


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = clamp01(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def window_progress(frame: int, start_sec: float, end_sec: float) -> float:
    t = frame / float(FPS)
    if t <= start_sec:
        return 0.0
    if t >= end_sec:
        return 1.0
    return (t - start_sec) / (end_sec - start_sec)


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z_clipped = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


# =============================================================================
# 3. MATHEMATICAL MODEL & REAL NUMERICAL OPTIMIZATION
# =============================================================================

@dataclass
class MathSimulationData:
    x_grid: np.ndarray          # (K,) dense evaluation grid [-3.2, 3.2]
    y_target_grid: np.ndarray   # (K,) noise-free target function
    x_data: np.ndarray          # (N,) noisy sample x-coordinates
    y_data: np.ndarray          # (N,) noisy sample y-coordinates
    ols_w: float                # Ordinary Least Squares slope
    ols_b: float                # Ordinary Least Squares intercept
    num_neurons: int            # m = 10 hidden sigmoid neurons
    # Optimization trajectory arrays of shape (T_steps, ...)
    traj_w: np.ndarray          # (T_steps, m)
    traj_b: np.ndarray          # (T_steps, m)
    traj_v: np.ndarray          # (T_steps, m)
    traj_b_out: np.ndarray      # (T_steps,)
    traj_preds: np.ndarray      # (T_steps, K)
    traj_losses: np.ndarray     # (T_steps,)


def target_function(x: np.ndarray) -> np.ndarray:
    return 0.5 * x + 1.2 * np.sin(1.5 * x)


def forward_network(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    b_out: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes hidden activations H = sigma(w*x + b), weighted activations V*H,
    and combined scalar output y_hat.
    x: (K,)
    w, b, v: (m,)
    Returns:
      H: (m, K)
      VH: (m, K)
      y_hat: (K,)
    """
    z = w[:, None] * x[None, :] + b[:, None]  # (m, K)
    H = sigmoid(z)                            # (m, K)
    VH = v[:, None] * H                       # (m, K)
    y_hat = b_out + np.sum(VH, axis=0)        # (K,)
    return H, VH, y_hat


def precompute_mathematics(seed: int = 42, num_neurons: int = 10) -> MathSimulationData:
    """
    Generates synthetic noisy data and trains a single-hidden-layer sigmoid
    neural network using exact analytical gradients (Adam + L-BFGS-B refinement),
    then re-parameterizes the trajectory by prediction-space arc length for
    smooth, truthful visualization.
    """
    rng = np.random.default_rng(seed)

    # 1. Dense curve evaluation grid & synthetic training dataset
    x_grid = np.linspace(-3.2, 3.2, 400)
    y_target_grid = target_function(x_grid)

    n_samples = 42
    x_base = np.linspace(-2.85, 2.85, n_samples)
    x_jitter = rng.uniform(-0.04, 0.04, size=n_samples)
    x_data = np.clip(np.sort(x_base + x_jitter), -2.95, 2.95)
    noise = rng.normal(0.0, 0.135, size=n_samples)
    y_data = target_function(x_data) + noise

    # 2. Ordinary Least Squares linear fit y = w_ols * x + b_ols
    A_ols = np.vstack([x_data, np.ones_like(x_data)]).T
    ols_params, _, _, _ = np.linalg.lstsq(A_ols, y_data, rcond=None)
    ols_w, ols_b = float(ols_params[0]), float(ols_params[1])

    # 3. Initialize 10 sigmoid neurons with distributed inflection points
    m = num_neurons
    centers = np.linspace(-2.6, 2.6, m)
    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(m)])
    w_init = signs * np.linspace(1.6, 2.4, m)
    b_init = -w_init * centers

    # Solve a heavily regularized ridge problem for small initial output weights
    # so the initial prediction at t=32s is a visible, gentle under-fitted wave
    H_init_data = sigmoid(w_init[:, None] * x_data[None, :] + b_init[:, None])  # (m, N)
    Phi = np.vstack([H_init_data, np.ones((1, n_samples))]).T  # (N, m+1)
    ridge_reg = 18.0 * np.eye(m + 1)
    ridge_reg[-1, -1] = 1.0
    theta_linear_init = np.linalg.solve(Phi.T @ Phi + ridge_reg, Phi.T @ y_data)
    v_init = theta_linear_init[:m]
    b_out_init = float(theta_linear_init[m])

    def pack_params(w: np.ndarray, b: np.ndarray, v: np.ndarray, b_out: float) -> np.ndarray:
        return np.concatenate([w, b, v, [b_out]])

    def unpack_params(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        w = theta[0:m]
        b = theta[m:2 * m]
        v = theta[2 * m:3 * m]
        b_out = float(theta[3 * m])
        return w, b, v, b_out

    def loss_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        w, b, v, b_out = unpack_params(theta)
        z = w[:, None] * x_data[None, :] + b[:, None]  # (m, N)
        H = sigmoid(z)                                  # (m, N)
        dH = H * (1.0 - H)                              # (m, N)
        y_pred = b_out + np.sum(v[:, None] * H, axis=0) # (N,)
        err = y_pred - y_data                           # (N,)

        mse = float(np.mean(err ** 2))
        reg = 1e-5 * float(np.sum(v ** 2) + 0.02 * np.sum(w ** 2) + 0.01 * np.sum(b ** 2))
        total_loss = mse + reg

        scale = 2.0 / n_samples
        grad_b_out = scale * float(np.sum(err))
        grad_v = scale * (H @ err) + 2e-5 * v
        dL_dz = (v[:, None] * dH) * (scale * err[None, :])  # (m, N)
        grad_w = np.sum(dL_dz * x_data[None, :], axis=1) + 4e-7 * w
        grad_b = np.sum(dL_dz, axis=1) + 2e-7 * b

        grad = pack_params(grad_w, grad_b, grad_v, grad_b_out)
        return total_loss, grad

    # 4. Run real numerical optimization: Adam trajectory + L-BFGS-B polish
    raw_thetas: List[np.ndarray] = []
    theta = pack_params(w_init.copy(), b_init.copy(), v_init.copy(), b_out_init)
    raw_thetas.append(theta.copy())

    m_adam = np.zeros_like(theta)
    v_adam = np.zeros_like(theta)
    lr = 0.032
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, 550):
        _, grad = loss_and_grad(theta)
        m_adam = beta1 * m_adam + (1.0 - beta1) * grad
        v_adam = beta2 * v_adam + (1.0 - beta2) * (grad ** 2)
        m_hat = m_adam / (1.0 - beta1 ** step)
        v_hat = v_adam / (1.0 - beta2 ** step)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        raw_thetas.append(theta.copy())

    # L-BFGS-B fine convergence
    def lbfgs_callback(xk: np.ndarray) -> None:
        raw_thetas.append(xk.copy())

    minimize(
        loss_and_grad,
        theta,
        method="L-BFGS-B",
        jac=True,
        callback=lbfgs_callback,
        options={"maxiter": 80, "ftol": 1e-10},
    )

    # 5. Reparameterize recorded trajectory by prediction-curve arc-length
    raw_thetas_arr = np.array(raw_thetas)
    raw_preds = []
    raw_losses = []
    for th in raw_thetas_arr:
        w, b, v, b_out = unpack_params(th)
        _, _, y_hat_grid = forward_network(x_grid, w, b, v, b_out)
        loss_val, _ = loss_and_grad(th)
        raw_preds.append(y_hat_grid)
        raw_losses.append(loss_val)

    raw_preds_arr = np.array(raw_preds)  # (M_raw, K)
    diffs = np.linalg.norm(np.diff(raw_preds_arr, axis=0), axis=1)
    cum_arc = np.concatenate([[0.0], np.cumsum(diffs)])
    total_arc = max(cum_arc[-1], 1e-8)
    norm_arc = cum_arc / total_arc

    # Resample onto 200 uniform arc-length milestones
    num_milestones = 200
    target_arcs = np.linspace(0.0, 1.0, num_milestones)
    traj_thetas = np.zeros((num_milestones, raw_thetas_arr.shape[1]))
    for d in range(raw_thetas_arr.shape[1]):
        traj_thetas[:, d] = np.interp(target_arcs, norm_arc, raw_thetas_arr[:, d])

    traj_w = traj_thetas[:, 0:m]
    traj_b = traj_thetas[:, m:2 * m]
    traj_v = traj_thetas[:, 2 * m:3 * m]
    traj_b_out = traj_thetas[:, 3 * m]

    traj_preds = np.zeros((num_milestones, len(x_grid)))
    traj_losses = np.zeros(num_milestones)
    for idx in range(num_milestones):
        _, _, traj_preds[idx] = forward_network(
            x_grid, traj_w[idx], traj_b[idx], traj_v[idx], float(traj_b_out[idx])
        )
        traj_losses[idx], _ = loss_and_grad(traj_thetas[idx])

    return MathSimulationData(
        x_grid=x_grid,
        y_target_grid=y_target_grid,
        x_data=x_data,
        y_data=y_data,
        ols_w=ols_w,
        ols_b=ols_b,
        num_neurons=m,
        traj_w=traj_w,
        traj_b=traj_b,
        traj_v=traj_v,
        traj_b_out=traj_b_out,
        traj_preds=traj_preds,
        traj_losses=traj_losses,
    )


# =============================================================================
# 4. SCENE & FRAME RENDERING ENGINE (ARTIST MUTATION, NO AX.CLEAR)
# =============================================================================

class MathAnimator:
    """
    Manages the Matplotlib FigureCanvasAgg and updates pre-allocated artists
    for any frame index in [0, 1499].
    """

    def __init__(self, sim: MathSimulationData, dpi: int = 300) -> None:
        self.sim = sim
        self.dpi = dpi

        # Configure clean STIX / Computer Modern mathematical typography
        plt.rcParams.update({
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "mathtext.fontset": "cm",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": COLOR_AXIS,
            "axes.linewidth": 1.2,
        })

        self.fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=self.dpi)
        self.canvas = FigureCanvasAgg(self.fig)

        # Full-canvas axis in normalized [0, 1] coordinates for crisp typography
        # and explicit world-to-screen coordinate mapping inside the safe zone
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=COLOR_BG)
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.axis("off")

        # World coordinate bounds for the primary mathematical plane
        self.x_min, self.x_max = -3.3, 3.3
        self.y_min, self.y_max = -2.8, 2.8

        # Central mobile-safe viewport box in normalized figure coordinates
        self.vp_left, self.vp_right = 0.10, 0.90
        self.vp_bottom, self.vp_top = 0.18, 0.75
        self.camera_zoom = 1.0

        self._init_artists()

    def world_to_screen(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """Maps mathematical plane coordinates (x, y) into normalized [0, 1] screen coords."""
        cx = 0.5 * (self.vp_left + self.vp_right)
        cy = 0.5 * (self.vp_bottom + self.vp_top)
        half_w = 0.5 * (self.vp_right - self.vp_left) * self.camera_zoom
        half_h = 0.5 * (self.vp_top - self.vp_bottom) * self.camera_zoom

        nx = (x / 3.3) * half_w + cx
        ny = (y / 2.8) * half_h + cy
        return nx, ny

    def _init_artists(self) -> None:
        # 1. Background Grid Lines
        self.grid_lines_x = []
        for gx in [-3, -2, -1, 1, 2, 3]:
            (line,) = self.ax.plot([], [], color=COLOR_GRID, lw=0.9, alpha=0.0, zorder=1)
            self.grid_lines_x.append((gx, line))

        self.grid_lines_y = []
        for gy in [-2, -1, 1, 2]:
            (line,) = self.ax.plot([], [], color=COLOR_GRID, lw=0.9, alpha=0.0, zorder=1)
            self.grid_lines_y.append((gy, line))

        # 2. Primary Coordinate Axes & Tick Marks
        (self.x_axis_line,) = self.ax.plot([], [], color=COLOR_AXIS, lw=1.6, alpha=0.0, zorder=2)
        (self.y_axis_line,) = self.ax.plot([], [], color=COLOR_AXIS, lw=1.6, alpha=0.0, zorder=2)

        self.tick_lines = []
        for tx in [-3, -2, -1, 1, 2, 3]:
            (tline,) = self.ax.plot([], [], color=COLOR_AXIS, lw=1.2, alpha=0.0, zorder=2)
            self.tick_lines.append(("x", tx, tline))
        for ty in [-2, -1, 1, 2]:
            (tline,) = self.ax.plot([], [], color=COLOR_AXIS, lw=1.2, alpha=0.0, zorder=2)
            self.tick_lines.append(("y", ty, tline))

        # Axis symbolic labels (x and y)
        self.label_x_axis = self.ax.text(
            0.89, 0.445, r"$x$", color=COLOR_TEXT_DIM, fontsize=15,
            ha="center", va="top", alpha=0.0, zorder=3
        )
        self.label_y_axis = self.ax.text(
            0.52, 0.74, r"$y$", color=COLOR_TEXT_DIM, fontsize=15,
            ha="left", va="center", alpha=0.0, zorder=3
        )

        # Sigmoid asymptote reference lines (y = 0 and y = 1)
        (self.asymptote_top,) = self.ax.plot(
            [], [], color=COLOR_SIGMOID, lw=1.0, ls="--", alpha=0.0, zorder=2
        )
        (self.asymptote_bot,) = self.ax.plot(
            [], [], color=COLOR_SIGMOID, lw=1.0, ls="--", alpha=0.0, zorder=2
        )
        self.label_asym_1 = self.ax.text(
            0.47, 0.56, r"$1$", color=COLOR_SIGMOID, fontsize=13,
            ha="right", va="bottom", alpha=0.0, zorder=3
        )
        self.label_asym_0 = self.ax.text(
            0.47, 0.445, r"$0$", color=COLOR_SIGMOID, fontsize=13,
            ha="right", va="top", alpha=0.0, zorder=3
        )

        # 3. Residual Segments (for Scene 2 linear fit mismatch)
        self.residual_lines = []
        for _ in range(len(self.sim.x_data)):
            (rline,) = self.ax.plot(
                [], [], color=COLOR_RESIDUAL, lw=1.0, ls=":", alpha=0.0, zorder=3
            )
            self.residual_lines.append(rline)

        # 4. Individual Hidden Neuron Curves (m = 10)
        self.neuron_lines = []
        for i in range(self.sim.num_neurons):
            col = COLOR_NEURON_PALETTE[i % len(COLOR_NEURON_PALETTE)]
            (nline,) = self.ax.plot([], [], color=col, lw=1.8, alpha=0.0, zorder=4)
            self.neuron_lines.append(nline)

        # 5. Target Ghost Curve & Initial Prediction Reference Curve
        (self.target_ghost_line,) = self.ax.plot(
            [], [], color=COLOR_TARGET_GHOST, lw=1.5, ls="--", alpha=0.0, zorder=3
        )
        (self.initial_pred_line,) = self.ax.plot(
            [], [], color=COLOR_INITIAL_PRED, lw=1.8, ls="--", alpha=0.0, zorder=5
        )

        # 6. Linear / Sigmoid Single Curve (Scenes 2 & 3)
        (self.single_model_glow,) = self.ax.plot(
            [], [], color=COLOR_LINEAR, lw=5.5, alpha=0.0, zorder=5, solid_capstyle="round"
        )
        (self.single_model_line,) = self.ax.plot(
            [], [], color=COLOR_LINEAR, lw=2.6, alpha=0.0, zorder=6, solid_capstyle="round"
        )

        # 7. Combined Output Curve y_hat (Scenes 4, 5, 6)
        (self.combined_glow_line,) = self.ax.plot(
            [], [], color=COLOR_FINAL_GLOW, lw=7.0, alpha=0.0, zorder=7, solid_capstyle="round"
        )
        (self.combined_main_line,) = self.ax.plot(
            [], [], color=COLOR_FINAL_CURVE, lw=3.0, alpha=0.0, zorder=8, solid_capstyle="round"
        )

        # 8. Data Points (Glow + Core Scatter)
        self.data_glow_scatter = self.ax.scatter(
            [], [], s=75, c=COLOR_DATA_GLOW, alpha=0.0, edgecolors="none", zorder=9
        )
        self.data_core_scatter = self.ax.scatter(
            [], [], s=24, c=COLOR_DATA, alpha=0.0, edgecolors="none", zorder=10
        )

        # 9. Essential Mathematical Equations (Top Safe Area: y in [0.80, 0.90])
        self.eq_primary = self.ax.text(
            0.50, 0.87, "", color=COLOR_TEXT, fontsize=25,
            ha="center", va="center", alpha=0.0, zorder=12
        )
        self.eq_secondary = self.ax.text(
            0.50, 0.81, "", color=COLOR_SIGMOID, fontsize=19,
            ha="center", va="center", alpha=0.0, zorder=12
        )
        self.eq_params = self.ax.text(
            0.50, 0.12, "", color=COLOR_TEXT_DIM, fontsize=16,
            ha="center", va="center", alpha=0.0, zorder=12
        )

    def _update_axes_geometry(self, axis_progress: float, grid_alpha: float) -> None:
        # X and Y axes grow outward from origin
        span_x = 3.2 * ease_out_cubic(axis_progress)
        span_y = 2.6 * ease_out_cubic(axis_progress)

        sx, sy = self.world_to_screen(np.array([-span_x, span_x]), np.array([0.0, 0.0]))
        self.x_axis_line.set_data(sx, sy)
        self.x_axis_line.set_alpha(clamp01(axis_progress * 1.5))

        yx, yy = self.world_to_screen(np.array([0.0, 0.0]), np.array([-span_y, span_y]))
        self.y_axis_line.set_data(yx, yy)
        self.y_axis_line.set_alpha(clamp01(axis_progress * 1.5))

        # Ticks
        tick_alpha = clamp01((axis_progress - 0.5) * 2.0) * 0.85
        for kind, val, tline in self.tick_lines:
            if kind == "x":
                tx, ty = self.world_to_screen(np.array([val, val]), np.array([-0.06, 0.06]))
            else:
                tx, ty = self.world_to_screen(np.array([-0.06, 0.06]), np.array([val, val]))
            tline.set_data(tx, ty)
            tline.set_alpha(tick_alpha)

        lx, ly = self.world_to_screen(3.15, -0.18)
        self.label_x_axis.set_position((lx, ly))
        self.label_x_axis.set_alpha(tick_alpha)

        lyx, lyy = self.world_to_screen(0.18, 2.48)
        self.label_y_axis.set_position((lyx, lyy))
        self.label_y_axis.set_alpha(tick_alpha)

        # Grid lines
        for gx, gline in self.grid_lines_x:
            gx_s, gy_s = self.world_to_screen(np.array([gx, gx]), np.array([-2.6, 2.6]))
            gline.set_data(gx_s, gy_s)
            gline.set_alpha(grid_alpha)

        for gy, gline in self.grid_lines_y:
            gx_s, gy_s = self.world_to_screen(np.array([-3.2, 3.2]), np.array([gy, gy]))
            gline.set_data(gx_s, gy_s)
            gline.set_alpha(grid_alpha)

    def _set_data_points(self, reveal_progress: float, alpha_scale: float = 1.0) -> None:
        n_total = len(self.sim.x_data)
        if reveal_progress <= 0.0 or alpha_scale <= 0.0:
            self.data_core_scatter.set_offsets(np.empty((0, 2)))
            self.data_glow_scatter.set_offsets(np.empty((0, 2)))
            return

        # Staggered point reveal from left to right
        n_visible = int(np.ceil(reveal_progress * n_total))
        n_visible = max(1, min(n_total, n_visible))

        xs = self.sim.x_data[:n_visible]
        ys = self.sim.y_data[:n_visible]
        sx, sy = self.world_to_screen(xs, ys)
        offsets = np.column_stack([sx, sy])

        self.data_core_scatter.set_offsets(offsets)
        self.data_glow_scatter.set_offsets(offsets)
        self.data_core_scatter.set_alpha(0.92 * alpha_scale)
        self.data_glow_scatter.set_alpha(0.24 * alpha_scale)

    def _hide_Transient_artists(self) -> None:
        self.asymptote_top.set_alpha(0.0)
        self.asymptote_bot.set_alpha(0.0)
        self.label_asym_1.set_alpha(0.0)
        self.label_asym_0.set_alpha(0.0)
        for rline in self.residual_lines:
            rline.set_alpha(0.0)
        for nline in self.neuron_lines:
            nline.set_alpha(0.0)
        self.target_ghost_line.set_alpha(0.0)
        self.initial_pred_line.set_alpha(0.0)
        self.single_model_line.set_alpha(0.0)
        self.single_model_glow.set_alpha(0.0)
        self.combined_main_line.set_alpha(0.0)
        self.combined_glow_line.set_alpha(0.0)
        self.eq_primary.set_alpha(0.0)
        self.eq_secondary.set_alpha(0.0)
        self.eq_params.set_alpha(0.0)

    def update_frame(self, frame: int) -> None:
        t = frame / float(FPS)
        self._hide_Transient_artists()
        self.camera_zoom = 1.0

        # Final subtle camera zoom-in during 43–50s
        if t >= 43.0:
            z_prog = smootherstep(window_progress(frame, 43.0, 47.0))
            self.camera_zoom = 1.0 + 0.055 * z_prog

        # Base axes and grid visibility across the 50s timeline
        axis_prog = window_progress(frame, 0.2, 1.6)
        grid_in = smoothstep(window_progress(frame, 0.8, 2.0))
        grid_out = smoothstep(window_progress(frame, 43.0, 45.5))
        grid_alpha = 0.85 * grid_in * (1.0 - 0.65 * grid_out)
        self._update_axes_geometry(axis_prog, grid_alpha)

        x_grid = self.sim.x_grid

        # =====================================================================
        # SCENE 1: 0–6s | THE DATA
        # =====================================================================
        if t < 6.0:
            pts_prog = ease_in_out_cubic(window_progress(frame, 1.4, 4.9))
            self._set_data_points(pts_prog, alpha_scale=1.0)
            return

        # =====================================================================
        # SCENE 2: 6–13s | THE LINEAR MODEL (y = wx + b)
        # =====================================================================
        if t < 13.0:
            self._set_data_points(1.0, alpha_scale=1.0)

            eq_in = smoothstep(window_progress(frame, 6.0, 6.9))
            self.eq_primary.set_text(r"$y = wx + b$")
            self.eq_primary.set_alpha(eq_in)

            # Animate slope w (7.0s -> 10.2s) and intercept b (10.2s -> 12.6s)
            if t < 7.2:
                w_val = 0.0
                b_val = 0.0
            elif t < 8.4:
                p = smootherstep(window_progress(frame, 7.2, 8.4))
                w_val = 0.0 + 1.05 * p
                b_val = 0.0
            elif t < 9.4:
                p = smootherstep(window_progress(frame, 8.4, 9.4))
                w_val = 1.05 - 1.50 * p
                b_val = 0.0
            elif t < 10.2:
                p = smootherstep(window_progress(frame, 9.4, 10.2))
                w_val = -0.45 + (self.sim.ols_w + 0.45) * p
                b_val = 0.0
            elif t < 11.1:
                p = smootherstep(window_progress(frame, 10.2, 11.1))
                w_val = self.sim.ols_w
                b_val = 0.0 + 1.15 * p
            elif t < 12.0:
                p = smootherstep(window_progress(frame, 11.1, 12.0))
                w_val = self.sim.ols_w
                b_val = 1.15 - 2.10 * p
            else:
                p = smootherstep(window_progress(frame, 12.0, 12.7))
                w_val = self.sim.ols_w
                b_val = -0.95 + (self.sim.ols_b + 0.95) * p

            line_draw_prog = ease_out_cubic(window_progress(frame, 6.2, 7.1))
            n_pts = max(2, int(line_draw_prog * len(x_grid)))
            xg_sub = x_grid[:n_pts]
            yg_sub = w_val * xg_sub + b_val

            sx, sy = self.world_to_screen(xg_sub, yg_sub)
            self.single_model_line.set_data(sx, sy)
            self.single_model_Line_color = COLOR_LINEAR
            self.single_model_line.set_color(COLOR_LINEAR)
            self.single_model_line.set_alpha(0.95 * eq_in)

            self.single_model_glow.set_data(sx, sy)
            self.single_model_glow.set_color(COLOR_LINEAR)
            self.single_model_glow.set_alpha(0.18 * eq_in)

            # Show residual error segments once line is drawn
            res_alpha = 0.55 * smoothstep(window_progress(frame, 7.3, 8.0)) * (
                1.0 - smoothstep(window_progress(frame, 12.5, 13.0))
            )
            if res_alpha > 0.01:
                for idx, rline in enumerate(self.residual_lines):
                    xd = self.sim.x_data[idx]
                    yd = self.sim.y_data[idx]
                    yp = w_val * xd + b_val
                    rx, ry = self.world_to_screen(np.array([xd, xd]), np.array([yd, yp]))
                    rline.set_data(rx, ry)
                    rline.set_alpha(res_alpha)

            self.eq_params.set_text(rf"$w = {w_val:+.2f} \qquad b = {b_val:+.2f}$")
            self.eq_params.set_alpha(0.85 * eq_in)
            return

        # =====================================================================
        # SCENE 3: 13–21s | NONLINEAR ACTIVATION (\sigma(z))
        # =====================================================================
        if t < 21.0:
            # Dim the data points slightly so the activation curve transformation stands out
            dim_factor = 1.0 - 0.65 * smoothstep(window_progress(frame, 13.0, 14.2))
            self._set_data_points(1.0, alpha_scale=dim_factor)

            self.eq_primary.set_text(r"$z = wx + b$")
            self.eq_primary.set_alpha(0.95)

            sig_eq_alpha = smoothstep(window_progress(frame, 13.4, 14.5))
            self.eq_secondary.set_text(r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")
            self.eq_secondary.set_alpha(sig_eq_alpha)

            # Morph from linear z = wx + b into sigmoid \sigma(wx + b) between 14.2s and 16.5s
            morph_alpha = smootherstep(window_progress(frame, 14.2, 16.5))

            if t < 16.5:
                p_w = smootherstep(window_progress(frame, 13.0, 14.2))
                w_val = self.sim.ols_w + (1.6 - self.sim.ols_w) * p_w
                b_val = self.sim.ols_b * (1.0 - p_w)
            elif t < 18.0:
                p = smootherstep(window_progress(frame, 16.5, 18.0))
                w_val = 1.6 + 2.6 * p
                b_val = 0.0
            elif t < 19.5:
                p = smootherstep(window_progress(frame, 18.0, 19.5))
                w_val = 4.2 - 6.8 * p
                b_val = -2.2 * np.sin(math.pi * p)
            else:
                p = smootherstep(window_progress(frame, 19.5, 20.8))
                w_val = -2.6 + 4.6 * p
                b_val = 1.8 * np.sin(math.pi * p)

            z_grid = w_val * x_grid + b_val
            y_lin = np.clip(z_grid, -2.7, 2.7)
            y_sig = sigmoid(z_grid)
            y_blend = (1.0 - morph_alpha) * y_lin + morph_alpha * y_sig

            sx, sy = self.world_to_screen(x_grid, y_blend)

            # Blend color from slate gray (COLOR_LINEAR) to cyan (COLOR_SIGMOID)
            self.single_model_line.set_data(sx, sy)
            self.single_model_line.set_color(COLOR_SIGMOID if morph_alpha > 0.35 else COLOR_LINEAR)
            self.single_model_line.set_alpha(0.95)

            self.single_model_glow.set_data(sx, sy)
            self.single_model_glow.set_color(COLOR_SIGMOID if morph_alpha > 0.35 else COLOR_LINEAR)
            self.single_model_glow.set_alpha(0.22)

            # Asymptote guides y = 0 and y = 1
            asym_alpha = 0.45 * morph_alpha * (1.0 - smoothstep(window_progress(frame, 20.2, 21.0)))
            ax1_x, ax1_y = self.world_to_screen(np.array([-3.2, 3.2]), np.array([1.0, 1.0]))
            ax0_x, ax0_y = self.world_to_screen(np.array([-3.2, 3.2]), np.array([0.0, 0.0]))
            self.asymptote_top.set_data(ax1_x, ax1_y)
            self.asymptote_top.set_alpha(asym_alpha)
            self.asymptote_bot.set_data(ax0_x, ax0_y)
            self.asymptote_bot.set_alpha(asym_alpha)

            l1x, l1y = self.world_to_screen(-0.15, 1.04)
            l0x, l0y = self.world_to_screen(-0.15, -0.06)
            self.label_asym_1.set_position((l1x, l1y))
            self.label_asym_1.set_alpha(asym_alpha * 1.6)
            self.label_asym_0.set_position((l0x, l0y))
            self.label_asym_0.set_alpha(asym_alpha * 1.6)

            self.eq_params.set_text(rf"$w = {w_val:+.2f} \qquad b = {b_val:+.2f}$")
            self.eq_params.set_alpha(0.85 * (1.0 - smoothstep(window_progress(frame, 20.3, 21.0))))
            return

        # =====================================================================
        # SCENE 4: 21–32s | MULTIPLE NEURONS (h_i = \sigma(w_i x + b_i) -> v_i h_i)
        # =====================================================================
        if t < 32.0:
            # Restore data points smoothly as we prepare to combine neurons
            data_alpha = 0.35 + 0.65 * smoothstep(window_progress(frame, 26.5, 29.0))
            self._set_data_points(1.0, alpha_scale=data_alpha)

            w0 = self.sim.traj_w[0]
            b0 = self.sim.traj_b[0]
            v0 = self.sim.traj_v[0]
            b_out0 = float(self.sim.traj_b_out[0])

            H0, VH0, y_hat0 = forward_network(x_grid, w0, b0, v0, b_out0)

            # Weighting transition: from raw h_i(x) in (0,1) to scaled v_i * h_i(x)
            weight_blend = smootherstep(window_progress(frame, 25.0, 28.2))

            if t < 25.0:
                self.eq_primary.set_text(r"$h_i = \sigma(w_i x + b_i)$")
                self.eq_primary.set_alpha(smoothstep(window_progress(frame, 21.0, 21.8)))
            elif t < 28.5:
                self.eq_primary.set_text(r"$v_i h_i = v_i \sigma(w_i x + b_i)$")
                self.eq_primary.set_alpha(0.95)
            else:
                self.eq_primary.set_text(r"$\hat{y} = b_{\mathrm{out}} + \sum_{i=1}^{m} v_i h_i$")
                self.eq_primary.set_alpha(0.95)

            # Staggered reveal of each of the m = 10 hidden neuron curves
            for i, nline in enumerate(self.neuron_lines):
                t_start = 21.2 + i * 0.28
                t_end = t_start + 0.85
                n_prog = ease_out_cubic(window_progress(frame, t_start, t_end))
                if n_prog <= 0.0:
                    nline.set_alpha(0.0)
                    continue

                curve_y = (1.0 - weight_blend) * H0[i] + weight_blend * VH0[i]
                n_pts = max(2, int(n_prog * len(x_grid)))
                sx, sy = self.world_to_screen(x_grid[:n_pts], curve_y[:n_pts])
                nline.set_data(sx, sy)
                nline.set_alpha(0.62 * min(1.0, n_prog * 1.5))

            # Progressive summation preview between 28.5s and 32.0s
            if t >= 28.5:
                sum_prog = smootherstep(window_progress(frame, 28.5, 31.8))
                # Partial sum accumulation across the m neurons
                exact_k = sum_prog * self.sim.num_neurons
                k_full = int(math.floor(exact_k))
                k_frac = exact_k - k_full

                partial_y = np.full_like(x_grid, b_out0 * sum_prog)
                for i in range(min(k_full, self.sim.num_neurons)):
                    partial_y += VH0[i]
                if k_full < self.sim.num_neurons:
                    partial_y += k_frac * VH0[k_full]

                sx, sy = self.world_to_screen(x_grid, partial_y)
                self.combined_main_line.set_data(sx, sy)
                self.combined_main_line.set_color(COLOR_INITIAL_PRED)
                self.combined_main_line.set_alpha(0.90 * smoothstep(window_progress(frame, 28.5, 29.3)))

            return

        # =====================================================================
        # SCENE 5: 32–43s | BUILDING & OPTIMIZING THE CURVE
        # =====================================================================
        if t < 43.0:
            self._set_data_points(1.0, alpha_scale=1.0)

            self.eq_primary.set_text(r"$\hat{y} = b_{\mathrm{out}} + \sum_{i=1}^{m} v_i h_i$")
            self.eq_primary.set_alpha(0.95)

            # Keep initial prediction visible as a dashed reference curve
            y_init_grid = self.sim.traj_preds[0]
            ix, iy = self.world_to_screen(x_grid, y_init_grid)
            self.initial_pred_line.set_data(ix, iy)
            self.initial_pred_line.set_alpha(0.48)

            # Map [33.0s, 42.2s] smoothly onto the precomputed optimization milestones
            opt_prog = smootherstep(window_progress(frame, 33.0, 42.2))
            max_idx = len(self.sim.traj_b_out) - 1
            exact_idx = opt_prog * max_idx
            idx_lo = int(math.floor(exact_idx))
            idx_hi = min(max_idx, idx_lo + 1)
            frac = exact_idx - idx_lo

            w_cur = (1.0 - frac) * self.sim.traj_w[idx_lo] + frac * self.sim.traj_w[idx_hi]
            b_cur = (1.0 - frac) * self.sim.traj_b[idx_lo] + frac * self.sim.traj_b[idx_hi]
            v_cur = (1.0 - frac) * self.sim.traj_v[idx_lo] + frac * self.sim.traj_v[idx_hi]
            b_out_cur = float(
                (1.0 - frac) * self.sim.traj_b_out[idx_lo] + frac * self.sim.traj_b_out[idx_hi]
            )

            _, VH_cur, y_hat_cur = forward_network(x_grid, w_cur, b_cur, v_cur, b_out_cur)

            # Update all m individual weighted neuron responses in real time
            for i, nline in enumerate(self.neuron_lines):
                nx, ny = self.world_to_screen(x_grid, VH_cur[i])
                nline.set_data(nx, ny)
                nline.set_alpha(0.36)

            # Update the combined prediction curve (transitions from initial slate to gold)
            cx, cy = self.world_to_screen(x_grid, y_hat_cur)
            self.combined_main_line.set_data(cx, cy)
            self.combined_main_line.set_color(COLOR_FINAL_CURVE if opt_prog > 0.05 else COLOR_INITIAL_PRED)
            self.combined_main_line.set_alpha(0.98)

            self.combined_glow_line.set_data(cx, cy)
            self.combined_glow_line.set_alpha(0.24 * opt_prog)
            return

        # =====================================================================
        # SCENE 6: 43–50s | FINAL REVEAL
        # =====================================================================
        self._set_data_points(1.0, alpha_scale=1.0)

        fade_inter = 1.0 - smoothstep(window_progress(frame, 43.0, 45.2))

        # Final optimized parameters
        w_fin = self.sim.traj_w[-1]
        b_fin = self.sim.traj_b[-1]
        v_fin = self.sim.traj_v[-1]
        b_out_fin = float(self.sim.traj_b_out[-1])
        _, VH_fin, y_hat_fin = forward_network(x_grid, w_fin, b_fin, v_fin, b_out_fin)

        # Smoothly fade out intermediate basis functions and initial prediction
        if fade_inter > 0.01:
            ix, iy = self.world_to_screen(x_grid, self.sim.traj_preds[0])
            self.initial_pred_line.set_data(ix, iy)
            self.initial_pred_line.set_alpha(0.48 * fade_inter)

            for i, nline in enumerate(self.neuron_lines):
                nx, ny = self.world_to_screen(x_grid, VH_fin[i])
                nline.set_data(nx, ny)
                nline.set_alpha(0.36 * fade_inter)

        # Subtle reference of the true underlying pattern y_target
        ghost_in = smoothstep(window_progress(frame, 44.5, 46.5))
        tx, ty = self.world_to_screen(x_grid, self.sim.y_target_grid)
        self.target_ghost_line.set_data(tx, ty)
        self.target_ghost_line.set_alpha(0.45 * ghost_in)

        # Final learned curve
        cx, cy = self.world_to_screen(x_grid, y_hat_fin)
        self.combined_main_line.set_data(cx, cy)
        self.combined_main_line.set_color(COLOR_FINAL_CURVE)
        self.combined_main_line.set_alpha(1.0)

        self.combined_glow_line.set_data(cx, cy)
        self.combined_glow_line.set_alpha(0.28)

        # Final compact unified mathematical model expression
        self.eq_primary.set_text(
            r"$\hat{y} = b_{\mathrm{out}} + \sum_{i=1}^{m} v_i \sigma(w_i x + b_i)$"
        )
        self.eq_primary.set_alpha(0.98)

    def render_rgba_bytes(self, frame: int) -> memoryview:
        self.update_frame(frame)
        self.canvas.draw()
        return self.canvas.buffer_rgba()


# =============================================================================
# 5. FFMPEG STREAMING & PARALLEL CHUNK PIPELINE
# =============================================================================

def resolve_ffmpeg_exe() -> str:
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg executable not found on PATH and imageio-ffmpeg is unavailable."
        ) from exc


def render_segment_worker(args: Tuple[int, int, int, str, str, MathSimulationData]) -> str:
    """
    Worker process that renders frames [start_frame, end_frame) and pipes raw
    RGBA bytes directly into an FFmpeg segment encoder.
    """
    start_frame, end_frame, dpi, output_segment_path, ffmpeg_exe, sim_data = args
    width = int(round(FIG_WIDTH_IN * dpi))
    height = int(round(FIG_HEIGHT_IN * dpi))

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-loglevel", "error",
        "-c:v", "libx264",
        "-preset", "fast",
        "-threads", "2",
        "-x264-params", "rc-lookahead=12:sync-lookahead=12",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_segment_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    animator = MathAnimator(sim=sim_data, dpi=dpi)
    try:
        assert proc.stdin is not None
        try:
            for f in range(start_frame, end_frame):
                buf = animator.render_rgba_bytes(f)
                proc.stdin.write(memoryview(buf))
                if (f - start_frame + 1) % 75 == 0:
                    print(f"      Segment [{start_frame}:{end_frame}] -> frame {f - start_frame + 1}/{end_frame - start_frame}", flush=True)
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(
                f"FFmpeg segment failed (exit {ret}): {stderr_bytes.decode('utf-8', errors='replace')}"
            )
    finally:
        plt.close(animator.fig)

    return output_segment_path


def verify_video_output(video_path: Path, ffmpeg_exe: str) -> Dict[str, str | int | float]:
    """
    Verifies the generated MP4 file's resolution, frame count, duration, and codec
    using ffprobe (if available) or ffmpeg -i stderr inspection.
    """
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Output file {video_path} is missing or empty.")

    info: Dict[str, str | int | float] = {
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
    }

    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        cmd = [
            ffprobe_exe,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
            "-of", "json",
            str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        stream = data["streams"][0]
        info["codec"] = stream.get("codec_name", "unknown")
        info["width"] = int(stream.get("width", 0))
        info["height"] = int(stream.get("height", 0))
        info["r_frame_rate"] = stream.get("r_frame_rate", "")
        info["nb_frames"] = int(stream.get("nb_read_frames", 0))
        info["duration_sec"] = float(stream.get("duration", 0.0))
    else:
        # Fallback verification via ffmpeg -i
        cmd = [ffmpeg_exe, "-i", str(video_path), "-f", "null", "-"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        info["ffmpeg_report"] = res.stderr.strip().splitlines()[-3:]

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the 50-second 4K vertical mathematical animation."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="final_short.mp4",
        help="Output MP4 filename (default: final_short.mp4)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render fast qHD preview (540x960 at 30 FPS) before full 4K render.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="Number of parallel rendering processes (default: up to 4).",
    )
    args = parser.parse_args()

    dpi = 75 if args.preview else 300
    width = int(round(FIG_WIDTH_IN * dpi))
    height = int(round(FIG_HEIGHT_IN * dpi))
    output_path = Path(args.output).resolve()

    t0 = time.perf_counter()
    print(f"[1/4] Precomputing data & optimizing 10-neuron sigmoid network (seed=42)...")
    sim_data = precompute_mathematics(seed=42, num_neurons=10)
    print(
        f"      Initial MSE: {sim_data.traj_losses[0]:.5f} -> "
        f"Final Optimized MSE: {sim_data.traj_losses[-1]:.5f}"
    )

    ffmpeg_exe = resolve_ffmpeg_exe()
    print(f"[2/4] Using FFmpeg binary: {ffmpeg_exe}")
    print(
        f"[3/4] Rendering {TOTAL_FRAMES} frames ({DURATION_SEC}s @ {FPS} FPS) "
        f"at {width}x{height} using {args.workers} worker(s)..."
    )

    if args.workers <= 1:
        render_segment_worker((0, TOTAL_FRAMES, dpi, str(output_path), ffmpeg_exe, sim_data))
    else:
        with tempfile.TemporaryDirectory(prefix="math_short_segments_") as tmpdir:
            num_workers = min(args.workers, TOTAL_FRAMES)
            chunk_size = TOTAL_FRAMES // num_workers
            tasks = []
            segment_paths: List[str] = []

            for i in range(num_workers):
                s_frame = i * chunk_size
                e_frame = TOTAL_FRAMES if i == num_workers - 1 else (i + 1) * chunk_size
                seg_path = os.path.join(tmpdir, f"seg_{i:03d}.mp4")
                segment_paths.append(seg_path)
                tasks.append((s_frame, e_frame, dpi, seg_path, ffmpeg_exe, sim_data))

            with mp.Pool(processes=num_workers) as pool:
                pool.map(render_segment_worker, tasks)

            concat_list_path = os.path.join(tmpdir, "segments.txt")
            with open(concat_list_path, "w", encoding="utf-8") as f_list:
                for seg_p in segment_paths:
                    escaped = seg_p.replace("'", "'\\''")
                    f_list.write(f"file '{escaped}'\n")

            concat_cmd = [
                ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]
            subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    elapsed = time.perf_counter() - t0
    print(f"[4/4] Render completed in {elapsed:.1f}s. Verifying output metadata...")
    meta = verify_video_output(output_path, ffmpeg_exe)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
