#!/usr/bin/env python3
"""Vortex Math 4.1: an annotated, educational spacetime laboratory.

This file deliberately reuses the tested Vortex Math 4.0 engine.  The new layer
adds teaching aids and visual explanations without mixing lesson text into the
field equations.

GLOSSARY
========
radius:        major size of a shape, measured in meters
thickness:     minor radius or characteristic width, measured in meters
density:       smooth dimensionless field strength
circulation:   Gamma, the strength of a vortex flow
magnetic:      B-field strength, measured in Tesla
pressure:      environmental pressure, measured in atmospheres
temperature:   environmental temperature, measured in Kelvin
dt:            base simulation time step, measured in seconds
domain:        the simulated spatial volume

SHAPE THEORY
============
Sphere, torus, cylinder, disk, shell, sheet, filament, and spheromak each
provide a density field, a velocity field, an optional magnetic field, and a
visual surface mesh.  The fields are analytic teaching models, not a complete
solution to the Navier-Stokes or magnetohydrodynamic equations.

ENGINE THEORY
=============
The SimulationEngine blends contributions from FlowObjects.  It intentionally
favors readable mathematics over conservation-law completeness.  The GUI only
asks an engine for fields, so a future CFD or neural backend can replace it.

SPACETIME THEORY
================
The laboratory has a spatial domain, a temporal domain, a grid resolution, and
a time step.  Time scale, time warp, time dilation, and visual spatial curvature
change the laboratory's coordinates or evolution; curvature is explicitly a
visual teaching effect, not general relativity.

ENVIRONMENTAL PHYSICS
=====================
Pressure and temperature scale the density model approximately as rho ~ P/T.
Magnetic pressure is computed from PB = B^2/(2 mu_0).  Container geometry and
boundary conditions determine which parts of the analytic field are visible.

COLLISION THEORY
================
When objects overlap, this engine adds their density, velocity, and magnetic
fields.  Merge, bounce, reaction, and shape transformation are extension points,
not claims that this lightweight model enforces physical conservation.

NEURAL SURROGATE THEORY
=======================
A future neural engine could learn field evolution from analytic simulations,
CFD output, or experimental data.  This version exposes the engine boundary and
keeps the visualization independent of that future implementation.

DESIGN PHILOSOPHY
=================
Readability over cleverness.  Explicit meanings over hidden state.  Every
visual effect is labeled when it is pedagogical rather than physically exact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


_BASE_PATH = Path(__file__).with_name("vortexMath4.0.py")
_BASE_SPEC = importlib.util.spec_from_file_location("vortex_math_40_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"Could not load base engine from {_BASE_PATH}")
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _BASE_MODULE
_BASE_SPEC.loader.exec_module(_BASE_MODULE)


EDUCATIONAL_MODES = ("Beginner", "Advanced", "Expert")
THEMES = ("Scientific", "Neon Plasma", "Wireframe", "Dark Mode")


class VortexMath41App(_BASE_MODULE.VortexMathApp):
    """Educational front-end that teaches the model while it runs."""

    def __init__(self) -> None:
        self.educational_mode = "Advanced"
        self.theme = "Scientific"
        self.show_field_lines = True
        self.show_curvature = True
        super().__init__()
        self._build_learning_controls()
        self._apply_theme(self.theme)

    def _build_learning_controls(self) -> None:
        """Add learner controls below the 3D view, outside the physics panel."""
        self.fig.text(0.04, 0.105, "Learning mode", fontsize=8, weight="bold")
        mode_axis = self.fig.add_axes([0.04, 0.015, 0.13, 0.08])
        self.mode_radio = _BASE_MODULE.RadioButtons(mode_axis, EDUCATIONAL_MODES, active=1)
        self.mode_radio.on_clicked(self._set_educational_mode)

        self.fig.text(0.20, 0.105, "Visual theme", fontsize=8, weight="bold")
        theme_axis = self.fig.add_axes([0.20, 0.015, 0.13, 0.08])
        self.theme_radio = _BASE_MODULE.RadioButtons(theme_axis, THEMES, active=0)
        self.theme_radio.on_clicked(self._apply_theme)

        self.lesson_text = self.fig.text(0.37, 0.04, "", fontsize=8, color="0.3", va="bottom")
        self._update_lesson_text()

    def _set_educational_mode(self, value: str) -> None:
        """Change how much mathematical context is shown to the learner."""
        self.educational_mode = value
        if value == "Beginner":
            self.grid_size = 24
        elif value == "Advanced":
            self.grid_size = 36
        else:
            self.grid_size = 48
        self._update_lesson_text()
        self.draw()

    def _apply_theme(self, value: str) -> None:
        """Apply a visual theme while leaving all physical values unchanged."""
        self.theme = value
        palettes = {
            "Scientific": ("#f4f1ea", "#ffffff", "#202020"),
            "Neon Plasma": ("#081525", "#081525", "#e8f8ff"),
            "Wireframe": ("#eef2f3", "#eef2f3", "#1c2830"),
            "Dark Mode": ("#111318", "#111318", "#f4f4f4"),
        }
        figure_color, axes_color, text_color = palettes[value]
        self.fig.patch.set_facecolor(figure_color)
        self.ax3d.set_facecolor(axes_color)
        for axis in self.fig.axes:
            axis.tick_params(colors=text_color, labelcolor=text_color)
        self.lesson_text.set_color(text_color)
        self._update_lesson_text()
        if hasattr(self, "fig"):
            self.fig.canvas.draw_idle()

    def _update_lesson_text(self) -> None:
        """Keep the short textbook annotation synchronized with the mode."""
        if not hasattr(self, "lesson_text"):
            return
        lessons = {
            "Beginner": "Beginner: density follows the simple teaching rule rho ~ P/T. Use Step to watch one small change.",
            "Advanced": "Advanced: magnetic pressure is PB = B^2/(2 mu_0); velocity and B-fields are analytic field overlays.",
            "Expert": "Expert: this is a field-composition laboratory, not a conservation-law solver or a general-relativistic metric.",
        }
        self.lesson_text.set_text(lessons[self.educational_mode])

    def _magnetic_vector(self, point: np.ndarray) -> np.ndarray:
        """Sample the combined magnetic field at one point for a field-line step."""
        coordinates = tuple(np.array([value], dtype=float) for value in point)
        result = np.zeros(3, dtype=float)
        for flow_object in self.engine.objects:
            contribution = flow_object.magnetic_field(*coordinates)
            result += np.array([component[0] for component in contribution])
        return result * self.engine.environment.magnetic_field_T

    def _draw_magnetic_field_lines(self) -> None:
        """Trace short streamlines so learners can see the magnetic geometry."""
        if not self.show_field_lines or self.engine.environment.magnetic_field_T <= 0.0:
            return
        radius = 2.2
        for angle in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
            seed = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            points = [seed.copy()]
            point = seed.copy()
            for _ in range(32):
                vector = self._magnetic_vector(point)
                magnitude = np.linalg.norm(vector)
                if magnitude < 1e-6:
                    break
                point = point + 0.10 * vector / magnitude
                if np.any(np.abs(point) > self.domain_half_width):
                    break
                points.append(point.copy())
            if len(points) > 2:
                line = np.asarray(points)
                color = "#65d9ff" if self.theme != "Dark Mode" else "#b7f3ff"
                self.ax3d.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=0.8, alpha=0.55)

    def _draw_curvature_overlay(self) -> None:
        """Draw labeled concentric rings for the visual-only curvature parameter."""
        curvature = self.engine.spacetime.spatial_curvature
        if not self.show_curvature or curvature <= 0.0:
            return
        angle = np.linspace(0.0, 2.0 * np.pi, 100)
        color = "#ffcc66" if self.theme != "Dark Mode" else "#ffe6a3"
        for z_level in (-2.0, 0.0, 2.0):
            radius = 1.0 + curvature * (1.0 + abs(z_level) / 2.0)
            self.ax3d.plot(radius * np.cos(angle), radius * np.sin(angle), z_level + 0.0 * angle, color=color, linewidth=0.6, alpha=0.35)

    def draw(self) -> None:
        """Render the 4.0 scientific view plus explanatory visual overlays."""
        super().draw()
        self._draw_magnetic_field_lines()
        self._draw_curvature_overlay()
        if self.theme == "Wireframe":
            for collection in self.ax3d.collections:
                collection.set_alpha(0.10)
        self.fig.canvas.draw_idle()


def main() -> None:
    """Launch the annotated Vortex Math 4.1 laboratory."""
    VortexMath41App().run()


if __name__ == "__main__":
    main()
