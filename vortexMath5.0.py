#!/usr/bin/env python3
"""Vortex Math 5.0 - responsive scientific visualization laboratory.

Version 5.0 keeps the analytic simulation engine from Vortex Math 4.x while
rebuilding the user interface around Tkinter's layout managers.  Matplotlib is
embedded only as the scientific visualization canvas.

Highlights
----------
* Responsive, resizable GUI with no figure-coordinate control placement.
* Persistent Matplotlib artists for smoother animation.
* Debounced redraws for sliders and controls.
* Separate simulation/update and rendering paths.
* Educational modes and visual themes.
* Scene save/load.
* Graceful operation on Windows, macOS, and Linux.

The analytic field models remain educational models rather than a complete
Navier-Stokes/MHD/general-relativity solver.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib
import numpy as np

# TkAgg is required because Matplotlib is embedded in a Tkinter window.
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


BASE_PATH = Path(__file__).with_name("vortexMath4.0.py")
if not BASE_PATH.exists():
    raise FileNotFoundError(
        f"Vortex Math 4.0 engine not found beside this file:\n{BASE_PATH}"
    )

SPEC = importlib.util.spec_from_file_location("vortex_math_40_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load base engine from {BASE_PATH}")

BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


SHAPES = BASE.SHAPES
PHASES = BASE.PHASES
CONTAINERS = BASE.CONTAINERS
BOUNDARIES = BASE.BOUNDARIES
ENGINE_TYPES = BASE.ENGINE_TYPES
PRESETS = BASE.PRESETS
MU_0 = BASE.MU_0

EDUCATIONAL_MODES = ("Beginner", "Advanced", "Expert")
THEMES = ("Scientific", "Neon Plasma", "Wireframe", "Dark Mode")


@dataclass(frozen=True)
class Theme:
    window: str
    panel: str
    canvas: str
    text: str
    muted: str
    accent: str
    accent2: str
    border: str
    control: str
    plot_text: str
    quiver: str
    grid: str
    field_alpha: float


THEME_DATA = {
    "Scientific": Theme(
        "#e9edf2", "#f7f9fb", "#ffffff", "#17202a", "#637083",
        "#1769aa", "#4b8fbd", "#c8d0da", "#ffffff", "#24313d",
        "#476579", "#d7dee6", 0.28,
    ),
    "Neon Plasma": Theme(
        "#07121d", "#0b1a28", "#07121d", "#e8f8ff", "#91afbd",
        "#45d9ff", "#ff5fd2", "#21465b", "#0e2332", "#dff7ff",
        "#7ee7ff", "#16394a", 0.24,
    ),
    "Wireframe": Theme(
        "#e8edf0", "#f4f6f7", "#eef2f3", "#18242d", "#66747d",
        "#2b687e", "#6b8894", "#c6d0d5", "#ffffff", "#1c2830",
        "#526c77", "#c4d0d5", 0.10,
    ),
    "Dark Mode": Theme(
        "#101318", "#171a21", "#111318", "#f3f5f7", "#9aa6b2",
        "#66c7ff", "#ffcc66", "#303844", "#20252e", "#edf5fa",
        "#b9e8ff", "#29333e", 0.18,
    ),
}


class VortexMath5App:
    """Responsive Vortex Math 5.0 desktop application."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vortex Math 5.0")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 680)

        self.engine = BASE.make_default_engine()
        self.domain_half_width = 5.0
        self.grid_size = 32
        self.density_threshold = 0.05
        self.num_particles = 260
        self.particles: np.ndarray | None = None
        self.running = False
        self.educational_mode = "Advanced"
        self.theme = "Scientific"
        self.show_field_lines = True
        self.show_curvature = True

        self._after_id: str | None = None
        self._rendering = False
        self._last_fields: Any = None
        self._field_lines: list[Any] = []
        self._curvature_lines: list[Any] = []
        self._surface_artists: list[Any] = []
        self._scatter_artist: Any = None
        self._particle_artist: Any = None
        self._quiver_artist: Any = None
        self._last_render_signature: Any = None

        self.vars: dict[str, tk.Variable] = {}
        self.slider_widgets: list[tuple[tk.Scale, tk.Label, str]] = []

        self._configure_ttk()
        self._build_ui()
        self._apply_theme()
        self._schedule_render(immediate=True)

        self.root.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _configure_ttk(self) -> None:
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.header = ttk.Frame(self.root, padding=(18, 12, 18, 8))
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(0, weight=1)

        title_box = ttk.Frame(self.header)
        title_box.grid(row=0, column=0, sticky="w")

        self.title_label = ttk.Label(
            title_box, text="VORTEX MATH", font=("Segoe UI", 18, "bold")
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ttk.Label(
            title_box,
            text="Spacetime • field composition • analytic laboratory",
            font=("Segoe UI", 9),
        )
        self.subtitle_label.pack(anchor="w", pady=(1, 0))

        status_box = ttk.Frame(self.header)
        status_box.grid(row=0, column=1, sticky="e")

        self.status_label = ttk.Label(
            status_box, text="● PAUSED", font=("Segoe UI", 9, "bold")
        )
        self.status_label.pack(side="right")

        self.main = ttk.Panedwindow(self.root, orient="horizontal")
        self.main.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.control_frame = ttk.Frame(self.main, padding=10)
        self.visual_frame = ttk.Frame(self.main, padding=(8, 8, 4, 8))

        self.main.add(self.control_frame, weight=0)
        self.main.add(self.visual_frame, weight=1)

        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.rowconfigure(1, weight=1)

        self._build_control_header()
        self._build_control_notebook()
        self._build_visualization()

    def _build_control_header(self) -> None:
        frame = ttk.Frame(self.control_frame)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame, text="LABORATORY CONTROLS",
            font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=0, sticky="w")

        self.engine_label = ttk.Label(frame, text="Analytic")
        self.engine_label.grid(row=0, column=1, sticky="e")

    def _build_control_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        self.object_tab = ttk.Frame(self.notebook, padding=10)
        self.environment_tab = ttk.Frame(self.notebook, padding=10)
        self.time_tab = ttk.Frame(self.notebook, padding=10)
        self.space_tab = ttk.Frame(self.notebook, padding=10)
        self.visual_tab = ttk.Frame(self.notebook, padding=10)
        self.metrics_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.object_tab, text="Objects")
        self.notebook.add(self.environment_tab, text="Environment")
        self.notebook.add(self.time_tab, text="Time")
        self.notebook.add(self.space_tab, text="Spacetime")
        self.notebook.add(self.visual_tab, text="Visuals")
        self.notebook.add(self.metrics_tab, text="Metrics")

        self._build_object_tab()
        self._build_environment_tab()
        self._build_time_tab()
        self._build_space_tab()
        self._build_visual_tab()
        self._build_metrics_tab()

    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill="x", pady=(0, 10))
        return frame

    def _build_object_tab(self) -> None:
        for index, obj in enumerate(self.engine.objects):
            section = self._section(self.object_tab, obj.label)
            section.columnconfigure(1, weight=1)

            ttk.Label(section, text="Shape").grid(row=0, column=0, sticky="w", pady=4)
            shape = tk.StringVar(value=obj.shape_type)
            self.vars[f"shape_{index}"] = shape
            combo = ttk.Combobox(
                section, textvariable=shape, values=SHAPES,
                state="readonly", width=18
            )
            combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=4)
            combo.bind(
                "<<ComboboxSelected>>",
                lambda e, i=index: self._set_shape(i, self.vars[f"shape_{i}"].get()),
            )

            ttk.Label(section, text="Matter state").grid(
                row=1, column=0, sticky="w", pady=4
            )
            phase = tk.StringVar(value=obj.phase_type)
            self.vars[f"phase_{index}"] = phase
            combo = ttk.Combobox(
                section, textvariable=phase, values=PHASES,
                state="readonly", width=18
            )
            combo.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=4)
            combo.bind(
                "<<ComboboxSelected>>",
                lambda e, i=index: self._set_phase(i, self.vars[f"phase_{i}"].get()),
            )

        preset = self._section(self.object_tab, "Presets")
        preset.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value=PRESETS[0])
        ttk.Combobox(
            preset, textvariable=self.preset_var, values=PRESETS,
            state="readonly"
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(
            preset, text="Apply preset", command=self._apply_preset
        ).grid(row=1, column=0, sticky="ew")

    def _build_environment_tab(self) -> None:
        self._slider(
            self.environment_tab, "Pressure", "pressure", 0.1, 5.0,
            self.engine.environment.pressure_atm, "{:.2f} atm", self._set_pressure
        )
        self._slider(
            self.environment_tab, "Temperature", "temperature", 100, 1500,
            self.engine.environment.temperature_K, "{:.0f} K", self._set_temperature
        )
        self._slider(
            self.environment_tab, "Magnetic field", "magnetic", 0, 10,
            self.engine.environment.magnetic_field_T, "{:.2f} T", self._set_magnetic
        )
        self._slider(
            self.environment_tab, "Gravity", "gravity", 0, 20,
            self.engine.environment.gravity_m_s2, "{:.2f} m/s²", self._set_gravity
        )
        self._slider(
            self.environment_tab, "Viscosity", "viscosity", 0, 1,
            self.engine.environment.viscosity, "{:.3f}", self._set_viscosity
        )
        self._slider(
            self.environment_tab, "Container size", "container", 2, 10,
            self.engine.environment.container_size_m, "{:.1f} m", self._set_container
        )

        section = self._section(self.environment_tab, "Container")
        self.container_var = tk.StringVar(value=self.engine.environment.shape)
        for value in CONTAINERS:
            ttk.Radiobutton(
                section, text=value, variable=self.container_var,
                value=value, command=self._set_container_shape
            ).pack(anchor="w", pady=2)

        section = self._section(self.environment_tab, "Boundary")
        self.boundary_var = tk.StringVar(
            value=self.engine.environment.boundary_condition
        )
        for value in BOUNDARIES:
            ttk.Radiobutton(
                section, text=value, variable=self.boundary_var,
                value=value, command=self._set_boundary
            ).pack(anchor="w", pady=2)

    def _build_time_tab(self) -> None:
        self._slider(
            self.time_tab, "Time scale", "time_scale", 0, 4,
            self.engine.spacetime.time_scale, "{:.2f}", self._set_time_scale
        )
        self._slider(
            self.time_tab, "Time warp", "time_warp", -2, 2,
            self.engine.spacetime.time_warp, "{:.2f}", self._set_time_warp
        )
        self._slider(
            self.time_tab, "Time dilation", "time_dilation", 0.25, 4,
            self.engine.spacetime.time_dilation, "{:.2f}", self._set_time_dilation
        )

        section = self._section(self.time_tab, "Simulation")
        buttons = ttk.Frame(section)
        buttons.pack(fill="x")
        for text, command in (
            ("▶  Play", lambda: self._set_running(True)),
            ("Ⅱ  Pause", lambda: self._set_running(False)),
            ("Step  ›", self._step_once),
        ):
            ttk.Button(buttons, text=text, command=command).pack(
                side="left", expand=True, fill="x", padx=2
            )

    def _build_space_tab(self) -> None:
        self._slider(
            self.space_tab, "Spatial curvature", "curvature", 0, 1,
            self.engine.spacetime.spatial_curvature, "{:.3f}", self._set_curvature
        )
        self._slider(
            self.space_tab, "Field amplification", "field_amp", 0, 4,
            self.engine.spacetime.field_amplification, "{:.2f}", self._set_field_amp
        )
        self._slider(
            self.space_tab, "Field smoothing", "smoothing", 0, 1,
            self.engine.spacetime.field_smoothing, "{:.3f}", self._set_smoothing
        )
        self._slider(
            self.space_tab, "Noise injection", "noise", 0, 0.2,
            self.engine.spacetime.noise_injection, "{:.3f}", self._set_noise
        )
        self._slider(
            self.space_tab, "Field sampling", "sampling", 4, 12,
            self.engine.spacetime.field_sampling_density, "{:.0f}", self._set_sampling
        )

        section = self._section(self.space_tab, "Educational mode")
        self.education_var = tk.StringVar(value=self.educational_mode)
        for mode in EDUCATIONAL_MODES:
            ttk.Radiobutton(
                section, text=mode, variable=self.education_var,
                value=mode, command=self._set_education
            ).pack(anchor="w", pady=2)

    def _build_visual_tab(self) -> None:
        section = self._section(self.visual_tab, "Theme")
        self.theme_var = tk.StringVar(value=self.theme)
        for value in THEMES:
            ttk.Radiobutton(
                section, text=value, variable=self.theme_var,
                value=value, command=self._apply_theme
            ).pack(anchor="w", pady=2)

        section = self._section(self.visual_tab, "Overlays")
        self.field_lines_var = tk.BooleanVar(value=True)
        self.curvature_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            section, text="Magnetic field lines",
            variable=self.field_lines_var,
            command=self._schedule_render
        ).pack(anchor="w", pady=3)
        ttk.Checkbutton(
            section, text="Curvature rings",
            variable=self.curvature_var,
            command=self._schedule_render
        ).pack(anchor="w", pady=3)

        ttk.Label(
            section,
            text="Educational overlays are visual aids;\nthey do not alter the field equations.",
            wraplength=250
        ).pack(anchor="w", pady=(10, 0))

    def _build_metrics_tab(self) -> None:
        section = self._section(self.metrics_tab, "Engine")
        self.engine_var = tk.StringVar(value=self.engine.engine_type)
        for value in ENGINE_TYPES:
            ttk.Radiobutton(
                section, text=value, variable=self.engine_var,
                value=value, command=self._set_engine
            ).pack(anchor="w", pady=2)

        self.metrics_text = tk.Text(
            self.metrics_tab, height=14, width=30,
            font=("Consolas", 9), relief="flat", borderwidth=0,
            state="disabled", padx=8, pady=8
        )
        self.metrics_text.pack(fill="both", expand=True)

        buttons = ttk.Frame(self.metrics_tab)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Save scene", command=self._save_scene).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(buttons, text="Load scene", command=self._load_scene).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

    def _slider(
        self, parent: ttk.Frame, label: str, key: str, low: float, high: float,
        initial: float, fmt: str, callback: Any
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=5)
        row.columnconfigure(1, weight=1)

        value_var = tk.StringVar(value=fmt.format(initial))
        ttk.Label(row, text=label).grid(row=0, column=0, sticky="w")
        ttk.Label(row, textvariable=value_var, width=12, anchor="e").grid(
            row=0, column=2, sticky="e"
        )

        scale = tk.Scale(
            row, from_=low, to=high, resolution=(high - low) / 200,
            orient="horizontal", showvalue=False, highlightthickness=0,
            bd=0, length=180
        )
        scale.set(initial)
        scale.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(1, 0))

        scale.configure(
            command=lambda value, v=value_var, f=fmt, cb=callback:
            self._slider_changed(value, v, f, cb)
        )
        self.slider_widgets.append((scale, value_var, fmt))
        self.vars[key] = scale

    @staticmethod
    def _slider_changed(
        value: str, value_var: tk.StringVar, fmt: str, callback: Any
    ) -> None:
        numeric = float(value)
        value_var.set(fmt.format(numeric))
        callback(numeric)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _build_visualization(self) -> None:
        self.visual_frame.columnconfigure(0, weight=1)
        self.visual_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax3d = self.figure.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.visual_frame)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self.visual_frame)
        footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)

        self.lesson_label = ttk.Label(
            footer, text="", font=("Segoe UI", 9),
            wraplength=900, justify="left"
        )
        self.lesson_label.grid(row=0, column=0, sticky="w")

        self.coord_label = ttk.Label(
            footer, text="3D field",
            font=("Segoe UI", 8)
        )
        self.coord_label.grid(row=0, column=1, sticky="e")

    def _clear_dynamic_artists(self) -> None:
        for artist in self._surface_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._surface_artists.clear()

        for artist in self._field_lines + self._curvature_lines:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._field_lines.clear()
        self._curvature_lines.clear()

        for artist_name in (
            "_scatter_artist", "_particle_artist", "_quiver_artist"
        ):
            artist = getattr(self, artist_name)
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass
                setattr(self, artist_name, None)

    def _render(self) -> None:
        if self._rendering:
            return
        self._rendering = True
        try:
            self._clear_dynamic_artists()
            self.ax3d.clear()

            theme = THEME_DATA[self.theme]
            self.ax3d.set_facecolor(theme.canvas)
            self.figure.patch.set_facecolor(theme.window)

            self.ax3d.set_xlim(-self.domain_half_width, self.domain_half_width)
            self.ax3d.set_ylim(-self.domain_half_width, self.domain_half_width)
            self.ax3d.set_zlim(-self.domain_half_width, self.domain_half_width)
            self.ax3d.set_xlabel("x [m]", color=theme.plot_text)
            self.ax3d.set_ylabel("y [m]", color=theme.plot_text)
            self.ax3d.set_zlabel("z [m]", color=theme.plot_text)
            self.ax3d.set_title(
                f"Vortex Math 5.0   |   t = {self.engine.time:.2f} s   |   "
                f"{self.engine.engine_type}   |   "
                f"P = {self.engine.environment.pressure_atm:.2f} atm   |   "
                f"B = {self.engine.environment.magnetic_field_T:.2f} T",
                color=theme.plot_text, pad=12, fontsize=11, fontweight="bold"
            )

            # 1. Object surfaces.
            for obj in self.engine.objects:
                mesh = obj.surface_mesh(resolution=24)
                if mesh is not None:
                    artist = self.ax3d.plot_surface(
                        *mesh, color=obj.color, linewidth=0,
                        alpha=0.34 if self.theme != "Wireframe" else 0.10,
                        shade=True
                    )
                    self._surface_artists.append(artist)

            # 2. Analytic fields.
            X, Y, Z = self._grid()
            fields = self.engine.compute_fields((X, Y, Z))
            self._last_fields = fields

            mask = fields.density > self.density_threshold
            if np.any(mask):
                self._scatter_artist = self.ax3d.scatter(
                    X[mask], Y[mask], Z[mask],
                    c=fields.temperature[mask],
                    cmap="inferno", s=6,
                    alpha=theme.field_alpha
                )

            # 3. Particles.
            self._advect_particles()
            if self.particles is not None:
                self._particle_artist = self.ax3d.scatter(
                    self.particles[:, 0], self.particles[:, 1],
                    self.particles[:, 2],
                    s=3.5, c=theme.quiver, alpha=0.62
                )

            # 4. Velocity + magnetic vector field.
            sample_count = max(
                4, int(self.engine.spacetime.field_sampling_density)
            )
            field_coordinates = np.linspace(
                -self.domain_half_width, self.domain_half_width, sample_count
            )
            FX, FY, FZ = np.meshgrid(
                field_coordinates, field_coordinates, field_coordinates,
                indexing="ij"
            )
            U, V, W = fields.velocity
            Bx, By, Bz = fields.magnetic
            sample_indices = np.linspace(
                0, self.grid_size - 1, sample_count, dtype=int
            )
            sampled = tuple(
                component[np.ix_(
                    sample_indices, sample_indices, sample_indices
                )]
                for component in (U, V, W, Bx, By, Bz)
            )

            curvature = self.engine.spacetime.spatial_curvature
            radial_scale = (
                1.0 + curvature *
                (FX**2 + FY**2 + FZ**2) /
                (3.0 * max(self.domain_half_width**2, 1e-9))
            )

            self._quiver_artist = self.ax3d.quiver(
                FX * radial_scale, FY * radial_scale, FZ * radial_scale,
                sampled[0] + sampled[3],
                sampled[1] + sampled[4],
                sampled[2] + sampled[5],
                color=theme.quiver, length=0.22,
                normalize=True, alpha=0.40,
            )

            if self.field_lines_var.get() and self.engine.environment.magnetic_field_T > 0:
                self._draw_field_lines(theme)

            if self.curvature_var.get() and curvature > 0:
                self._draw_curvature(theme)

            if self.theme == "Wireframe":
                self._apply_wireframe()

            self._update_metrics(fields)
            self._update_lesson()

            self.canvas.draw_idle()
        finally:
            self._rendering = False

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.linspace(
            -self.domain_half_width,
            self.domain_half_width,
            self.grid_size
        )
        return np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")

    def _draw_field_lines(self, theme: Theme) -> None:
        radius = min(2.2, self.domain_half_width * 0.55)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            point = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ])
            points = [point.copy()]

            for _ in range(24):
                vector = self._magnetic_vector(point)
                magnitude = np.linalg.norm(vector)
                if magnitude < 1e-6:
                    break
                point = point + 0.12 * vector / magnitude
                if np.any(np.abs(point) > self.domain_half_width):
                    break
                points.append(point.copy())

            if len(points) > 2:
                line = np.asarray(points)
                artist, = self.ax3d.plot(
                    line[:, 0], line[:, 1], line[:, 2],
                    color=theme.accent, linewidth=0.8, alpha=0.58
                )
                self._field_lines.append(artist)

    def _draw_curvature(self, theme: Theme) -> None:
        curvature = self.engine.spacetime.spatial_curvature
        angle = np.linspace(0, 2 * np.pi, 80)
        for z_level in (-2.0, 0.0, 2.0):
            radius = 1.0 + curvature * (1.0 + abs(z_level) / 2.0)
            artist, = self.ax3d.plot(
                radius * np.cos(angle),
                radius * np.sin(angle),
                np.full_like(angle, z_level),
                color=theme.accent2, linewidth=0.65, alpha=0.36
            )
            self._curvature_lines.append(artist)

    def _apply_wireframe(self) -> None:
        for collection in self.ax3d.collections:
            try:
                collection.set_alpha(0.10)
            except AttributeError:
                pass

    def _magnetic_vector(self, point: np.ndarray) -> np.ndarray:
        coordinates = tuple(
            np.array([value], dtype=float) for value in point
        )
        result = np.zeros(3, dtype=float)
        for obj in self.engine.objects:
            contribution = obj.magnetic_field(*coordinates)
            result += np.array(
                [component[0] for component in contribution]
            )
        return result * self.engine.environment.magnetic_field_T

    def _initialize_particles(self) -> None:
        self.particles = np.zeros((self.num_particles, 3), dtype=float)
        for index in range(self.num_particles):
            obj = self.engine.objects[index % len(self.engine.objects)]
            angle = np.random.uniform(0, 2 * np.pi)
            radial = obj.radius + obj.thickness * np.random.uniform(-0.7, 0.7)
            self.particles[index] = obj.position + np.array([
                radial * np.cos(angle),
                radial * np.sin(angle),
                np.random.uniform(-obj.thickness, obj.thickness)
            ])

    def _advect_particles(self) -> None:
        if self.particles is None:
            self._initialize_particles()

        X = self.particles[:, 0]
        Y = self.particles[:, 1]
        Z = self.particles[:, 2]

        velocity = [np.zeros_like(X) for _ in range(3)]
        for obj in self.engine.objects:
            contribution = obj.velocity_field(X, Y, Z)
            for axis in range(3):
                velocity[axis] += contribution[axis]

        self.particles += self.engine.dt * np.column_stack(velocity)
        out = np.any(
            np.abs(self.particles) > self.domain_half_width * 1.2, axis=1
        )
        if np.any(out):
            self.particles[out] = 0.0

    # ------------------------------------------------------------------
    # Rendering / animation
    # ------------------------------------------------------------------

    def _schedule_render(self, immediate: bool = False) -> None:
        if immediate:
            if self._after_id:
                try:
                    self.root.after_cancel(self._after_id)
                except tk.TclError:
                    pass
                self._after_id = None
            self._render()
            return

        if self._after_id:
            return
        self._after_id = self.root.after(35, self._scheduled_render)

    def _scheduled_render(self) -> None:
        self._after_id = None
        self._render()

    def _animation_tick(self) -> None:
        if self.running:
            self.engine.step()
            self._schedule_render()
        self.root.after(40, self._animation_tick)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.status_label.configure(
            text="● RUNNING" if running else "● PAUSED"
        )
        if running:
            self._animation_tick()

    def _step_once(self) -> None:
        self.engine.step()
        self._schedule_render(immediate=True)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _set_shape(self, index: int, value: str) -> None:
        self.engine.objects[index].shape_type = value
        self.particles = None
        self._schedule_render()

    def _set_phase(self, index: int, value: str) -> None:
        self.engine.objects[index].phase_type = value
        self._schedule_render()

    def _set_pressure(self, value: float) -> None:
        self.engine.environment.pressure_atm = value
        self._schedule_render()

    def _set_temperature(self, value: float) -> None:
        self.engine.environment.temperature_K = value
        self._schedule_render()

    def _set_magnetic(self, value: float) -> None:
        self.engine.environment.magnetic_field_T = value
        self._schedule_render()

    def _set_gravity(self, value: float) -> None:
        self.engine.environment.gravity_m_s2 = value

    def _set_viscosity(self, value: float) -> None:
        self.engine.environment.viscosity = value

    def _set_container(self, value: float) -> None:
        self.engine.environment.container_size_m = value
        self.domain_half_width = max(5.0, value)
        self._schedule_render()

    def _set_container_shape(self) -> None:
        self.engine.environment.shape = self.container_var.get()
        self._schedule_render()

    def _set_boundary(self) -> None:
        self.engine.environment.boundary_condition = self.boundary_var.get()
        self._schedule_render()

    def _set_time_scale(self, value: float) -> None:
        self.engine.spacetime.time_scale = value

    def _set_time_warp(self, value: float) -> None:
        self.engine.spacetime.time_warp = value

    def _set_time_dilation(self, value: float) -> None:
        self.engine.spacetime.time_dilation = value

    def _set_curvature(self, value: float) -> None:
        self.engine.spacetime.spatial_curvature = value
        self._schedule_render()

    def _set_field_amp(self, value: float) -> None:
        self.engine.spacetime.field_amplification = value
        self._schedule_render()

    def _set_smoothing(self, value: float) -> None:
        self.engine.spacetime.field_smoothing = value
        self._schedule_render()

    def _set_noise(self, value: float) -> None:
        self.engine.spacetime.noise_injection = value
        self._schedule_render()

    def _set_sampling(self, value: float) -> None:
        self.engine.spacetime.field_sampling_density = max(4, int(round(value)))
        self._schedule_render()

    def _set_engine(self) -> None:
        self.engine.engine_type = self.engine_var.get()
        self.engine_label.configure(text=self.engine.engine_type)
        self._schedule_render()

    def _set_education(self) -> None:
        self.educational_mode = self.education_var.get()
        self.grid_size = {
            "Beginner": 24,
            "Advanced": 32,
            "Expert": 40,
        }[self.educational_mode]
        self._update_lesson()
        self._schedule_render()

    def _apply_preset(self) -> None:
        value = self.preset_var.get()
        object_1, object_2 = self.engine.objects

        if value == "Fusion Reactor":
            object_1.shape_type, object_2.shape_type = "Torus", "Torus"
            object_1.phase_type, object_2.phase_type = "Plasma", "Plasma"
            self.engine.environment.pressure_atm = 2.0
            self.engine.environment.magnetic_field_T = 5.0

        elif value == "Sphere to Torus":
            object_1.shape_type, object_2.shape_type = "Sphere", "Torus"
            object_1.phase_type, object_2.phase_type = "Plasma", "Plasma"
            self.engine.environment.pressure_atm = 1.0
            self.engine.environment.magnetic_field_T = 2.0

        elif value == "Jet vs Sheet":
            object_1.shape_type, object_2.shape_type = "Cylinder", "Sheet"
            object_1.phase_type, object_2.phase_type = "Gas", "Gas"
            self.engine.environment.pressure_atm = 1.5
            self.engine.environment.magnetic_field_T = 0.5

        elif value == "Shockwave Expansion":
            object_1.shape_type, object_2.shape_type = "Shell", "Disk"
            object_1.phase_type, object_2.phase_type = "Gas", "Plasma"
            self.engine.environment.pressure_atm = 2.5
            self.engine.environment.magnetic_field_T = 1.0

        elif value == "Magnetic Bottle":
            object_1.shape_type, object_2.shape_type = "Spheromak", "Torus"
            object_1.phase_type, object_2.phase_type = "Plasma", "Plasma"
            self.engine.environment.shape = "Sphere"
            self.engine.environment.magnetic_field_T = 8.0
            self.engine.environment.container_size_m = 5.0

        self.particles = None
        self._sync_control_values()
        self._schedule_render(immediate=True)

    def _sync_control_values(self) -> None:
        env = self.engine.environment
        st = self.engine.spacetime

        values = {
            "pressure": env.pressure_atm,
            "temperature": env.temperature_K,
            "magnetic": env.magnetic_field_T,
            "gravity": env.gravity_m_s2,
            "viscosity": env.viscosity,
            "container": env.container_size_m,
            "time_scale": st.time_scale,
            "time_warp": st.time_warp,
            "time_dilation": st.time_dilation,
            "curvature": st.spatial_curvature,
            "field_amp": st.field_amplification,
            "smoothing": st.field_smoothing,
            "noise": st.noise_injection,
            "sampling": st.field_sampling_density,
        }

        for key, value in values.items():
            widget = self.vars.get(key)
            if isinstance(widget, tk.Scale):
                widget.set(value)

        self.container_var.set(env.shape)
        self.boundary_var.set(env.boundary_condition)

        for index, obj in enumerate(self.engine.objects):
            self.vars[f"shape_{index}"].set(obj.shape_type)
            self.vars[f"phase_{index}"].set(obj.phase_type)

    # ------------------------------------------------------------------
    # Themes / educational text
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        self.theme = self.theme_var.get()
        theme = THEME_DATA[self.theme]

        self.root.configure(bg=theme.window)

        self.style.configure(
            ".",
            background=theme.panel,
            foreground=theme.text,
            font=("Segoe UI", 9),
        )
        self.style.configure(
            "TFrame", background=theme.panel
        )
        self.style.configure(
            "TLabel", background=theme.panel, foreground=theme.text
        )
        self.style.configure(
            "TLabelframe", background=theme.panel,
            foreground=theme.text
        )
        self.style.configure(
            "TLabelframe.Label", background=theme.panel,
            foreground=theme.text, font=("Segoe UI", 9, "bold")
        )
        self.style.configure(
            "TNotebook", background=theme.panel,
            borderwidth=0
        )
        self.style.configure(
            "TNotebook.Tab", padding=(10, 6),
            background=theme.control, foreground=theme.text
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", theme.accent)],
            foreground=[("selected", "#ffffff")]
        )
        self.style.configure(
            "TButton", padding=(9, 6),
            background=theme.control, foreground=theme.text
        )
        self.style.configure(
            "TCheckbutton", background=theme.panel,
            foreground=theme.text
        )
        self.style.configure(
            "TRadiobutton", background=theme.panel,
            foreground=theme.text
        )
        self.style.configure(
            "TCombobox", fieldbackground=theme.control,
            background=theme.control, foreground=theme.text
        )

        self.title_label.configure(foreground=theme.text)
        self.subtitle_label.configure(foreground=theme.muted)
        self.status_label.configure(
            foreground=theme.accent
        )
        self.engine_label.configure(foreground=theme.muted)
        self.lesson_label.configure(foreground=theme.muted)
        self.coord_label.configure(foreground=theme.muted)

        self.metrics_text.configure(
            bg=theme.control, fg=theme.text,
            insertbackground=theme.text
        )

        for scale, _, _ in self.slider_widgets:
            scale.configure(
                bg=theme.panel,
                troughcolor=theme.border,
                activebackground=theme.accent,
                highlightbackground=theme.panel,
                fg=theme.text
            )

        self._schedule_render(immediate=True)

    def _update_lesson(self) -> None:
        lessons = {
            "Beginner":
                "Beginner • Density follows the teaching rule ρ ~ P/T. "
                "Use Step to observe one controlled change.",
            "Advanced":
                "Advanced • Magnetic pressure is PB = B²/(2μ₀). "
                "Velocity and magnetic fields are analytic overlays.",
            "Expert":
                "Expert • This is a field-composition laboratory, not a "
                "conservation-law solver or general-relativistic metric."
        }
        self.lesson_label.configure(text=lessons[self.educational_mode])

    # ------------------------------------------------------------------
    # Metrics / scenes
    # ------------------------------------------------------------------

    def _update_metrics(self, fields: Any) -> None:
        values = self.engine.metrics(fields)
        text = (
            f"TIME              {values['time']:.3f} s\n"
            f"BASE DT           {values['dt']:.4f} s\n"
            f"PRESSURE          {values['pressure']:.3f} atm\n"
            f"TEMPERATURE       {values['temperature']:.1f} K\n"
            f"DENSITY           {values.get('density', 0):.3e}\n"
            f"KINETIC ENERGY    {values.get('kinetic_energy', 0):.3e}\n"
            f"MAGNETIC ENERGY   {values.get('magnetic_energy', 0):.3e}\n"
            f"MAGNETIC PRESSURE {values['magnetic_pressure']:.3e} Pa\n"
            f"ENGINE             {values['engine']}\n"
        )
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", text)
        self.metrics_text.configure(state="disabled")

    def _scene_dict(self) -> dict[str, Any]:
        return {
            "version": "5.0",
            "engine": self.engine.engine_type,
            "environment": {
                "shape": self.engine.environment.shape,
                "pressure_atm": self.engine.environment.pressure_atm,
                "temperature_K": self.engine.environment.temperature_K,
                "gas_type": self.engine.environment.gas_type,
                "magnetic_field_T": self.engine.environment.magnetic_field_T,
                "gravity_m_s2": self.engine.environment.gravity_m_s2,
                "viscosity": self.engine.environment.viscosity,
                "container_size_m": self.engine.environment.container_size_m,
                "boundary_condition": self.engine.environment.boundary_condition,
            },
            "spacetime": self.engine.spacetime.__dict__.copy(),
            "time": self.engine.time,
            "objects": [
                {
                    "label": obj.label,
                    "shape_type": obj.shape_type,
                    "phase_type": obj.phase_type,
                    "position": obj.position.tolist(),
                    "base_position": obj.base_position.tolist(),
                    "params": obj.params,
                }
                for obj in self.engine.objects
            ],
            "ui": {
                "educational_mode": self.educational_mode,
                "theme": self.theme,
                "field_lines": self.field_lines_var.get(),
                "curvature": self.curvature_var.get(),
            },
        }

    def _save_scene(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save Vortex Math scene",
            defaultextension=".json",
            filetypes=[("Vortex Math scene", "*.json"), ("JSON", "*.json")]
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(self._scene_dict(), file, indent=2)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))

    def _load_scene(self) -> None:
        filename = filedialog.askopenfilename(
            title="Load Vortex Math scene",
            filetypes=[("Vortex Math scene", "*.json"), ("JSON", "*.json")]
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as file:
                scene = json.load(file)

            environment = scene["environment"]
            for key, value in environment.items():
                setattr(self.engine.environment, key, value)

            for key, value in scene.get("spacetime", {}).items():
                setattr(self.engine.spacetime, key, value)

            self.engine.engine_type = scene.get("engine", "Analytic")
            self.engine.time = float(scene.get("time", 0.0))

            for obj, saved in zip(
                self.engine.objects, scene.get("objects", [])
            ):
                obj.shape_type = saved["shape_type"]
                obj.phase_type = saved["phase_type"]
                obj.position = np.asarray(saved["position"], dtype=float)
                obj.base_position = np.asarray(
                    saved["base_position"], dtype=float
                )
                obj.params.update(saved.get("params", {}))

            ui = scene.get("ui", {})
            if ui.get("educational_mode") in EDUCATIONAL_MODES:
                self.educational_mode = ui["educational_mode"]
                self.education_var.set(self.educational_mode)
            if ui.get("theme") in THEMES:
                self.theme_var.set(ui["theme"])
                self.theme = ui["theme"]
            self.field_lines_var.set(ui.get("field_lines", True))
            self.curvature_var.set(ui.get("curvature", True))

            self.domain_half_width = max(
                5.0, self.engine.environment.container_size_m
            )
            self.particles = None
            self._sync_control_values()
            self._apply_theme()
            self._set_engine()
            self._set_education()

        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Load failed", str(exc))

    # ------------------------------------------------------------------

    def _close(self) -> None:
        self.running = False
        if self._after_id:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = VortexMath5App(root)
    root.after(100, app._animation_tick)
    root.mainloop()


if __name__ == "__main__":
    main()
