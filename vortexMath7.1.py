#!/usr/bin/env python3
"""Vortex Math Field Library: a single-file, extensible physics-engine host.

WHAT THIS IS
============
This file is two things stacked on top of each other:

  1. A small but complete analytic field simulator (VortexEngine) that models
     vortex-like shapes -- sphere, torus, cylinder, disk, shell, sheet,
     filament, spheromak -- as density, velocity, and magnetic fields, plus a
     Tkinter/Matplotlib GUI to sculpt, animate, and inspect them in 3D.

  2. A thin, deliberately generic "engine host" layer (FieldEngine,
     EngineRegistry, FieldRegistry, PluginRegistry, VortexExtensionAPI) that
     the GUI talks to instead of talking to VortexEngine directly. The built-in
     analytic engine is registered through this same layer, as
     NativeVortexEngineAdapter, so it is not a special case.

The point of the host layer is that it does not care what is actually
producing the fields. Anything that can hand back a dict of density/velocity/
magnetic arrays on request -- a hand-written formula, a CFD solver, a neural
surrogate trained on simulation or experimental data -- can be wrapped in a
few methods (step/fields/metrics) and dropped in next to the built-in engine.
Swapping engines at runtime, discovering what fields an engine exposes, and
running diagnostics or parameter sweeps against whichever engine is active are
all handled generically, so a new engine gets all of that for free instead of
each one reinventing it.

WHO THIS IS FOR
===============
Two overlapping audiences:

  - Someone who wants to spend a day inside one file, learning by poking at
    it: turn a knob, watch the vorticity number move, break a preset, put it
    back together. Nothing here is hidden behind an import you can't open.

  - Someone building or training a physics engine (analytic, CFD, or learned)
    who wants a lightweight harness to plug it into, visualize its output
    next to a known-good reference engine, and get diagnostics/export tooling
    without writing their own GUI.

The analytic equations in VortexEngine are intentionally simple teaching
models, not a validated CFD or MHD solver -- treat their numbers as
illustrative, not physically authoritative, unless a plugged-in engine says
otherwise.
"""

from __future__ import annotations

import csv
import json
import tkinter as tk
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Protocol

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SHAPES = ("Sphere", "Torus", "Cylinder", "Disk", "Shell", "Sheet", "Filament", "Spheromak")
PHASES = ("Gas", "Liquid", "Plasma")
CONTAINERS = ("Box", "Sphere", "Cylinder")
BOUNDARIES = ("Reflective", "Absorbing", "Periodic")
PRESETS = ("Fusion Reactor", "Sphere to Torus", "Jet vs Sheet", "Magnetic Bottle")
MU_0 = 4.0e-7 * np.pi


@dataclass
class Environment:
    shape: str = "Box"
    pressure: float = 1.0
    temperature: float = 300.0
    magnetic: float = 1.0
    gravity: float = 0.0
    viscosity: float = 0.01
    container_size: float = 5.0
    boundary: str = "Reflective"

    @property
    def density_factor(self) -> float:
        return max(0.05, self.pressure * 300.0 / max(self.temperature, 1.0))


@dataclass
class Spacetime:
    time_scale: float = 1.0
    time_warp: float = 0.0
    time_dilation: float = 1.0
    curvature: float = 0.0
    field_amplification: float = 1.0
    smoothing: float = 0.0
    sampling: int = 7


@dataclass
class FlowObject:
    label: str
    shape: str
    phase: str
    position: np.ndarray
    params: dict[str, float]
    color: tuple[float, float, float, float]

    @property
    def radius(self) -> float:
        return max(self.params.get("radius", 1.5), 0.05)

    @property
    def thickness(self) -> float:
        return max(self.params.get("thickness", 0.35), 0.03)

    def _coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
        dx, dy, dz = x - self.position[0], y - self.position[1], z - self.position[2]
        return dx, dy, dz, np.sqrt(dx * dx + dy * dy)

    def density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        dx, dy, dz, radial = self._coords(x, y, z)
        radius, thickness = self.radius, self.thickness
        if self.shape == "Sphere":
            value = np.exp(-(dx * dx + dy * dy + dz * dz) / radius**2)
        elif self.shape == "Torus":
            value = np.exp(-((radial - radius) ** 2 + dz * dz) / thickness**2)
        elif self.shape == "Cylinder":
            value = np.exp(-((radial / radius) ** 4)) * np.exp(-((dz / radius) ** 8))
        elif self.shape in {"Disk", "Sheet"}:
            value = np.exp(-((radial / radius) ** 6) - (dz / thickness) ** 2)
        elif self.shape == "Shell":
            value = np.exp(-((np.sqrt(dx * dx + dy * dy + dz * dz) - radius) ** 2) / thickness**2)
        elif self.shape == "Filament":
            value = np.exp(-(radial / thickness) ** 2 - (dz / radius) ** 2)
        else:
            envelope = np.exp(-(dx * dx + dy * dy + dz * dz) / radius**2)
            core = np.exp(-((radial - radius * 0.5) ** 2 + dz * dz) / thickness**2)
            value = envelope * core
        return max(self.params.get("density", 1.0), 0.0) * value

    def velocity(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
        density = self.density(x, y, z)
        dx, dy, dz, radial = self._coords(x, y, z)
        safe = np.maximum(radial, 1e-3)
        circulation = self.params.get("circulation", 5.0)
        swirl = circulation * density / (safe + 0.2)
        u, v, w = -dy / safe * swirl, dx / safe * swirl, np.zeros_like(x)
        if self.shape in {"Cylinder", "Filament"}:
            w = circulation * density
        elif self.shape == "Sheet":
            u = 0.7 * circulation * density
        return u, v, w

    def magnetic_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.phase != "Plasma":
            return tuple(np.zeros_like(x) for _ in range(3))
        u, v, w = self.velocity(x, y, z)
        strength = self.params.get("magnetic", 1.8)
        return strength * v, -strength * u, 0.35 * strength * w

    def surface(self, resolution: int = 22) -> tuple[np.ndarray, ...] | None:
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)
        r, t = self.radius, self.thickness
        if self.shape in {"Torus", "Spheromak"}:
            major = r if self.shape == "Torus" else r * 0.55
            x = (major + t * np.cos(theta)) * np.cos(phi)
            y = (major + t * np.cos(theta)) * np.sin(phi)
            z = t * np.sin(theta)
        elif self.shape in {"Sphere", "Shell"}:
            x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
        elif self.shape == "Cylinder":
            x, y, z = r * np.cos(theta), r * np.sin(theta), r * np.sin(phi)
        elif self.shape in {"Disk", "Sheet"}:
            x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), t * np.cos(theta)
        else:
            return None
        return x + self.position[0], y + self.position[1], z + self.position[2]


class VortexEngine:
    """Self-contained analytic engine used exclusively by Vortex Math 6.0."""

    def __init__(self) -> None:
        self.environment = Environment()
        self.spacetime = Spacetime()
        self.objects = self._default_objects()
        self.plugins = PluginRegistry()
        self.extensions = VortexExtensionAPI(self)
        self.engine_registry = EngineRegistry()
        self.engine_registry.register(NativeVortexEngineAdapter(self))
        self.field_registry = FieldRegistry()
        self._register_builtin_fields()
        self.grid_size = 30
        self.domain = 5.0
        self.time = 0.0

    @staticmethod
    def _default_objects() -> list[FlowObject]:
        return [
            FlowObject(
                "Object 1", "Torus", "Plasma", np.array([0.0, 0.0, -2.0]),
                {"radius": 1.5, "thickness": 0.35, "density": 1.0,
                 "circulation": 5.0, "magnetic": 1.8},
                (0.12, 0.65, 0.95, 0.85)
            ),
            FlowObject(
                "Object 2", "Torus", "Plasma", np.array([0.0, 0.0, 2.0]),
                {"radius": 1.55, "thickness": 0.38, "density": 1.1,
                 "circulation": 5.2, "magnetic": 2.1},
                (0.96, 0.52, 0.25, 0.8)
            ),
        ]

    def grid(self) -> tuple[np.ndarray, ...]:
        values = np.linspace(-self.domain, self.domain, self.grid_size)
        return np.meshgrid(values, values, values, indexing="ij")

    def _register_builtin_fields(self) -> None:
        self.field_registry.register(FieldDefinition(
            "density", "normalized", "Scalar mass-density proxy."
        ))
        self.field_registry.register(FieldDefinition(
            "velocity", "m/s", "Three-component velocity field.", "vector"
        ))
        self.field_registry.register(FieldDefinition(
            "magnetic", "T", "Three-component magnetic field.", "vector"
        ))

    def register_engine(self, engine: FieldEngine) -> None:
        self.engine_registry.register(engine)

    def select_engine(self, name: str) -> None:
        self.engine_registry.select(name)

    @property
    def active_engine(self) -> FieldEngine:
        return self.engine_registry.active

    def step(self, dt: float = 0.02) -> None:
        warp = 1.0 + self.spacetime.time_warp * np.tanh(self.time / 5.0)
        actual = dt * self.spacetime.time_scale * warp / max(self.spacetime.time_dilation, 0.05)
        for index, obj in enumerate(self.objects):
            direction = 1.0 if index == 0 else -1.0
            obj.position[2] += direction * 0.8 * np.sin(self.time * 3.0 + index * 0.7) * actual
        self.time += actual

    def fields(self, grid: tuple[np.ndarray, ...]) -> dict[str, Any]:
        x, y, z = grid
        density = np.zeros_like(x)
        velocity = [np.zeros_like(x) for _ in range(3)]
        magnetic = [np.zeros_like(x) for _ in range(3)]
        for obj in self.objects:
            part = obj.density(x, y, z) * self.environment.density_factor
            density += part
            values = obj.velocity(x, y, z)
            field = obj.magnetic_field(x, y, z)
            for axis in range(3):
                velocity[axis] += values[axis]
                magnetic[axis] += field[axis] * self.environment.magnetic
        velocity = [part / (1.0 + self.environment.viscosity) for part in velocity]
        velocity[2] -= self.environment.gravity * density * 0.01
        return {"density": density, "velocity": velocity, "magnetic": magnetic}

    def metrics(self, fields: dict[str, Any]) -> dict[str, float]:
        density = fields["density"]
        velocity = fields["velocity"]
        magnetic = fields["magnetic"]
        kinetic = sum(float(np.mean(part * part)) for part in velocity) * 0.5
        magnetic_energy = sum(float(np.mean(part * part)) for part in magnetic) / (2 * MU_0)
        return {"time": self.time, "density": float(np.mean(density)), "kinetic": kinetic,
                "magnetic_energy": magnetic_energy,
                "magnetic_pressure": self.environment.magnetic**2 / (2 * MU_0)}


THEMES = {
    "Scientific": ("#e9edf2", "#f7f9fb", "#ffffff", "#17202a", "#1769aa"),
    "Neon Plasma": ("#07121d", "#0b1a28", "#07121d", "#e8f8ff", "#45d9ff"),
    "Wireframe": ("#e8edf0", "#f4f6f7", "#eef2f3", "#18242d", "#2b687e"),
    "Dark Mode": ("#101318", "#171a21", "#111318", "#f3f5f7", "#66c7ff"),
}



# ---------------------------------------------------------------------------
# EXTENSIBILITY LAYER
# ---------------------------------------------------------------------------

class FieldEngine(Protocol):
    """Adapter contract for any internal or external simulation engine."""
    name: str
    description: str

    def step(self, dt: float) -> None:
        ...

    def fields(self, grid: tuple[np.ndarray, ...]) -> dict[str, Any]:
        ...

    def metrics(self, fields: dict[str, Any]) -> dict[str, float]:
        ...


class FieldPlugin(Protocol):
    """Protocol for optional field-analysis plugins."""
    name: str

    def analyze(self, fields: dict[str, Any], engine: "VortexEngine") -> dict[str, float]:
        ...



class NativeVortexEngineAdapter:
    """Expose the native VortexEngine through the FieldEngine contract."""

    name = "Vortex Analytic Engine"
    description = "Built-in analytic vortex/field model."

    def __init__(self, engine: "VortexEngine") -> None:
        self.engine = engine

    def step(self, dt: float) -> None:
        self.engine.step(dt)

    def fields(self, grid: tuple[np.ndarray, ...]) -> dict[str, Any]:
        return self.engine.fields(grid)

    def metrics(self, fields: dict[str, Any]) -> dict[str, float]:
        return self.engine.metrics(fields)


class EngineRegistry:
    """Registry for native, third-party, or open-source engine adapters."""

    def __init__(self) -> None:
        self._engines: dict[str, FieldEngine] = {}
        self._active: str | None = None

    def register(self, engine: FieldEngine) -> None:
        if not getattr(engine, "name", "").strip():
            raise ValueError("Engine must provide a non-empty name.")
        self._engines[engine.name] = engine
        if self._active is None:
            self._active = engine.name

    def names(self) -> tuple[str, ...]:
        return tuple(self._engines.keys())

    def select(self, name: str) -> None:
        if name not in self._engines:
            raise KeyError(f"Unknown engine: {name}")
        self._active = name

    @property
    def active(self) -> FieldEngine:
        if self._active is None:
            raise RuntimeError("No field engine is registered.")
        return self._engines[self._active]


class OpenSourceEngineAdapter:
    """Base adapter for external CFD/MHD/high-energy libraries.

    A third-party library should be wrapped rather than imported by the GUI.
    Subclasses only need to translate the library's state into the common
    Vortex Math field contract.
    """

    name = "External Engine"
    description = "External simulation engine adapter."

    def __init__(self, external_engine: Any) -> None:
        self.external_engine = external_engine

    def step(self, dt: float) -> None:
        if hasattr(self.external_engine, "step"):
            self.external_engine.step(dt)
        elif hasattr(self.external_engine, "advance"):
            self.external_engine.advance(dt)
        else:
            raise NotImplementedError("External engine needs step() or advance().")

    def fields(self, grid: tuple[np.ndarray, ...]) -> dict[str, Any]:
        if hasattr(self.external_engine, "fields"):
            return self.external_engine.fields(grid)
        if hasattr(self.external_engine, "get_fields"):
            return self.external_engine.get_fields(grid)
        raise NotImplementedError("External engine must expose fields() or get_fields().")

    def metrics(self, fields: dict[str, Any]) -> dict[str, float]:
        if hasattr(self.external_engine, "metrics"):
            return self.external_engine.metrics(fields)
        return {
            "time": float(getattr(self.external_engine, "time", 0.0)),
            "density": float(np.mean(fields.get("density", 0.0))),
        }


@dataclass
class FieldDefinition:
    """Metadata for a scalar/vector field exposed by an engine."""

    name: str
    unit: str = ""
    description: str = ""
    kind: str = "scalar"


class FieldRegistry:
    """Metadata registry so visualizers can discover fields dynamically."""

    def __init__(self) -> None:
        self._fields: dict[str, FieldDefinition] = {}

    def register(self, definition: FieldDefinition) -> None:
        self._fields[definition.name] = definition

    def definitions(self) -> tuple[FieldDefinition, ...]:
        return tuple(self._fields.values())


class Diagnostics:
    """Non-invasive diagnostics operating on the engine's existing field API."""

    @staticmethod
    def _gradient_components(a: np.ndarray, spacing: float) -> tuple[np.ndarray, ...]:
        return tuple(np.gradient(a, spacing, edge_order=1))

    @classmethod
    def analyze(cls, fields: dict[str, Any], engine: "VortexEngine") -> dict[str, float]:
        density = fields["density"]
        velocity = fields["velocity"]
        magnetic = fields["magnetic"]

        domain = max(getattr(engine, "domain", engine.environment.container_size), 5.0)
        spacing = (2.0 * domain) / max(engine.grid_size - 1, 1)

        du_dx, du_dy, du_dz = cls._gradient_components(velocity[0], spacing)
        dv_dx, dv_dy, dv_dz = cls._gradient_components(velocity[1], spacing)
        dw_dx, dw_dy, dw_dz = cls._gradient_components(velocity[2], spacing)

        db_dx, db_dy, db_dz = cls._gradient_components(magnetic[0], spacing)
        eb_dx, eb_dy, eb_dz = cls._gradient_components(magnetic[1], spacing)
        fb_dx, fb_dy, fb_dz = cls._gradient_components(magnetic[2], spacing)

        velocity_divergence = du_dx + dv_dy + dw_dz
        magnetic_divergence = db_dx + eb_dy + fb_dz

        vorticity_x = dw_dy - dv_dz
        vorticity_y = du_dz - dw_dx
        vorticity_z = dv_dx - du_dy

        current_x = fb_dy - eb_dz
        current_y = db_dz - fb_dx
        current_z = eb_dx - db_dy

        velocity_magnitude = np.sqrt(sum(component * component for component in velocity))
        magnetic_magnitude = np.sqrt(sum(component * component for component in magnetic))
        vorticity_magnitude = np.sqrt(
            vorticity_x * vorticity_x +
            vorticity_y * vorticity_y +
            vorticity_z * vorticity_z
        )
        current_magnitude = np.sqrt(
            current_x * current_x +
            current_y * current_y +
            current_z * current_z
        )

        return {
            "density_mean": float(np.mean(density)),
            "density_std": float(np.std(density)),
            "velocity_mean": float(np.mean(velocity_magnitude)),
            "velocity_max": float(np.max(velocity_magnitude)),
            "magnetic_mean": float(np.mean(magnetic_magnitude)),
            "magnetic_max": float(np.max(magnetic_magnitude)),
            "vorticity_mean": float(np.mean(vorticity_magnitude)),
            "vorticity_max": float(np.max(vorticity_magnitude)),
            "velocity_divergence_rms": float(np.sqrt(np.mean(velocity_divergence ** 2))),
            "magnetic_divergence_rms": float(np.sqrt(np.mean(magnetic_divergence ** 2))),
            "current_density_proxy_mean": float(np.mean(current_magnitude)),
        }


class DiagnosticsPlugin:
    name = "Field Diagnostics"

    def analyze(self, fields: dict[str, Any], engine: "VortexEngine") -> dict[str, float]:
        return Diagnostics.analyze(fields, engine)


class EnergyBudgetPlugin:
    """Bolt-on plugin: tracks a rolling energy budget over simulated time.

    This is a second, independent FieldPlugin -- it does not replace or
    depend on DiagnosticsPlugin. Where Diagnostics reports the field's
    current state, this plugin remembers a short history of it, so a caller
    (typically a GUI panel) can plot how energy moves between kinetic,
    magnetic, and a density-proxy "potential" term as the sim runs, and watch
    whether the total drifts -- a lightweight, at-a-glance conservation check
    that only makes sense as a track record, not a single snapshot.

    It follows the same registration contract as any other plugin:

        engine.extensions.register_plugin(EnergyBudgetPlugin())

    No change to VortexEngine, PluginRegistry, or DiagnosticsPlugin was
    needed to add this -- that absence of a required core edit is the whole
    point of the bolt-on plugin seam.
    """

    name = "Energy Budget"

    def __init__(self, max_history: int = 400) -> None:
        self.max_history = max_history
        self.history: list[dict[str, float]] = []
        self.enabled = True

    def reset(self) -> None:
        self.history.clear()

    def analyze(self, fields: dict[str, Any], engine: "VortexEngine") -> dict[str, float]:
        density = fields["density"]
        velocity = fields["velocity"]
        magnetic = fields["magnetic"]

        kinetic = sum(float(np.mean(part * part)) for part in velocity) * 0.5
        magnetic_energy = sum(float(np.mean(part * part)) for part in magnetic) / (2 * MU_0)
        density_proxy = float(np.mean(density * density)) * 0.5
        total = kinetic + magnetic_energy + density_proxy

        previous_total = self.history[-1]["total"] if self.history else total
        drift = total - previous_total

        if self.enabled:
            self.history.append({
                "time": engine.time,
                "kinetic": kinetic,
                "magnetic_energy": magnetic_energy,
                "density_proxy": density_proxy,
                "total": total,
                "drift": drift,
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)

        return {
            "energy_kinetic": kinetic,
            "energy_magnetic": magnetic_energy,
            "energy_density_proxy": density_proxy,
            "energy_total": total,
            "energy_drift": drift,
        }


@dataclass
class ExperimentResult:
    parameter: str
    values: list[float]
    metrics: list[dict[str, float]]
    baseline: float | None = None


@dataclass
class ParameterExperiment:
    """Reusable parameter-sweep experiment.

    parameter examples:
        environment.magnetic
        environment.viscosity
        objects.0.params.density
    """
    parameter: str
    start: float
    stop: float
    steps: int = 20
    measurements: tuple[str, ...] = ("density", "kinetic", "magnetic_energy")

    def _get(self, engine: "VortexEngine") -> float:
        target: Any = engine
        parts = self.parameter.split(".")
        for part in parts:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part) if hasattr(target, part) else target[part]
        return float(target)

    def _set(self, engine: "VortexEngine", value: float) -> None:
        parts = self.parameter.split(".")
        target: Any = engine
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part) if hasattr(target, part) else target[part]

        final = parts[-1]
        if final.isdigit():
            target[int(final)] = value
        elif isinstance(target, dict):
            target[final] = value
        else:
            setattr(target, final, value)

    def run(self, engine: "VortexEngine") -> ExperimentResult:
        if self.steps < 2:
            raise ValueError("Experiment steps must be at least 2.")

        original = self._get(engine)
        values = np.linspace(self.start, self.stop, self.steps)
        measurements: list[dict[str, float]] = []

        try:
            for value in values:
                self._set(engine, float(value))
                fields = engine.active_engine.fields(engine.grid())
                base = engine.metrics(fields)
                diagnostics = Diagnostics.analyze(fields, engine)
                combined = {**base, **diagnostics}
                measurements.append({
                    name: float(combined[name])
                    for name in self.measurements
                    if name in combined
                })
        finally:
            self._set(engine, original)

        return ExperimentResult(
            parameter=self.parameter,
            values=[float(v) for v in values],
            metrics=measurements,
            baseline=original,
        )


class PluginRegistry:
    """Central registry for bolt-on analysis capabilities."""

    def __init__(self) -> None:
        self.field_plugins: dict[str, FieldPlugin] = {}
        self.register_field_plugin(DiagnosticsPlugin())

    def register_field_plugin(self, plugin: FieldPlugin) -> None:
        self.field_plugins[plugin.name] = plugin

    def analyze(self, fields: dict[str, Any], engine: "VortexEngine") -> dict[str, float]:
        results: dict[str, float] = {}
        for plugin in self.field_plugins.values():
            results.update(plugin.analyze(fields, engine))
        return results


class VortexExtensionAPI:
    """Small public API intended for future external modules."""

    def __init__(self, engine: "VortexEngine") -> None:
        self.engine = engine

    def add_object(self, obj: FlowObject) -> None:
        self.engine.objects.append(obj)

    def remove_object(self, index: int) -> FlowObject:
        if not 0 <= index < len(self.engine.objects):
            raise IndexError("Object index out of range.")
        return self.engine.objects.pop(index)

    def diagnostics(self, fields: dict[str, Any] | None = None) -> dict[str, float]:
        fields = fields or self.engine.fields(self.engine.grid())
        return self.engine.plugins.analyze(fields, self.engine)

    def experiment(self, experiment: ParameterExperiment) -> ExperimentResult:
        return experiment.run(self.engine)

    def register_engine(self, engine: FieldEngine) -> None:
        self.engine.register_engine(engine)

    def register_plugin(self, plugin: FieldPlugin) -> None:
        self.engine.plugins.register_field_plugin(plugin)

    def register_field(self, definition: FieldDefinition) -> None:
        self.engine.field_registry.register(definition)


def register_plugin(engine: "VortexEngine", plugin: FieldPlugin) -> None:
    """Convenience function for third-party bolt-on plugins."""
    engine.plugins.register_field_plugin(plugin)



class VortexMath6App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vortex Math 7.1 — Extensible Field Laboratory")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 680)
        self.engine = VortexEngine()
        self.grid_size = 30
        self.running = False
        self.theme = "Scientific"
        self.vars: dict[str, Any] = {}
        self._render_pending = False
        # Bolt-on: this plugin is not part of VortexEngine's own setup. It is
        # created and registered here, from the outside, through the same
        # VortexExtensionAPI a third-party module would use.
        self.energy_plugin = EnergyBudgetPlugin()
        self.engine.extensions.register_plugin(self.energy_plugin)
        self._build_ui()
        self._apply_theme()
        self._render()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=(18, 12))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="VORTEX MATH 7.0", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(header, text="Standalone analytic field laboratory", font=("Segoe UI", 9)).pack(anchor="w")
        self.status = ttk.Label(header, text="PAUSED", font=("Segoe UI", 9, "bold"))
        self.status.pack(anchor="e")
        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.controls = ttk.Frame(pane, padding=10)
        visual = ttk.Frame(pane, padding=8)
        pane.add(self.controls, weight=0)
        pane.add(visual, weight=1)
        self.notebook = ttk.Notebook(self.controls)
        self.notebook.pack(fill="both", expand=True)
        self.object_tab, self.env_tab, self.time_tab, self.visual_tab, self.analysis_tab = [ttk.Frame(self.notebook, padding=10) for _ in range(5)]
        for tab, name in ((self.object_tab, "Objects"), (self.env_tab, "Environment"), (self.time_tab, "Time"), (self.visual_tab, "Visuals"), (self.analysis_tab, "Analysis")):
            self.notebook.add(tab, text=name)
        self._build_objects(); self._build_environment(); self._build_time(); self._build_visuals(); self._build_analysis()
        visual.columnconfigure(0, weight=1); visual.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=visual)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.lesson = ttk.Label(visual, text="", wraplength=900, justify="left")
        self.lesson.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        section = ttk.LabelFrame(parent, text=title, padding=10)
        section.pack(fill="x", pady=(0, 10))
        return section

    def _build_objects(self) -> None:
        for index, obj in enumerate(self.engine.objects):
            section = self._section(self.object_tab, obj.label)
            choice_box = ttk.Frame(section)
            choice_box.grid(row=0, column=0, columnspan=2, sticky="ew")
            choice_box.columnconfigure(1, weight=1)
            ttk.Label(choice_box, text="Shape").grid(row=0, column=0, sticky="w", pady=4)
            shape = tk.StringVar(value=obj.shape); self.vars[f"shape{index}"] = shape
            ttk.Combobox(choice_box, textvariable=shape, values=SHAPES, state="readonly").grid(row=0, column=1, sticky="ew", padx=8)
            shape.trace_add("write", lambda *_args, i=index: self._object_changed(i))
            ttk.Label(choice_box, text="Phase").grid(row=1, column=0, sticky="w", pady=4)
            phase = tk.StringVar(value=obj.phase); self.vars[f"phase{index}"] = phase
            ttk.Combobox(choice_box, textvariable=phase, values=PHASES, state="readonly").grid(row=1, column=1, sticky="ew", padx=8)
            phase.trace_add("write", lambda *_args, i=index: self._object_changed(i))
            parameter_box = ttk.Frame(section)
            parameter_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
            for key, label, low, high in (("radius", "Radius", .2, 3), ("thickness", "Thickness", .05, 1), ("density", "Density", 0, 3), ("circulation", "Circulation", 0, 10), ("magnetic", "Magnetic", 0, 5)):
                self._slider(parameter_box, f"{key}{index}", label, low, high, obj.params[key], lambda value, i=index, k=key: self._param_changed(i, k, value))
            section.columnconfigure(1, weight=1)
        preset = self._section(self.object_tab, "Presets")
        self.preset = tk.StringVar(value=PRESETS[0])
        ttk.Combobox(preset, textvariable=self.preset, values=PRESETS, state="readonly").pack(fill="x", pady=(0, 6))
        ttk.Button(preset, text="Apply preset", command=self._apply_preset).pack(fill="x")

    def _build_environment(self) -> None:
        for key, label, low, high in (("pressure", "Pressure", .1, 5), ("temperature", "Temperature", 100, 1500), ("magnetic", "Magnetic field", 0, 10), ("gravity", "Gravity", 0, 20), ("viscosity", "Viscosity", 0, 1), ("container_size", "Container size", 2, 10)):
            self._slider(self.env_tab, key, label, low, high, getattr(self.engine.environment, key), lambda value, k=key: self._env_changed(k, value))
        section = self._section(self.env_tab, "Container")
        self.container = tk.StringVar(value=self.engine.environment.shape)
        for value in CONTAINERS: ttk.Radiobutton(section, text=value, variable=self.container, value=value, command=self._env_options).pack(anchor="w", pady=2)
        section = self._section(self.env_tab, "Boundary")
        self.boundary = tk.StringVar(value=self.engine.environment.boundary)
        for value in BOUNDARIES: ttk.Radiobutton(section, text=value, variable=self.boundary, value=value, command=self._env_options).pack(anchor="w", pady=2)

    def _build_time(self) -> None:
        for key, label, low, high in (("time_scale", "Time scale", 0, 4), ("time_warp", "Time warp", -2, 2), ("time_dilation", "Time dilation", .25, 4)):
            self._slider(self.time_tab, key, label, low, high, getattr(self.engine.spacetime, key), lambda value, k=key: setattr(self.engine.spacetime, k, value))
        section = self._section(self.time_tab, "Simulation")
        for text, command in (("Play", lambda: self._set_running(True)), ("Pause", lambda: self._set_running(False)), ("Step", self._step)):
            ttk.Button(section, text=text, command=command).pack(side="left", expand=True, fill="x", padx=2)

    def _build_visuals(self) -> None:
        section = self._section(self.visual_tab, "Theme")
        self.theme_var = tk.StringVar(value=self.theme)
        for value in THEMES: ttk.Radiobutton(section, text=value, variable=self.theme_var, value=value, command=self._apply_theme).pack(anchor="w", pady=2)
        section = self._section(self.visual_tab, "Simulation Engine")
        self.engine_var = tk.StringVar(value=self.engine.engine_registry.active.name)
        self.engine_combo = ttk.Combobox(
            section,
            textvariable=self.engine_var,
            values=self.engine.engine_registry.names(),
            state="readonly"
        )
        self.engine_combo.pack(fill="x", pady=(0, 4))
        ttk.Label(
            section,
            text="External CFD/MHD engines can be added through adapters.",
            wraplength=260,
            justify="left"
        ).pack(fill="x", pady=(0, 4))
        ttk.Button(
            section,
            text="Activate engine",
            command=self._select_engine
        ).pack(fill="x")
        section = self._section(self.visual_tab, "Overlays")
        self.field_lines = tk.BooleanVar(value=True); self.curvature = tk.BooleanVar(value=True)
        ttk.Checkbutton(section, text="Magnetic field lines", variable=self.field_lines, command=self._queue_render).pack(anchor="w", pady=3)
        ttk.Checkbutton(section, text="Curvature rings", variable=self.curvature, command=self._queue_render).pack(anchor="w", pady=3)
        ttk.Button(section, text="Save screenshot", command=self._save_image).pack(fill="x", pady=(8, 2))
        ttk.Button(section, text="Export metrics CSV", command=self._export_metrics).pack(fill="x", pady=2)
        ttk.Button(section, text="Export diagnostics CSV", command=self._export_diagnostics).pack(fill="x", pady=2)
        ttk.Button(section, text="Save scene", command=self._save_scene).pack(fill="x", pady=2)
        ttk.Button(section, text="Load scene", command=self._load_scene).pack(fill="x", pady=2)

    def _build_analysis(self) -> None:
        section = self._section(self.analysis_tab, "Energy Budget")
        ttk.Label(
            section,
            text="Kinetic, magnetic, and density-proxy energy, tracked over time by\n"
                 "the Energy Budget plugin -- a bolt-on, not a core engine feature.",
            wraplength=280,
            justify="left"
        ).pack(fill="x", pady=(0, 6))
        self.track_energy = tk.BooleanVar(value=self.energy_plugin.enabled)
        ttk.Checkbutton(
            section, text="Track history", variable=self.track_energy,
            command=self._toggle_energy_tracking
        ).pack(anchor="w", pady=3)
        button_row = ttk.Frame(section)
        button_row.pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Reset history", command=self._reset_energy_history).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(button_row, text="Export CSV", command=self._export_energy_history).pack(side="left", expand=True, fill="x", padx=(3, 0))

        readout_section = self._section(self.analysis_tab, "Latest values")
        self.energy_readout = ttk.Label(readout_section, text="No data yet.", justify="left", wraplength=280)
        self.energy_readout.pack(fill="x")

        chart_section = self._section(self.analysis_tab, "History")
        self.energy_figure = Figure(figsize=(3.4, 2.6), dpi=100)
        self.energy_ax = self.energy_figure.add_subplot(111)
        self.energy_canvas = FigureCanvasTkAgg(self.energy_figure, master=chart_section)
        self.energy_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _slider(self, parent: ttk.Frame, key: str, label: str, low: float, high: float, initial: float, callback: Any) -> None:
        row = ttk.Frame(parent); row.pack(fill="x", pady=4); row.columnconfigure(0, weight=1)
        value = tk.StringVar(value=f"{initial:.2f}"); self.vars[key] = value
        ttk.Label(row, text=label).grid(row=0, column=0, sticky="w"); ttk.Label(row, textvariable=value, width=8, anchor="e").grid(row=0, column=1)
        scale = tk.Scale(row, from_=low, to=high, resolution=(high-low)/200, showvalue=False, highlightthickness=0, bd=0, orient="horizontal", command=lambda raw: (value.set(f"{float(raw):.2f}"), callback(float(raw))))
        scale.set(initial); scale.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.vars[f"scale_{key}"] = scale

    def _object_changed(self, index: int) -> None:
        self.engine.objects[index].shape = self.vars[f"shape{index}"].get(); self.engine.objects[index].phase = self.vars[f"phase{index}"].get(); self._queue_render()

    def _param_changed(self, index: int, key: str, value: float) -> None:
        self.engine.objects[index].params[key] = value; self._queue_render()

    def _env_changed(self, key: str, value: float) -> None:
        setattr(self.engine.environment, key, value)
        if key == "container_size": self.domain = max(5.0, value)
        self._queue_render()

    def _env_options(self) -> None:
        self.engine.environment.shape = self.container.get(); self.engine.environment.boundary = self.boundary.get(); self._queue_render()

    def _apply_preset(self) -> None:
        value = self.preset.get(); a, b = self.engine.objects
        if value == "Fusion Reactor": a.shape = b.shape = "Torus"; a.phase = b.phase = "Plasma"; self.engine.environment.magnetic = 5
        elif value == "Sphere to Torus": a.shape, b.shape = "Sphere", "Torus"; self.engine.environment.magnetic = 2
        elif value == "Jet vs Sheet": a.shape, b.shape = "Cylinder", "Sheet"; a.phase = b.phase = "Gas"; self.engine.environment.magnetic = .5
        else: a.shape, b.shape = "Spheromak", "Torus"; a.phase = b.phase = "Plasma"; self.engine.environment.shape = "Sphere"; self.engine.environment.magnetic = 8
        self._sync_controls(); self._queue_render(immediate=True)

    def _sync_controls(self) -> None:
        for index, obj in enumerate(self.engine.objects): self.vars[f"shape{index}"].set(obj.shape); self.vars[f"phase{index}"].set(obj.phase)
        self.container.set(self.engine.environment.shape); self._queue_render()

    def _grid(self) -> tuple[np.ndarray, ...]:
        domain = getattr(self, "domain", 5.0); values = np.linspace(-domain, domain, self.grid_size); return np.meshgrid(values, values, values, indexing="ij")

    def _render(self) -> None:
        self._render_pending = False; self.ax.clear(); theme = THEMES[self.theme]
        self.ax.set_facecolor(theme[2]); self.figure.patch.set_facecolor(theme[0]); domain = getattr(self, "domain", 5.0)
        self.ax.set_xlim(-domain, domain); self.ax.set_ylim(-domain, domain); self.ax.set_zlim(-domain, domain)
        for obj in self.engine.objects:
            mesh = obj.surface()
            if mesh is not None: self.ax.plot_surface(*mesh, color=obj.color, alpha=.35, linewidth=0)
        fields = self.engine.active_engine.fields(self._grid()); mask = fields["density"] > .05
        if np.any(mask):
            x, y, z = self._grid()
            scatter: Any = self.ax.scatter
            scatter(x[mask], y[mask], z[mask], c=fields["density"][mask], cmap="inferno", s=4, alpha=.25)
        self.ax.set_title(f"Vortex Math 7.1 | t = {self.engine.time:.2f} s | {self.engine.environment.magnetic:.2f} T", color=theme[3], pad=12)
        self.ax.set_xlabel("x [m]"); self.ax.set_ylabel("y [m]"); self.ax.set_zlabel("z [m]")
        values = self.engine.active_engine.metrics(fields)
        diagnostics = self.engine.plugins.analyze(fields, self.engine)
        self._update_energy_panel()
        self.lesson.configure(
            text=(
                f"Density {values['density']:.3e}   "
                f"Kinetic {values['kinetic']:.3e}   "
                f"Magnetic energy {values['magnetic_energy']:.3e}   "
                f"Magnetic pressure {values['magnetic_pressure']:.3e} Pa\n"
                f"Vorticity RMS {diagnostics['vorticity_mean']:.3e}   "
                f"∇·v RMS {diagnostics['velocity_divergence_rms']:.3e}   "
                f"∇·B RMS {diagnostics['magnetic_divergence_rms']:.3e}"
            )
        )
        self.canvas.draw_idle()

    def _queue_render(self, immediate: bool = False) -> None:
        if immediate: self._render(); return
        if not self._render_pending: self._render_pending = True; self.root.after(35, self._render)

    def _animation_tick(self) -> None:
        if self.running: self.engine.active_engine.step(0.02); self._queue_render()
        if self.root.winfo_exists(): self.root.after(40, self._animation_tick)

    def _set_running(self, running: bool) -> None:
        if running and not self.running: self.root.after(0, self._animation_tick)
        self.running = running; self.status.configure(text="RUNNING" if running else "PAUSED")

    def _step(self) -> None: self.engine.step(); self._queue_render(immediate=True)

    def _toggle_energy_tracking(self) -> None:
        self.energy_plugin.enabled = self.track_energy.get()

    def _reset_energy_history(self) -> None:
        self.energy_plugin.reset()
        self._update_energy_panel()

    def _export_energy_history(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=(("CSV file", "*.csv"),))
        if not path:
            return
        history = self.energy_plugin.history
        if not history:
            messagebox.showinfo("Export energy history", "No history recorded yet.")
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    def _update_energy_panel(self) -> None:
        history = self.energy_plugin.history
        theme = THEMES[self.theme]
        self.energy_ax.clear()
        self.energy_ax.set_facecolor(theme[2])
        self.energy_figure.patch.set_facecolor(theme[0])
        if history:
            times = [entry["time"] for entry in history]
            self.energy_ax.plot(times, [entry["kinetic"] for entry in history], label="Kinetic", linewidth=1.2)
            self.energy_ax.plot(times, [entry["magnetic_energy"] for entry in history], label="Magnetic", linewidth=1.2)
            self.energy_ax.plot(times, [entry["total"] for entry in history], label="Total", linewidth=1.6, linestyle="--")
            self.energy_ax.legend(fontsize=6, loc="upper left")
            latest = history[-1]
            self.energy_readout.configure(
                text=(
                    f"t = {latest['time']:.2f} s\n"
                    f"Kinetic {latest['kinetic']:.3e}   Magnetic {latest['magnetic_energy']:.3e}\n"
                    f"Density proxy {latest['density_proxy']:.3e}\n"
                    f"Total {latest['total']:.3e}   Drift/step {latest['drift']:+.3e}"
                )
            )
        else:
            self.energy_readout.configure(text="No data yet." if self.energy_plugin.enabled else "Tracking paused.")
        self.energy_ax.set_xlabel("t [s]", fontsize=7, color=theme[3])
        self.energy_ax.tick_params(labelsize=6, colors=theme[3])
        self.energy_canvas.draw_idle()

    def _select_engine(self) -> None:
        try:
            self.engine.select_engine(self.engine_var.get())
            self.status.configure(text=f"ENGINE: {self.engine.active_engine.name}")
            self._queue_render(immediate=True)
        except (KeyError, RuntimeError) as error:
            messagebox.showerror("Simulation engine", str(error))

    def _save_image(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG image", "*.png"),))
        if path: self.figure.savefig(path, dpi=180, bbox_inches="tight")

    def _export_metrics(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=(("CSV file", "*.csv"),))
        if path:
            fields = self.engine.fields(self._grid()); values = self.engine.metrics(fields)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=values.keys())
                writer.writeheader()
                writer.writerow(values)

    def _export_diagnostics(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV file", "*.csv"),)
        )
        if path:
            fields = self.engine.fields(self._grid())
            diagnostics = self.engine.plugins.analyze(fields, self.engine)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=diagnostics.keys())
                writer.writeheader()
                writer.writerow(diagnostics)

    def _save_scene(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("JSON scene", "*.json"),))
        if path:
            data = {"version": "7.1", "format": "vortex-scene", "environment": asdict(self.engine.environment), "spacetime": asdict(self.engine.spacetime), "time": self.engine.time, "objects": [{"label": o.label, "shape": o.shape, "phase": o.phase, "position": o.position.tolist(), "params": o.params} for o in self.engine.objects]}
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_scene(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("JSON scene", "*.json"),))
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8")); self.engine.environment = Environment(**data["environment"]); self.engine.spacetime = Spacetime(**data["spacetime"]); self.engine.time = data.get("time", 0)
            for obj, saved in zip(self.engine.objects, data["objects"]): obj.shape, obj.phase, obj.params = saved["shape"], saved["phase"], saved["params"]; obj.position = np.array(saved["position"], dtype=float)
            self._sync_controls(); self._render()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error: messagebox.showerror("Load scene", str(error))

    def _apply_theme(self) -> None:
        self.theme = self.theme_var.get(); window, panel, canvas, text, accent = THEMES[self.theme]; self.root.configure(bg=window); style = ttk.Style(self.root); style.configure(".", background=panel, foreground=text, font=("Segoe UI", 9)); style.configure("TFrame", background=panel); style.configure("TLabel", background=panel, foreground=text); style.configure("TLabelframe", background=panel, foreground=text); style.configure("TLabelframe.Label", background=panel, foreground=text); self._queue_render(immediate=True)


def main() -> None:
    root = tk.Tk(); VortexMath6App(root); root.mainloop()


if __name__ == "__main__": main()