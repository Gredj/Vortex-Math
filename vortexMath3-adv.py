#!/usr/bin/env python3
"""Two toroidal fusion-plasma vortex rings with magnetic field overlays.

This script is intentionally modular: the physics constants and geometry are kept in
USER_EDITABLE_PARAMETERS so they can be tuned in a text editor without hunting through
multiple functions. It is designed as a visual overlay for a higher-fidelity plasma or
fluid engine rather than a full solver.
"""

import os

import numpy as np
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==============================================================================
# USER-EDITABLE PARAMETERS
# Edit these values in a code editor to change the plasma behavior.
# ==============================================================================
USER_EDITABLE_PARAMETERS = {
    "domain": {
        "half_width": 5.0,
        "grid_size": 60,
    },
    "time": {
        "dt": 0.02,
        "density_threshold": 0.05,
    },
    "ring_1": {
        "label": "Ring A",
        "base_radius": 1.5,
        "tube_radius": 0.35,
        "z_center": -2.0,
        "mass": 1.0,
        "temperature": 1.4,
        "circulation": 5.0,
        "density_scale": 1.0,
        "radius_growth_rate": 0.012,
        "magnetic_strength": 1.8,
        "color": (0.12, 0.65, 0.95, 0.85),
    },
    "ring_2": {
        "label": "Ring B",
        "base_radius": 1.55,
        "tube_radius": 0.38,
        "z_center": 2.0,
        "mass": 1.05,
        "temperature": 1.7,
        "circulation": 5.2,
        "density_scale": 1.1,
        "radius_growth_rate": 0.014,
        "magnetic_strength": 2.1,
        "color": (0.96, 0.52, 0.25, 0.75),
    },
    "global": {
        "stretch_factor": 0.03,
        "viscous_damping": 0.01,
        "instability_strength": 0.0,
        "reaction_energy": 0.0,
    },
    "visualization": {
        "num_particles": 300,
    },
}

# ============================================================================== 
# Simulation state
# ============================================================================== 
L = USER_EDITABLE_PARAMETERS["domain"]["half_width"]
N = USER_EDITABLE_PARAMETERS["domain"]["grid_size"]

dt = USER_EDITABLE_PARAMETERS["time"]["dt"]
density_threshold = USER_EDITABLE_PARAMETERS["time"]["density_threshold"]

ring1 = {
    "label": USER_EDITABLE_PARAMETERS["ring_1"]["label"],
    "base_radius": USER_EDITABLE_PARAMETERS["ring_1"]["base_radius"],
    "radius": USER_EDITABLE_PARAMETERS["ring_1"]["base_radius"],
    "tube_radius": USER_EDITABLE_PARAMETERS["ring_1"]["tube_radius"],
    "tube_radius_base": USER_EDITABLE_PARAMETERS["ring_1"]["tube_radius"],
    "z_center": USER_EDITABLE_PARAMETERS["ring_1"]["z_center"],
    "mass": USER_EDITABLE_PARAMETERS["ring_1"]["mass"],
    "temperature": USER_EDITABLE_PARAMETERS["ring_1"]["temperature"],
    "circulation": USER_EDITABLE_PARAMETERS["ring_1"]["circulation"],
    "density_scale": USER_EDITABLE_PARAMETERS["ring_1"]["density_scale"],
    "radius_growth_rate": USER_EDITABLE_PARAMETERS["ring_1"]["radius_growth_rate"],
    "magnetic_strength": USER_EDITABLE_PARAMETERS["ring_1"]["magnetic_strength"],
    "color": USER_EDITABLE_PARAMETERS["ring_1"]["color"],
}

ring2 = {
    "label": USER_EDITABLE_PARAMETERS["ring_2"]["label"],
    "base_radius": USER_EDITABLE_PARAMETERS["ring_2"]["base_radius"],
    "radius": USER_EDITABLE_PARAMETERS["ring_2"]["base_radius"],
    "tube_radius": USER_EDITABLE_PARAMETERS["ring_2"]["tube_radius"],
    "tube_radius_base": USER_EDITABLE_PARAMETERS["ring_2"]["tube_radius"],
    "z_center": USER_EDITABLE_PARAMETERS["ring_2"]["z_center"],
    "mass": USER_EDITABLE_PARAMETERS["ring_2"]["mass"],
    "temperature": USER_EDITABLE_PARAMETERS["ring_2"]["temperature"],
    "circulation": USER_EDITABLE_PARAMETERS["ring_2"]["circulation"],
    "density_scale": USER_EDITABLE_PARAMETERS["ring_2"]["density_scale"],
    "radius_growth_rate": USER_EDITABLE_PARAMETERS["ring_2"]["radius_growth_rate"],
    "magnetic_strength": USER_EDITABLE_PARAMETERS["ring_2"]["magnetic_strength"],
    "color": USER_EDITABLE_PARAMETERS["ring_2"]["color"],
}

stretch_factor = USER_EDITABLE_PARAMETERS["global"]["stretch_factor"]
viscous_damping = USER_EDITABLE_PARAMETERS["global"]["viscous_damping"]
instability_strength = USER_EDITABLE_PARAMETERS["global"]["instability_strength"]
reaction_energy = USER_EDITABLE_PARAMETERS["global"]["reaction_energy"]
num_particles = USER_EDITABLE_PARAMETERS["visualization"]["num_particles"]
particles = None
sim_time = 0.0
step = 0

equation_artists = []

# ============================================================================== 
# Figure and GUI
# ============================================================================== 
fig = plt.figure(figsize=(14, 10))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_position([0.07, 0.18, 0.55, 0.72])

# README-style scientific panel on right


def draw_equation_panel():
    """Render the governing equations and scientific units in the GUI."""
    global equation_artists
    for artist in equation_artists:
        artist.remove()
    equation_artists.clear()

    eqns = [
        r"$v_{self} \approx \frac{\Gamma}{4\pi R}\left(\ln\frac{8R}{a} - \frac{1}{4}\right)$",
        r"$\rho = \mathrm{scale}\,\exp\left(-\frac{d^2}{a^2}\right)$",
        r"$\frac{dR}{dt} = k\,\frac{T}{m}\left(1 + \alpha\,\Gamma\right)$",
        r"$\mathbf{B} \approx \nabla \times \mathbf{A}$",
        r"$\nabla\cdot\mathbf{B} = 0$",
    ]
    units = [
        r"R, a in m; $\Gamma$ in m$^2$/s",
        r"T in K; m in kg; $\rho$ in kg/m$^3$",
        r"t, dt in s; k is a scaling coefficient",
        r"B in Tesla; A in Wb/m",
        r"SI-like scientific units",
    ]

    title = fig.text(0.68, 0.76, "Governing equations", fontsize=11, weight='bold')
    equation_artists.append(title)
    y = 0.72
    for i, eq in enumerate(eqns):
        eq_text = fig.text(0.68, y, eq, fontsize=9)
        unit_text = fig.text(0.85, y, units[i], fontsize=8)
        equation_artists.extend([eq_text, unit_text])
        y -= 0.075

    fig.text(0.68, 0.18, "Academic overlay: visualizing vortex\nplasma motion and magnetic geometry.", fontsize=8, color="0.35")


# Slider axes
ax_R1 = plt.axes([0.68, 0.62, 0.24, 0.02])
ax_R2 = plt.axes([0.68, 0.58, 0.24, 0.02])
ax_a1 = plt.axes([0.68, 0.54, 0.24, 0.02])
ax_a2 = plt.axes([0.68, 0.50, 0.24, 0.02])
ax_G1 = plt.axes([0.68, 0.46, 0.24, 0.02])
ax_G2 = plt.axes([0.68, 0.42, 0.24, 0.02])
ax_d1 = plt.axes([0.68, 0.38, 0.24, 0.02])
ax_d2 = plt.axes([0.68, 0.34, 0.24, 0.02])
ax_stretch = plt.axes([0.68, 0.30, 0.24, 0.02])
ax_damp = plt.axes([0.68, 0.26, 0.24, 0.02])
ax_instab = plt.axes([0.68, 0.22, 0.24, 0.02])
ax_reaction = plt.axes([0.68, 0.18, 0.24, 0.02])
ax_play = plt.axes([0.68, 0.10, 0.11, 0.045])
ax_pause = plt.axes([0.81, 0.10, 0.11, 0.045])

s_R1 = Slider(ax_R1, 'R1', 0.8, 3.0, valinit=ring1["base_radius"])
s_R2 = Slider(ax_R2, 'R2', 0.8, 3.0, valinit=ring2["base_radius"])
s_a1 = Slider(ax_a1, 'a1', 0.10, 1.0, valinit=ring1["tube_radius"])
s_a2 = Slider(ax_a2, 'a2', 0.10, 1.0, valinit=ring2["tube_radius"])
s_G1 = Slider(ax_G1, 'Γ1', 0.5, 12.0, valinit=ring1["circulation"])
s_G2 = Slider(ax_G2, 'Γ2', 0.5, 12.0, valinit=ring2["circulation"])
s_d1 = Slider(ax_d1, 'ρ1', 0.1, 2.5, valinit=ring1["density_scale"])
s_d2 = Slider(ax_d2, 'ρ2', 0.1, 2.5, valinit=ring2["density_scale"])
s_stretch = Slider(ax_stretch, 'stretch', 0.0, 0.1, valinit=stretch_factor)
s_damp = Slider(ax_damp, 'damp', 0.0, 0.05, valinit=viscous_damping)
s_instab = Slider(ax_instab, 'instab', 0.0, 1.0, valinit=instability_strength)
s_reaction = Slider(ax_reaction, 'Q_rxn', 0.0, 2.0, valinit=reaction_energy)
b_play = Button(ax_play, 'Play')
b_pause = Button(ax_pause, 'Pause')

# ============================================================================== 
# Analytic functions
# ============================================================================== 

def initialize_particles():
    """Seed particles around both toroids to visualize chaotic plasma motion."""
    global particles
    if num_particles <= 0:
        particles = np.empty((0, 3))
        return

    particles = np.zeros((num_particles, 3), dtype=float)
    ring_choice = np.random.randint(0, 2, size=num_particles)
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=num_particles)
    radial_offsets = np.random.uniform(-0.5, 0.5, size=num_particles)
    z_offsets = np.random.uniform(-0.5, 0.5, size=num_particles)

    for i in range(num_particles):
        ring = ring1 if ring_choice[i] == 0 else ring2
        radius = ring["radius"] + ring["tube_radius"] * radial_offsets[i]
        particles[i, 0] = radius * np.cos(theta[i])
        particles[i, 1] = radius * np.sin(theta[i])
        particles[i, 2] = ring["z_center"] + ring["tube_radius"] * z_offsets[i]


def advect_particles():
    """Evolve tracer particles in the vortex flow field."""
    global particles
    if particles is None or particles.size == 0:
        initialize_particles()
        return

    xp = particles[:, 0]
    yp = particles[:, 1]
    zp = particles[:, 2]

    U1, V1, W1 = vortex_velocity_field(xp, yp, zp, ring1["radius"], ring1["tube_radius"], ring1["z_center"], ring1["circulation"])
    U2, V2, W2 = vortex_velocity_field(xp, yp, zp, ring2["radius"], ring2["tube_radius"], ring2["z_center"], ring2["circulation"])

    particles[:, 0] += dt * (U1 + U2)
    particles[:, 1] += dt * (V1 + V2)
    particles[:, 2] += dt * (W1 + W2)

    out_of_bounds = np.any(np.abs(particles) > L * 1.2, axis=1)
    if np.any(out_of_bounds):
        mask_idx = np.where(out_of_bounds)[0]
        ring_choice = np.random.randint(0, 2, size=len(mask_idx))
        theta = np.random.uniform(0.0, 2.0 * np.pi, size=len(mask_idx))
        particles[mask_idx, 0] = 0.0
        particles[mask_idx, 1] = 0.0
        particles[mask_idx, 2] = 0.0
        for j, choice in enumerate(ring_choice):
            ring = ring1 if choice == 0 else ring2
            idx = mask_idx[j]
            particles[idx, 0] = ring["radius"] * np.cos(theta[j])
            particles[idx, 1] = ring["radius"] * np.sin(theta[j])
            particles[idx, 2] = ring["z_center"] + ring["tube_radius"] * np.random.uniform(-0.5, 0.5)


def torus_density(x, y, z, R, a, z_center=0.0, scale=1.0):
    """Gaussian density field for a toroidal plasma core."""
    r = np.sqrt(x**2 + y**2)
    d = np.sqrt((r - R) ** 2 + (z - z_center) ** 2)
    return scale * np.exp(-(d**2) / (a**2 + 1e-6))


def self_induced_speed(R, a, Gamma):
    """Approximate self-induced axial speed of a thin vortex ring."""
    R_eff = max(float(R), 1e-3)
    a_eff = max(float(a), 1e-3)
    return (Gamma / (4.0 * np.pi * R_eff)) * (np.log(8.0 * R_eff / a_eff) - 0.25)


def torus_mesh(R, tube_radius, z_center=0.0, phase=0.0):
    """Build a torus surface with a small phase offset."""
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    phi = np.linspace(0.0, 2.0 * np.pi, 60)
    Theta, Phi = np.meshgrid(theta, phi)

    x = (R + tube_radius * np.cos(Theta + phase)) * np.cos(Phi)
    y = (R + tube_radius * np.cos(Theta + phase)) * np.sin(Phi)
    z = z_center + tube_radius * np.sin(Theta + phase)
    return x, y, z


def ring_radius_at_time(ring, time_value):
    """Time-dependent torus radius growth rule."""
    thermal_factor = 1.0 + 0.75 * ring["temperature"]
    mass_factor = 1.0 / (0.5 + ring["mass"])
    return ring["base_radius"] + ring["radius_growth_rate"] * time_value * thermal_factor * mass_factor


def magnetic_field_vector(X, Y, Z, ring):
    """Approximate toroidal magnetic field around the ring core."""
    eps = 1e-3
    r = np.sqrt(X**2 + Y**2)
    core_distance = np.sqrt((r - ring["radius"]) ** 2 + (Z - ring["z_center"]) ** 2)
    gauss = np.exp(-(core_distance**2) / ((ring["tube_radius"] ** 2) + eps))

    radial = np.sqrt(r**2 + eps)
    Bx = -ring["magnetic_strength"] * Y * gauss / (radial + eps)
    By = ring["magnetic_strength"] * X * gauss / (radial + eps)
    Bz = 0.2 * ring["magnetic_strength"] * (Z - ring["z_center"]) * gauss / (radial + eps)
    return Bx, By, Bz


def vortex_velocity_field(X, Y, Z, R, a, z_center, Gamma):
    """Construct a compact vortex-induced velocity field around a torus."""
    eps = 1e-3
    r = np.sqrt(X**2 + Y**2)
    d = np.sqrt((r - R) ** 2 + (Z - z_center) ** 2)
    gaussian = np.exp(-(d**2) / (a**2 + eps))

    safe_r = np.where(r > 0.0, r, 1.0)
    x_term = -Gamma * Y / (safe_r**2 + eps)
    y_term = Gamma * X / (safe_r**2 + eps)
    z_term = 0.5 * Gamma * (Z - z_center) / (np.sqrt(r**2 + (Z - z_center) ** 2) + eps)

    return x_term * gaussian, y_term * gaussian, z_term * gaussian


def update_parameters_from_sliders():
    """Synchronize the slider values back into the state dictionary."""
    global stretch_factor, viscous_damping, instability_strength, reaction_energy, num_particles

    ring1["base_radius"] = s_R1.val
    ring2["base_radius"] = s_R2.val
    ring1["tube_radius"] = s_a1.val
    ring2["tube_radius"] = s_a2.val
    ring1["circulation"] = s_G1.val
    ring2["circulation"] = s_G2.val
    ring1["density_scale"] = s_d1.val
    ring2["density_scale"] = s_d2.val
    stretch_factor = s_stretch.val
    viscous_damping = s_damp.val
    instability_strength = s_instab.val
    reaction_energy = s_reaction.val
    num_particles = max(50, int(round(USER_EDITABLE_PARAMETERS["visualization"]["num_particles"])))


# ============================================================================== 
# Rendering
# ============================================================================== 

def draw_vortex_pair():
    global sim_time

    ax3d.clear()
    ax3d.set_xlim(-L, L)
    ax3d.set_ylim(-L, L)
    ax3d.set_zlim(-L, L)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_title('Fusion Plasma Toroidal Vortex Rings')

    ring1["radius"] = ring_radius_at_time(ring1, sim_time)
    ring2["radius"] = ring_radius_at_time(ring2, sim_time)

    reaction_gain = 1.0 + 0.12 * reaction_energy * np.sin(sim_time * 2.5)
    ring1["tube_radius"] = max(0.12, ring1["tube_radius_base"] * reaction_gain * (1.0 + 0.06 * ring1["temperature"] * np.tanh(sim_time / 3.0)))
    ring2["tube_radius"] = max(0.12, ring2["tube_radius_base"] * reaction_gain * (1.0 + 0.06 * ring2["temperature"] * np.tanh(sim_time / 3.0)))

    ring1["z_center"] = USER_EDITABLE_PARAMETERS["ring_1"]["z_center"] + 0.8 * np.sin(sim_time * 3.0 + 0.5)
    ring2["z_center"] = USER_EDITABLE_PARAMETERS["ring_2"]["z_center"] - 0.8 * np.sin(sim_time * 3.0 + 1.2)

    x1, y1, z1_mesh = torus_mesh(ring1["radius"], ring1["tube_radius"], ring1["z_center"], phase=0.25)
    x2, y2, z2_mesh = torus_mesh(ring2["radius"], ring2["tube_radius"], ring2["z_center"], phase=1.10)

    ax3d.plot_surface(x1, y1, z1_mesh, color=ring1["color"], linewidth=0, antialiased=True, alpha=0.82, shade=True)
    ax3d.plot_surface(x2, y2, z2_mesh, color=ring2["color"], linewidth=0, antialiased=True, alpha=0.82, shade=True)

    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ring1_x = (ring1["radius"] + ring1["tube_radius"] * np.cos(theta)) * np.cos(theta)
    ring1_y = (ring1["radius"] + ring1["tube_radius"] * np.cos(theta)) * np.sin(theta)
    ring1_z = ring1["z_center"] + ring1["tube_radius"] * np.sin(theta)
    ring2_x = (ring2["radius"] + ring2["tube_radius"] * np.cos(theta)) * np.cos(theta)
    ring2_y = (ring2["radius"] + ring2["tube_radius"] * np.cos(theta)) * np.sin(theta)
    ring2_z = ring2["z_center"] + ring2["tube_radius"] * np.sin(theta)

    ax3d.plot(ring1_x, ring1_y, ring1_z, color='cyan', lw=1.4)
    ax3d.plot(ring2_x, ring2_y, ring2_z, color='orange', lw=1.4)

    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    z = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    density1 = torus_density(X, Y, Z, ring1["radius"], ring1["tube_radius"], ring1["z_center"], ring1["density_scale"])
    density2 = torus_density(X, Y, Z, ring2["radius"], ring2["tube_radius"], ring2["z_center"], ring2["density_scale"])
    density = density1 + density2
    temp_field = ring1["temperature"] * density1 + ring2["temperature"] * density2
    mask = density > density_threshold
    if np.any(mask):
        ax3d.scatter(X[mask], Y[mask], Z[mask], c=temp_field[mask], cmap='inferno', s=8, alpha=0.42, depthshade=True)

    advect_particles()
    if particles is not None and particles.size > 0:
        ax3d.scatter(particles[:, 0], particles[:, 1], particles[:, 2], s=4, c='white', alpha=0.7)

    field_x = np.linspace(-L, L, 8)
    field_y = np.linspace(-L, L, 8)
    field_z = np.linspace(-L, L, 8)
    FX, FY, FZ = np.meshgrid(field_x, field_y, field_z, indexing='ij')

    U1, V1, W1 = vortex_velocity_field(FX, FY, FZ, ring1["radius"], ring1["tube_radius"], ring1["z_center"], ring1["circulation"])
    U2, V2, W2 = vortex_velocity_field(FX, FY, FZ, ring2["radius"], ring2["tube_radius"], ring2["z_center"], ring2["circulation"])

    B1x, B1y, B1z = magnetic_field_vector(FX, FY, FZ, ring1)
    B2x, B2y, B2z = magnetic_field_vector(FX, FY, FZ, ring2)

    field_mask = (
        np.abs(U1) + np.abs(V1) + np.abs(W1)
        + np.abs(U2) + np.abs(V2) + np.abs(W2)
        + np.abs(B1x) + np.abs(B1y) + np.abs(B1z)
        + np.abs(B2x) + np.abs(B2y) + np.abs(B2z)
    ) > 1e-4

    if np.any(field_mask):
        ax3d.quiver(
            FX[field_mask],
            FY[field_mask],
            FZ[field_mask],
            (U1 + U2 + B1x + B2x)[field_mask],
            (V1 + V2 + B1y + B2y)[field_mask],
            (W1 + W2 + B1z + B2z)[field_mask],
            color='white',
            length=0.24,
            normalize=True,
            alpha=0.48,
        )

    draw_equation_panel()
    fig.canvas.draw_idle()


# ============================================================================== 
# Animation and update callbacks
# ============================================================================== 

def update_state(_=None):
    global sim_time, step

    update_parameters_from_sliders()

    phase = step * dt
    v1 = self_induced_speed(ring1["radius"], ring1["tube_radius"], ring1["circulation"])
    v2 = self_induced_speed(ring2["radius"], ring2["tube_radius"], ring2["circulation"])

    ring1["z_center"] = USER_EDITABLE_PARAMETERS["ring_1"]["z_center"] + 0.80 * np.sin(phase * 3.0 + 0.5) + 0.20 * v1 * stretch_factor
    ring2["z_center"] = USER_EDITABLE_PARAMETERS["ring_2"]["z_center"] - 0.80 * np.sin(phase * 3.0 + 1.2) - 0.20 * v2 * stretch_factor

    if instability_strength > 0:
        ring1["radius"] += instability_strength * 0.01 * np.sin(sim_time * 6.0 + 0.5)
        ring2["radius"] += instability_strength * 0.01 * np.sin(sim_time * 6.0 + 1.4)

    if reaction_energy > 0:
        ring1["temperature"] = USER_EDITABLE_PARAMETERS["ring_1"]["temperature"] + 0.05 * reaction_energy * np.sin(sim_time * 2.0)
        ring2["temperature"] = USER_EDITABLE_PARAMETERS["ring_2"]["temperature"] + 0.05 * reaction_energy * np.cos(sim_time * 2.0)

    sim_time += dt

    if viscous_damping > 0:
        ring1["tube_radius"] *= 1.0 - viscous_damping
        ring2["tube_radius"] *= 1.0 - viscous_damping
        ring1["tube_radius"] = np.clip(ring1["tube_radius"], 0.1, 1.0)
        ring2["tube_radius"] = np.clip(ring2["tube_radius"], 0.1, 1.0)

    step += 1
    draw_vortex_pair()


s_R1.on_changed(lambda val: update_state(val))
s_R2.on_changed(lambda val: update_state(val))
s_a1.on_changed(lambda val: update_state(val))
s_a2.on_changed(lambda val: update_state(val))
s_G1.on_changed(lambda val: update_state(val))
s_G2.on_changed(lambda val: update_state(val))
s_d1.on_changed(lambda val: update_state(val))
s_d2.on_changed(lambda val: update_state(val))
s_stretch.on_changed(lambda val: update_state(val))
s_damp.on_changed(lambda val: update_state(val))
s_instab.on_changed(lambda val: update_state(val))
s_reaction.on_changed(lambda val: update_state(val))

# ============================================================================== 
# Main entry point
# ============================================================================== 

def main():
    global animation

    initialize_particles()
    draw_equation_panel()
    draw_vortex_pair()
    animation = FuncAnimation(fig, lambda frame: update_state(), interval=60, blit=False)

    def play(_event):
        animation.event_source.start()

    def pause(_event):
        animation.event_source.stop()

    b_play.on_clicked(play)
    b_pause.on_clicked(pause)
    animation.event_source.stop()
    plt.show()


if __name__ == "__main__":
    main()
