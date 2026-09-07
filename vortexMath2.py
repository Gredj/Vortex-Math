#!/usr/bin/env python3
"""Two toroidal vortex rings rendered in 3D with live parameter sliders."""

import os

import numpy as np
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==========================
# Base parameters
# ==========================

L = 5.0
N = 60

dt = 0.02
density_threshold = 0.05

z1_init = -2.0
z2_init = 2.0

R1_init = 1.5
R2_init = 1.5
a1_init = 0.35
a2_init = 0.35
Gamma1_init = 5.0
Gamma2_init = 5.0
density_scale1_init = 1.0
density_scale2_init = 1.0
stretch_factor_init = 0.03
viscous_damping_init = 0.01

# ==========================
# Analytic functions
# ==========================


def self_induced_speed(R, a, Gamma):
    """Approximate self-induced axial speed of a thin vortex ring."""
    R_eff = max(float(R), 1e-3)
    a_eff = max(float(a), 1e-3)
    return (Gamma / (4.0 * np.pi * R_eff)) * (np.log(8.0 * R_eff / a_eff) - 0.25)


def torus_density(x, y, z, R, a, z_center=0.0, scale=1.0):
    """Gaussian density field representing a toroidal vortex core."""
    r = np.sqrt(x**2 + y**2)
    d = np.sqrt((r - R) ** 2 + (z - z_center) ** 2)
    return scale * np.exp(-(d**2) / (a**2))


def torus_mesh(R, tube_radius, z_center=0.0, phase=0.0):
    """Build a torus surface with a modest phase offset."""
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    phi = np.linspace(0.0, 2.0 * np.pi, 60)
    Theta, Phi = np.meshgrid(theta, phi)

    x = (R + tube_radius * np.cos(Theta + phase)) * np.cos(Phi)
    y = (R + tube_radius * np.cos(Theta + phase)) * np.sin(Phi)
    z = z_center + tube_radius * np.sin(Theta + phase)
    return x, y, z


# ==========================
# State variables
# ==========================

step = 0
z1 = z1_init
z2 = z2_init

R1 = R1_init
R2 = R2_init
a1 = a1_init
a2 = a2_init
Gamma1_curr = Gamma1_init
Gamma2_curr = Gamma2_init
density_scale1 = density_scale1_init
density_scale2 = density_scale2_init
stretch_factor = stretch_factor_init
viscous_damping = viscous_damping_init

# ==========================
# Figure and axes
# ==========================

fig = plt.figure(figsize=(11, 9))
ax3d = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.35)

ax3d.set_xlim(-L, L)
ax3d.set_ylim(-L, L)
ax3d.set_zlim(-L, L)
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
ax3d.set_title('Two Toroidal Vortex Rings')

# ==========================
# Slider axes
# ==========================

ax_R1 = plt.axes([0.10, 0.30, 0.35, 0.03])
ax_R2 = plt.axes([0.55, 0.30, 0.35, 0.03])
ax_a1 = plt.axes([0.10, 0.26, 0.35, 0.03])
ax_a2 = plt.axes([0.55, 0.26, 0.35, 0.03])
ax_G1 = plt.axes([0.10, 0.22, 0.35, 0.03])
ax_G2 = plt.axes([0.55, 0.22, 0.35, 0.03])
ax_d1 = plt.axes([0.10, 0.18, 0.35, 0.03])
ax_d2 = plt.axes([0.55, 0.18, 0.35, 0.03])
ax_stretch = plt.axes([0.10, 0.14, 0.35, 0.03])
ax_damp = plt.axes([0.55, 0.14, 0.35, 0.03])

# ==========================
# Sliders
# ==========================

s_R1 = Slider(ax_R1, 'R1', 0.3, 3.0, valinit=R1_init)
s_R2 = Slider(ax_R2, 'R2', 0.3, 3.0, valinit=R2_init)

s_a1 = Slider(ax_a1, 'a1', 0.1, 1.0, valinit=a1_init)
s_a2 = Slider(ax_a2, 'a2', 0.1, 1.0, valinit=a2_init)

s_G1 = Slider(ax_G1, 'Γ1', 0.1, 10.0, valinit=Gamma1_init)
s_G2 = Slider(ax_G2, 'Γ2', 0.1, 10.0, valinit=Gamma2_init)

s_d1 = Slider(ax_d1, 'dens1', 0.1, 3.0, valinit=density_scale1_init)
s_d2 = Slider(ax_d2, 'dens2', 0.1, 3.0, valinit=density_scale2_init)

s_stretch = Slider(ax_stretch, 'stretch', 0.0, 0.1, valinit=stretch_factor_init)
s_damp = Slider(ax_damp, 'damp', 0.0, 0.05, valinit=viscous_damping_init)

# ==========================
# Render helpers
# ==========================

def update_parameters_from_sliders():
    global R1, R2, a1, a2, Gamma1_curr, Gamma2_curr, density_scale1, density_scale2, stretch_factor, viscous_damping
    R1 = s_R1.val
    R2 = s_R2.val
    a1 = s_a1.val
    a2 = s_a2.val
    Gamma1_curr = s_G1.val
    Gamma2_curr = s_G2.val
    density_scale1 = s_d1.val
    density_scale2 = s_d2.val
    stretch_factor = s_stretch.val
    viscous_damping = s_damp.val


def draw_vortex_pair():
    global step, z1, z2

    ax3d.clear()
    ax3d.set_xlim(-L, L)
    ax3d.set_ylim(-L, L)
    ax3d.set_zlim(-L, L)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_title('Two Toroidal Vortex Rings')

    x1, y1, z1_mesh = torus_mesh(R1, a1, z1, phase=0.25)
    x2, y2, z2_mesh = torus_mesh(R2, a2, z2, phase=1.10)

    ax3d.plot_surface(
        x1,
        y1,
        z1_mesh,
        color=(0.12, 0.65, 0.95, 0.85),
        linewidth=0,
        antialiased=True,
        alpha=0.8,
        shade=True,
    )
    ax3d.plot_surface(
        x2,
        y2,
        z2_mesh,
        color=(0.96, 0.52, 0.25, 0.75),
        linewidth=0,
        antialiased=True,
        alpha=0.8,
        shade=True,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ring1_x = (R1 + a1 * np.cos(theta)) * np.cos(theta)
    ring1_y = (R1 + a1 * np.cos(theta)) * np.sin(theta)
    ring1_z = z1 + a1 * np.sin(theta)
    ring2_x = (R2 + a2 * np.cos(theta)) * np.cos(theta)
    ring2_y = (R2 + a2 * np.cos(theta)) * np.sin(theta)
    ring2_z = z2 + a2 * np.sin(theta)

    ax3d.plot(ring1_x, ring1_y, ring1_z, color='cyan', lw=1.4)
    ax3d.plot(ring2_x, ring2_y, ring2_z, color='orange', lw=1.4)

    # Vortex-density style field as a soft cloud around the core.
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    z = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    density = torus_density(X, Y, Z, R1, a1, z_center=z1, scale=density_scale1)
    density += torus_density(X, Y, Z, R2, a2, z_center=z2, scale=density_scale2)
    mask = density > density_threshold
    if np.any(mask):
        ax3d.scatter(
            X[mask],
            Y[mask],
            Z[mask],
            c=density[mask],
            cmap='plasma',
            s=8,
            alpha=0.35,
            depthshade=True,
        )

    fig.canvas.draw_idle()


# ==========================
# Animation and update callbacks
# ==========================

def update_state(_=None):
    global step, z1, z2, a1, a2

    update_parameters_from_sliders()

    phase = step * dt
    v1 = self_induced_speed(R1, a1, Gamma1_curr)
    v2 = self_induced_speed(R2, a2, Gamma2_curr)

    z1 = z1_init + 0.80 * np.sin(phase * 3.0 + 0.5) + 0.20 * v1 * stretch_factor
    z2 = z2_init - 0.80 * np.sin(phase * 3.0 + 1.2) - 0.20 * v2 * stretch_factor

    if viscous_damping > 0:
        a1 *= 1.0 - viscous_damping
        a2 *= 1.0 - viscous_damping
        a1 = np.clip(a1, 0.1, 1.0)
        a2 = np.clip(a2, 0.1, 1.0)

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


# ==========================
# Main entry point
# ==========================

def main():
    update_state()
    animation = FuncAnimation(fig, lambda frame: update_state(), interval=60, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
