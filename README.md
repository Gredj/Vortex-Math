# Vortex-Math
This is a python script to simulate two volumes moving in space

<img width="1362" height="947" alt="image" src="https://github.com/user-attachments/assets/15affcf7-ea2f-4eb7-a5d4-95437577ca74" />

<img width="547" height="427" alt="image" src="https://github.com/user-attachments/assets/165641a8-96e5-498d-b313-13d001d126f3" />

<img width="544" height="421" alt="image" src="https://github.com/user-attachments/assets/91234538-813e-4c8f-a411-294d46ddfd33" />

started to get mature after this version

4.1

<img width="1907" height="985" alt="image" src="https://github.com/user-attachments/assets/713a7f01-c8a5-4b2d-b9dd-6f7509ae047a" />

<img width="1901" height="992" alt="image" src="https://github.com/user-attachments/assets/bd3e0ea3-1aee-4669-86b0-aae263a8acc0" />

<img width="1902" height="980" alt="image" src="https://github.com/user-attachments/assets/ac2d09b5-b301-44f3-b12f-4d5c378d88a7" />

5.0

<img width="1906" height="965" alt="image" src="https://github.com/user-attachments/assets/4e4d0e79-1c9f-49e6-8060-09f592619250" />

6.0 

quite fond of this version. It removes the 4.0 engine reliance which was in 5.0

<img width="1913" height="991" alt="image" src="https://github.com/user-attachments/assets/963f136b-0bca-4010-93fd-b34e5d70a8c0" />

7.0 

we might be on this version for a while...

<img width="1907" height="1011" alt="image" src="https://github.com/user-attachments/assets/7e2293e6-dea4-463a-8cf2-fde7df5dff6a" />

for some code explantions

 It's a self-contained Tkinter + Matplotlib app that visualizes toy vortex/field physics in 3D. A VortexEngine holds FlowObjects (sphere, torus, cylinder, disk, shell, sheet, filament, spheromak) that each generate an analytic density/velocity/magnetic field, plus an Environment (pressure, temperature, magnetic strength, gravity, viscosity, container) and a Spacetime config (time scale/warp/dilation, visual curvature). The GUI lets you drag sliders, apply presets ("Fusion Reactor," "Jet vs Sheet," etc.), animate, and export screenshots/CSVs/scene JSON.

The part that matches your goal: there's already an extensibility layer built for exactly what you described — FieldEngine is a Protocol (duck-typed interface: step, fields, metrics), EngineRegistry lets you register and switch between multiple engines at runtime, OpenSourceEngineAdapter is a template for wrapping an external simulator, FieldRegistry/FieldDefinition let a new engine declare what fields it exposes, and PluginRegistry + VortexExtensionAPI let you bolt on analysis (like the built-in Diagnostics for vorticity/divergence) without touching the GUI. So the "universal simulator overhead" concept isn't aspirational here — the seams for swapping in a model-trained engine already exist; a new engine just has to implement .step()/.fields()/.metrics() and call register_engine().

This here is an after image

<img width="1782" height="956" alt="image" src="https://github.com/user-attachments/assets/e8de7311-d941-492a-be4c-6bcdb352bcb1" />

The idea is one particle splits another and makes the second object into a torrid. this torrid then collides with another torrid of a different diameter and splits that torrid. making two going the way of the 'larger' on and then the original smaller one that got spun by the first particle. The tidal forces on that second larger torrid that was split into two then remerges with exceptional force enough to create an implosion with the right enviroment





