# Plan parking example

Add a file parking.jl to the examples folder.

# Settings
SIM_TIME = 10 # simulation time
V_WIND = 10 # ground wind speed at 6 m height
TETHER_LENGTH = 150 # initial length [m]
ELEVATION = 72 # initial elevation angle
REL_DEPOWER = 0.3 # initial depower setting

# Code
- initialize the simulation
- run it in a loop
- visualize the result

# Visualization
- v_reelout (reel_out_speed)
- winch_force (single line
- elevation
- heading
- AoA
- rel_depower (logged each step, replaces l_diff — V3Kite has no
  front/back tether-length difference)
- L/D_wing and L/D_eff (wing lift / wing drag vs. wing lift / total drag)

Requires a new tether-drag helper in src/ (V3Kite has no total_drag
equivalent yet):
- `compute_tether_drag(sam)`: sum of `drag_force` for all non-WING
  points (tether + bridle + KCU), projected onto the apparent-wind
  direction — reuses the existing point-drag-projection pattern from
  `compute_tether_drag_coeff`/`compute_bridle_drag_coeff`/
  `compute_kcu_drag_coeff` in sim_helpers.jl, but returns a force [N]
  instead of a coefficient.
- `total_drag(s::V3KITE)` in interface.jl: `(wing_drag, tether_drag,
  wing_drag + tether_drag)`.

# Depower / winch behavior
Bake REL_DEPOWER into settle_wing (depower=REL_DEPOWER, steering=0.0),
so the wing starts already trimmed. Brake the single winch
(sys.winches[1].brake = true) for the whole loop — constant tether
length, no ramp logic needed.

# Add LaTeXStrings a package and use LaTeX for labels

