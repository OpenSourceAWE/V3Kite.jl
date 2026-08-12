# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Beam-Wing Example

The same depower/steering ramp as a particle-lattice run, on the Timoshenko-beam
wing built from the SurfplanAdapter export by `V3Kite.SurfplanAdapter`. Everything
below the geometry filenames is shared with the particle model: the tapes are found
by name instead of by index, so `STRUC_YAML` is the only line that differs.

`aero_geometry.yaml` is reused unchanged — the beam is emitted in the same CAD frame
(see `V3_ADAPTER_FRAME_OFFSET`), so the VSM sections still sit on the wing.

The bridle is the measured 2025 line system, not the export's design bridle, so the
KCU, the pulleys, the M-line and the three tapes are the flown ones.

Built on the `KernelBackend`, which assembles one kernel per component instead of
compiling a single monolithic ODE — the 22 beam bodies and 21 joints make the
monolithic build the dominant cost here.

`BODY_DAMPING` and `WORLD_DAMPING` are both set on every point. Body-frame damping
damps a node against its wing frame, which only `DYNAMIC` points get, so on this
geometry — whose wing nodes are `BODY_STATIC` points riding bodies — it reaches
nothing and only the bridle-carrying world-frame term acts. The wing itself is
damped by the Timoshenko joints and the canopy segments, whose damping lives in
the YAML.

`STRUC_YAML` and the relaxed state it starts from are both in git and both come
from their own example: `v3beam_geometry.jl` emits the geometry from the
SurfplanAdapter export, `relax_bridle.jl` relaxes it and logs the state. Neither
runs here — this example only flies what they wrote, so a run costs nothing but
the run.

The three knobs below decide what a bridle line does as it goes slack and are set
on the loaded structure by [`apply_bridle_material!`](@ref), so sweeping them
needs neither a rewrite nor a recompile. `COMPRESSION_DAMPING_FRAC =
COMPRESSION_FRAC` leaves the bridle with a single damping ratio instead of one
that jumps the moment a line unloads, and both at `0` make a slack line carry no
force at all. The two compression fractions reach the canopy membrane segments as
well as the bridle, since fabric is tension-only for the same reason a line is,
while `BRIDLE_DAMPING_PER_STIFFNESS` reaches only the lines and tapes. The
segment count is not among them: that one changes how many segments exist, so it
lives in `v3beam_geometry.jl`.

The YAML carries constant Breukels bending per joint because it cannot hold a
callable. Pointing `ADAPTER_DIR` at the export swaps in the curvature-softening
Comer-Levy law, which is what lets a tube keep bending past its collapse moment.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using MakieControlPlots
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

STRUC_YAML = "struc_geometry_beam.yaml"
AERO_YAML = "aero_geometry.yaml"
VSM_SETTINGS = "vsm_settings.yaml"

BRIDLE_DAMPING_PER_STIFFNESS = 0.001 # bridle unit_damping / unit_stiffness [s]
COMPRESSION_FRAC = 0.01              # stiffness left under compression
COMPRESSION_DAMPING_FRAC = 1.0      # damping left under compression

# Export the beam was built from; `nothing` keeps the linear Breukels bending,
# the export directory swaps in the curvature-softening Comer-Levy law.
ADAPTER_DIR = nothing

# BODY_DAMPING reaches nothing on a beam wing, see the docstring above.
BODY_DAMPING = [0.0, 0.0, 20.0]
WORLD_DAMPING = [0.0, 0.0, 0.0]

SIM_TIME = 10.0
FPS = 20
V_WIND = 15.4
TETHER_LENGTH = 250.0
ELEVATION = 70.0      # degrees
DEPOWER = 0.20        # fraction [0, 1]; has to match relax_bridle.jl
STEERING = 0.0        # fraction [-1, 1]
RAMP_TIME = 5.0       # seconds to ramp the controls in over
VSM_INTERVAL = 1      # steps between VSM aero solves

# =============================================================================
# Model
# =============================================================================

@info "V3 beam-wing example" STRUC_YAML AERO_YAML

beam_topology = V3BeamTopology(
    bridle_rel_damping = BRIDLE_DAMPING_PER_STIFFNESS,
    compression_frac = COMPRESSION_FRAC,
    compression_damping_frac = COMPRESSION_DAMPING_FRAC,
)

sam, sys = create_v3_model(V3SimConfig(
    struc_yaml_path = STRUC_YAML,
    aero_yaml_path = AERO_YAML,
    vsm_settings_path = VSM_SETTINGS,
    aero_mode = ContinuousAero(),
    backend = KernelBackend(),
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    elevation = ELEVATION,
    damping_pattern = BODY_DAMPING,
    world_damping_pattern = WORLD_DAMPING,
))

@info "Beam wing" bodies=length(sys.bodies) joints=length(sys.timoshenko_joints) points=length(sys.points) segments=length(sys.segments)

if !isnothing(ADAPTER_DIR) && isdir(ADAPTER_DIR)
    apply_comer_bending!(sys, ADAPTER_DIR, beam_topology)
    @info "Comer-Levy bending applied" joints=length(sys.timoshenko_joints)
else
    @info "Linear Breukels bending (set ADAPTER_DIR for Comer-Levy)"
end

apply_bridle_material!(sys, beam_topology)
@info("Bridle material", COMPRESSION_FRAC, COMPRESSION_DAMPING_FRAC,
    BRIDLE_DAMPING_PER_STIFFNESS)

geom = V3GeomAdjustConfig()
set_depower!(sys, DEPOWER, STEERING, geom)
set_steering!(sys, STEERING, geom)

init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
sys.winches[1].brake = true

state_path = joinpath(v3_data_path(),
    relaxed_state_name(STRUC_YAML, DEPOWER) * ".arrow")
start_from_state!(sam, sys, state_path) ||
    error("No relaxed state at $state_path; run relax_bridle.jl first")
@info "Started from the relaxed state" state_path

# =============================================================================
# Simulation loop
# =============================================================================

n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

@info "Starting simulation" n_steps dt
sim_start = time()

for step in 1:n_steps
    t = step * dt
    ramp = ramp_factor(t, 0.0, RAMP_TIME)

    # Depower is not ramped: the relaxed state sits at DEPOWER already, so
    # ramping from zero would yank the power tape.
    set_depower!(sys, DEPOWER, ramp * STEERING, geom)
    set_steering!(sys, ramp * STEERING, geom)

    if !sim_step!(sam; dt, vsm_interval = VSM_INTERVAL)
        @error "Simulation failed" step
        break
    end
    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start
        @info "Step $step/$n_steps" times_realtime=round(t/elapsed, digits=2)
    end
end

report_performance(SIM_TIME, time() - sim_start)

save_log(logger, "v3beam_example")
syslog = load_log("v3beam_example")

# =============================================================================
# Visualization
# =============================================================================

@info "Creating visualization..."
scene = SymbolicAWEModels.replay(syslog, sam.sys_struct)
display(GLMakie.Screen(), scene)

@info "Example complete!"
nothing
