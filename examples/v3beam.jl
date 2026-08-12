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
KCU, the pulleys, the M-line and the three tapes are the flown ones. Its `power_tape`
rest length is 2.2 m, which is `DEPOWER = 0.40`, so the emitted geometry already sits
at this example's setting.

Built on the `KernelBackend`, which assembles one kernel per component instead of
compiling a single monolithic ODE — the 22 beam bodies and 21 joints make the
monolithic build the dominant cost here.

`BODY_DAMPING` and `WORLD_DAMPING` are both set on every point. Body-frame damping
damps a node against its wing frame, which only `DYNAMIC` points get, so on this
geometry — whose wing nodes are `BODY_STATIC` points riding bodies — it reaches
nothing and only the bridle-carrying world-frame term acts. The wing itself is
damped by the Timoshenko joints and the canopy segments, whose damping lives in
the YAML.

`BRIDLE_SEGMENTS` splits every bridle line into that many spring-damper segments, a
pulley's two legs included; only the three KCU tapes stay single, their length being
driven directly. It is the one knob that changes how many segments exist, so it only
takes effect through `REGENERATE_STRUC_YAML`, which rewrites `STRUC_YAML` from the
export before the run, and it recompiles the kernels on its first run at a new value.

The other three decide what a line does as it goes slack and are set on the loaded
structure by [`apply_bridle_material!`](@ref), so sweeping them needs neither a
rewrite nor a recompile. `COMPRESSION_DAMPING_FRAC = COMPRESSION_FRAC` leaves the
bridle with a single damping ratio instead of one that jumps the moment a line
unloads, and both at `0` make a slack line carry no force at all. The two compression
fractions reach the canopy membrane segments as well as the bridle, since fabric is
tension-only for the same reason a line is, while `BRIDLE_DAMPING_PER_STIFFNESS`
reaches only the lines and tapes.

The YAML carries constant Breukels bending per joint because it cannot hold a
callable. Pointing `ADAPTER_DIR` at the export swaps in the curvature-softening
Comer-Levy law, which is what lets a tube keep bending past its collapse moment.
That needs the tube laws from an unreleased SymbolicAWEModels, so it is off here.
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

SURFPLAN_DIR = joinpath(homedir(), "Code", "Kite", "SurfplanAdapter",
    "processed_data", "TUDELFT_V3_KITE")

# Rewrite STRUC_YAML from the SurfplanAdapter export in SURFPLAN_DIR before
# running. Only needed for BRIDLE_SEGMENTS; the other three knobs are applied to
# the loaded structure below.
REGENERATE_STRUC_YAML = true

BRIDLE_SEGMENTS = 1                 # spring-damper segments per bridle line
BRIDLE_DAMPING_PER_STIFFNESS = 0.001 # bridle unit_damping / unit_stiffness [s]
COMPRESSION_FRAC = 0.01              # stiffness left under compression
COMPRESSION_DAMPING_FRAC = 1.0      # damping left under compression

# Export the beam was built from; `nothing` keeps the linear Breukels bending,
# SURFPLAN_DIR swaps in the curvature-softening Comer-Levy law. That needs the
# tube laws, which are not in a released SymbolicAWEModels yet.
ADAPTER_DIR = nothing

# BODY_DAMPING reaches nothing on a beam wing, see the docstring above.
BODY_DAMPING = [0.0, 0.0, 20.0]
WORLD_DAMPING = [0.0, 0.0, 0.0]

SIM_TIME = 10.0
FPS = 20
V_WIND = 15.4
TETHER_LENGTH = 250.0
ELEVATION = 70.0      # degrees
DEPOWER = 0.20        # fraction [0, 1]; the emitted power tape is 2.2 m = 0.40
STEERING = 0.0        # fraction [-1, 1]
RAMP_TIME = 5.0       # seconds to ramp the controls in over
VSM_INTERVAL = 1      # steps between VSM aero solves

# =============================================================================
# Model
# =============================================================================

@info "V3 beam-wing example" STRUC_YAML AERO_YAML

beam_topology = V3BeamTopology(
    bridle_segments = BRIDLE_SEGMENTS,
    bridle_rel_damping = BRIDLE_DAMPING_PER_STIFFNESS,
    compression_frac = COMPRESSION_FRAC,
    compression_damping_frac = COMPRESSION_DAMPING_FRAC,
    tether_length = TETHER_LENGTH,
    elevation_deg = ELEVATION,
)

if REGENERATE_STRUC_YAML
    counts = surfplan_to_struc(SURFPLAN_DIR,
        joinpath(v3_data_path(), STRUC_YAML);
        topo = beam_topology, wing_only = false)
    @info "Regenerated geometry" STRUC_YAML counts
end

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

# The measured bridle lengths and the measured node coordinates come from two
# different upstream files and disagree — several lines start above 100 % strain,
# which no implicit solver can take a first step from. Relax before simulating.
scale, relax_steps, residual = relax_bridle!(sam, sys)
scale < 1.0 && @error "Bridle relaxation did not reach full stiffness" scale residual

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

    # Depower is not ramped: the emitted bridle already sits at DEPOWER and the
    # relaxation settled it there, so ramping from zero would yank the power tape.
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
