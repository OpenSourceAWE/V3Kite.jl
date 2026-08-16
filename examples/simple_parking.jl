# Copyright (c) 2025 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Example script demonstrating a parking maneuver of the V3 kite using the
high-level `init` + `step!` interface.

The wing is settled at a fixed depower setting (DEPOWER_SETPOINT) and then
parked at a constant tether length: `step!` runs in POSITION MODE, with
the caller's cascaded winch length loop holding the initial tether length
controller, so the kite parks without any net reel-out.

The manual, braked-winch reference `examples/parking.jl` is not modified; this
script is its `init`/`step!` counterpart. Logs the run to "tmp_parking" in the
`output` folder (`examples/../output`, created if missing).

The largest compression force over all tether and bridle segments is logged
per step into the free `var_01` slot (see `log_max_compression!`) and its peak
over the run is printed at the end; `simple_parking_plots.jl` plots it. Every
segment whose peak exceeds COMPRESSION_LIMIT is listed by
`print_compression_report`, which needs the per-segment peaks and so has no
counterpart in the plots script — the log only carries the maximum.

At the end it prints the AoA ripple metrics (see `src/ripple_metrics.jl` and
PlanSuppressOscillations.md) together with the solver cost and the wall clock, so
a change to e.g. the body-frame damping can be judged on both the oscillation and
the simulation speed. The numbers are only comparable across runs that fix
PROJECT, V_WIND, TETHER_LENGTH, DEPOWER_SETPOINT, REL_STEERING, SIM_TIME, DT and
the damping constants.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: init, step!
import KiteUtils   # for KiteUtils.syslog; V3Kite does not re-export it, and the
                   # plots scripts bind the bare name `syslog` to a SysLog value
using LinearAlgebra: norm
using Printf: @printf, @sprintf
include(joinpath(@__DIR__, "winch_adapter.jl"))  # winch_torque!: V3Kite is torque-only

@info "simple_parking.jl: parking the V3 kite via the init/step! interface."

# ==================== USER PARAMETERS ==================== #

PROJECT =        "system_reelout.yaml"  # System project to use (see data/system_*.yaml)
SIM_TIME         = 10.0     # Total simulation time [s]
DT               = 0.05/3     # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
USE_BRAKE        = true     # use the brake of the winch (or not)
DEPOWER_SETPOINT = 0.25     # Depower setting held during parking [-]
REL_STEERING     = 0.0040   # Fixed steering trim, tuned so |heading(end)| < 10 degrees
AERO_MODE        = ContinuousAero() # ContinuousAero() or AeroDirect()
VSM_INTERVAL     = 1   # steps between VSM aero solves
# `INITIAL_BODY_DAMPING` shapes the settling transient; it decays to
# `FLOWN_BODY_DAMPING`, the floor the parked run actually flies with.
INITIAL_BODY_DAMPING = [0.0, 0.0, 40.0]             # Damping settling starts from, per axis [1/s]
FLOWN_BODY_DAMPING   = 0.8 .* INITIAL_BODY_DAMPING  # Damping floor the parked run flies with, per axis [1/s]
# Tether/bridle damping-to-stiffness ratio, overriding the `dyneema` material
# default in `data/struc_geometry.yaml`. `init` floors it during settling
# (see `stabilization.jl`) then applies the raw value to the settled structure.
DAMPING_PER_STIFFNESS = 0.001  # Damping per stiffness of tether and bridles [s]
COMPRESSION_LIMIT = 10.0  # segments whose peak compression exceeds this are reported [N]

# ======================== INIT =========================== #

# `init` leaves the data path alone, so `save_log` below needs it set here.
set_data_path(v3_data_path())
s = init(V_WIND, TETHER_LENGTH; body_damping = INITIAL_BODY_DAMPING,
    min_damping = FLOWN_BODY_DAMPING, damping_per_stiffness = DAMPING_PER_STIFFNESS,
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
    system_yaml = PROJECT, aero_mode = AERO_MODE)

# `init` leaves the winch un-braked; brake it to park at constant tether length.
s.sys.winches[1].brake = USE_BRAKE

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]
# The winch length loop is the CALLER's now (V3Kite's `step!` takes a torque).
wcs = load_wc_settings("wc_settings.yaml"; dt = s.dt)
wpc = WinchPosController(wcs; dt = s.dt)

# ============== TETHER AND BRIDLE SEGMENTS =============== #

# The segments the compression report below is taken over — the same ones `init`
# applied `DAMPING_PER_STIFFNESS` to. `tether_bridle_segments` comes from V3Kite
# (`src/model_setup.jl`) and takes the `SystemStructure`, not the `V3KITE` model.
line_segs = tether_bridle_segments(s.sam.sys_struct)

# ================= COMPRESSION LOGGING =================== #

"""
    log_max_compression!(s, seg_idxs, peaks) -> (force, seg_idx)

Largest compression force [N] over the segments `seg_idxs`, reported as a
positive number, with `(0.0, 0)` while all of them are in tension. Also raises
`peaks[i]`, the running peak of segment `seg_idxs[i]`. `segment.force` is
signed — negative means compression — and is refreshed from the integrator by
every `step!`, so the value belongs to the row `step!` has just logged and is
written into its free `var_01` slot.
"""
function log_max_compression!(s, seg_idxs, peaks)
    segments = s.sam.sys_struct.segments
    force, seg_idx = 0.0, 0
    for (i, idx) in enumerate(seg_idxs)
        f = -segments[idx].force
        f > peaks[i] && (peaks[i] = f)
        if f > force
            force, seg_idx = f, idx
        end
    end
    s.sys_state.var_01 = force
    s.logger.var_01_vec[max(s.logger.index - 1, 1)] = force
    return force, seg_idx
end

"""
    print_compression_report(s, seg_idxs, peaks, limit)

Print the tether and bridle segments whose peak compression force over the run
exceeded `limit` [N], strongest first, with their endpoint points and the
`compression_frac` that scales their stiffness below the rest length.
"""
function print_compression_report(s, seg_idxs, peaks, limit)
    segments = s.sam.sys_struct.segments
    sel = [i for i in sortperm(peaks; rev = true) if peaks[i] > limit]
    if isempty(sel)
        println("\nNo tether or bridle segment exceeded ",
                @sprintf("%.1f N", limit), " compression.")
        return
    end
    @printf("\ncompression > %.1f N: %d of %d tether and bridle segments\n",
            limit, length(sel), length(seg_idxs))
    @printf("  %-9s %-11s %-16s %s\n", "segment", "points", "compression_frac",
            "peak [N]")
    for i in sel
        seg = segments[seg_idxs[i]]
        p1, p2 = seg.point_idxs
        @printf("  %-9d %-11s %-16.3f %8.2f\n", seg.idx, "$p1 - $p2",
                seg.compression_frac, peaks[i])
    end
end

# ==================== SIMULATION LOOP ==================== #

steps_done = 0
F_compr_max, F_compr_seg, F_compr_t = 0.0, 0, 0.0
compr_peaks = zeros(length(line_segs))
t_loop = @elapsed try
    for _ in 1:s.steps
        # Position mode: hold the mean tether length at its initial value.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = REL_STEERING,
              set_torque = winch_torque!(wpc, s, l0), vsm_interval = VSM_INTERVAL)
        global steps_done += 1
        # The current system state is available via `s.sys_state`.
        F_c, seg_idx = log_max_compression!(s, line_segs, compr_peaks)
        if F_c > F_compr_max
            global F_compr_max, F_compr_seg, F_compr_t = F_c, seg_idx, s.sys_state.time
        end
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
mkpath(OUTPUT_DIR)
save_log(s.logger, "tmp_parking"; path=OUTPUT_DIR, colmeta=timestamp_colmeta())

@info "Wind speed at kite height: $(round(norm(v_wind_kite(s)), digits=2)) m/s"

if F_compr_max > 0
    @info "Peak compression: $(round(F_compr_max, digits=2)) N in segment $F_compr_seg " *
          "at t=$(round(F_compr_t, digits=2)) s"
else
    @info "No compression: all tether and bridle segments stayed in tension."
end
print_compression_report(s, line_segs, compr_peaks, COMPRESSION_LIMIT)

# ==================== RIPPLE METRICS ===================== #

sl = KiteUtils.syslog(s.logger)
ripple = aoa_ripple(sl)
print("\n", format_ripple_report(ripple; sl, stats = s.sam.integrator.stats,
                                 t_loop, n_steps = steps_done))

nothing
