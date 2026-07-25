# Reduce time-to-start-simulation (TTFX)

**Status: DONE. Workload implemented and measured — TTFX 47.4 s → 4.0 s.**

An earlier revision of this document rejected the workload on the basis of a
trace-compile analysis. That verdict was wrong by a factor of ~40; the analysis
error is recorded below so it is not repeated.

Measurements: 2026-07-25, Julia 1.12.6, `--project` (repo root), warm
`data/*.bin`, `examples/simple_sinus.jl` parameters (`v_wind=10`,
`l_tether=150`, `depower_setpoint=0.25`, `system_cabauw.yaml`).

## Measured baseline

| Config | Sysimage | Workload | `using V3Kite` | `init(...)` | 1st `step!` | **TTFX to end of init** |
| --- | --- | --- | --- | --- | --- | --- |
| **A** | no | no | 6.52 s | 104.27 s | 12.73 s | **110.8 s** |
| **C** | yes | no | 0.02 s | 47.4 s | 9.9 s | **47.4 s** |
| **D** | yes | **yes** | 0.25 s | 3.76 s | **0.04 s** | **4.0 s** |

Both are reproducible across three runs — C: 47.39 / 47.82 / 46.54 s;
D: 4.01 / 4.01 / 3.97 s. Steady-state `step!` is ~17 ms in both.
Config C is the everyday workflow (`bin/run_julia`), so **D − C is the metric
that matters**:

| Metric | C | D | Gain |
| --- | --- | --- | --- |
| TTFX to end of `init` | 47.4 s | **4.0 s** | **−43.4 s** |
| TTFX incl. first `step!` | 57.3 s | **4.0 s** | **−53.3 s** |
| of which: model init | 35.0 s | 2.5 s | −32.5 s |
| of which: first `step!` | 9.9 s | 0.04 s | −9.9 s |
| `using V3Kite` | 0.02 s | 0.25 s | +0.23 s (bigger pkgimage) |
| `Pkg.precompile` V3Kite | ~7 s | 75 s | +68 s (budget was ~90 s) |

Far past the ≥ 20 s target. The first `step!` essentially disappears (9.9 s →
0.04 s), which the workload's three `step!` calls cover directly.

Verified that precompilation does **not** write to `data/` — mtimes on
`model_*.bin` and `settled_*.bin` are unchanged across the precompile run, since
the workload inherits `init`'s `remake=false`.

Cost is one-time per process — calling `init()` twice in one process gives
46.54 s then **0.89 s** — i.e. ~45.6 s of `init` is compilation, not I/O and not
algorithm. That is the pool a workload draws from.

### The "28 s" does not reproduce

Config C measures **47.4 s** in a fresh process, consistently. The 28 s figure
was most likely timed by `include`ing `simple_sinus.jl` into an already-warm
REPL. Worth confirming before 28 s anchors a target.

## Proof that this works: HybridWings.jl

`~/repos/HybridWings.jl` has the same architecture — same `bin/create_sys_image`
pattern, same deliberate exclusion of the package itself from the image — and a
`@compile_workload` there **improved TTFX by 22 s**. Its own comment states the
mechanism precisely (`src/HybridWings.jl:79-85`):

> The sysimage built by bin/create_sys_image deliberately excludes
> HybridWings.jl itself (so Revise keeps working on it), so none of this is
> otherwise precompiled.

Structure of that workload (`src/HybridWings.jl:86-183`):

1. Build settings + `SystemStructure` from the bundled YAML.
2. `SymbolicAWEModel(set, sys_struct)` then `init!(sam; remake=false, remake_vsm=true)`.
3. Construct the high-level wrapper (`HybridWing`), a `Logger`, a `SysState`,
   and the `DiscretePID`s.
4. **Run 3 full control steps**, each calling `lift_drag`, `total_drag`,
   `calc_steady_torque`, `next_step!`, `update_sys_state!`, `log!`.
5. The whole thing wrapped in `try`/`catch` with a `@warn`, so a workload failure
   cannot break `using`.

Note it uses `remake=false` — it consumes the existing serialized model rather
than rebuilding one, so it is hermetic with respect to `data/`.

## Where the earlier rejection went wrong

The rejection rested on a trace-compile bucketing of config C: 300 events, 257
"involving `RuntimeGeneratedFunctions`", only 13 "mentioning V3Kite". Two errors:

1. **Grepping for `V3Kite` measured the wrong thing.** PrecompileTools caches
   compiled code for *dependency-owned* methods invoked from the workload
   (external `CodeInstance`s land in V3Kite's pkgimage). The addressable surface
   is most of the 300 events, not the 13 that happen to have "V3Kite" in the
   type signature.
2. **"Involves RGF" ≠ "uncacheable".** The generated function *bodies* are built
   at run time, true. But the specializations of `generated_callfunc` and of
   everything downstream are keyed on the RGF type tag, and that tag is
   deterministic given the same serialized model — so those specializations
   *are* cacheable. HybridWings demonstrates this empirically.

There is also a V3Kite-specific reason the gap should be large. The sysimage's
`precompile_execution_file` is `test/test_for_precompile.jl`, which runs
`examples/v3kite.jl` — and that example calls `settle_wing` and `sim_step!`
**directly**. It never calls `init()` or `step!(::V3KITE, …)`. So the entire
`init`/`step!` interface path used by `simple_sinus.jl` — the `V3KITE` wrapper,
KCU actuator dynamics, `winch_position_torque!`, `update_sys_state!` — is absent
from the sysimage trace. That is precisely what the workload covers — and it is
why the gain here (43 s) exceeds the 22 s seen in HybridWings.

## What was implemented

`src/precompile.jl`, and `include("precompile.jl")` re-enabled in `src/V3Kite.jl`
(last line before `end # module`). The workload:

- calls `init(10.0, 150.0; depower_setpoint=0.25, sim_time=1.0,
  system_yaml="system_cabauw.yaml")` — `sim_time` only sizes the logger, so keep
  it small;
- runs 3 `step!` calls with `rel_depower`/`rel_steering`/`set_length`, covering
  the KCU → geometry → winch-control → `sim_step!` → logging chain;
- calls the query functions the examples use (`lift_drag`, `total_drag`,
  `pos_kite`, `winch_force`, `reel_out_speed`, `cl_cd`, `calc_elevation`,
  `calc_azimuth`, `calc_heading`);
- is wrapped in `try`/`catch` with a `@warn`, as HybridWings does, so a workload
  failure can never break `using V3Kite`.

It inherits `init`'s `remake=false`, so the serialized settled geometry and model
in `data/` are consumed, never rebuilt — no `remake_cache=true` and no writes to
`data/` (the abandoned 2025 draft did write there, which is the likely reason it
was disabled).

**The iteration loop is cheap** — because V3Kite is not in the sysimage, testing
a workload needs no ~30 min / ~64 GB image rebuild, only:

```bash
touch src/*.jl
julia --startup-file=no --project -J bin/kps-image-1.12.so -e 'using V3Kite'
```

### Caveat: the workload depends on the `data/*.bin` caches

If `model_*.bin` or the matching `settled_*.bin` is absent — e.g. after a
SymbolicAWEModels bump changes the model filename — the workload will *build*
them during precompilation. That makes precompilation slow (minutes) and writes
into `data/`, which would fail outright on a read-only install. HybridWings
accepts the same trade-off. If this becomes a problem, guard the workload on
`isfile` of the expected cache; the cost of guarding is that a fresh clone or CI
gets no benefit until the caches exist.

## Also found: the model cache is not portable across sysimages

Independent of TTFX, and worth fixing on its own.
`data/model_v0.11.1_jl1.12_….bin` fails to deserialize when the process switches
between the stock sysimage and `bin/kps-image-1.12.so`:

```text
Failure to deserialize .../model_..._1wch.bin: TypeError
TypeError: in typeassert,
  expected OrderedCollections.OrderedDict{...BasicSymbolicImpl{SymReal}, Number},
  got a value of type Dict{...BasicSymbolicImpl{SymReal}, Number}
```

(`SymbolicUtils/src/hashconsing.jl:515` via `Moshi`, from
`SymbolicAWEModels.load_serialized_model!`.)

On failure the model is rebuilt (~90–130 s) and **re-serialized**, so the two
configurations ping-pong the cache — every run under one image invalidates it
for the other. Observed: 187 s and 102 s for the two cold `init` calls.

A `Dict` is being written where an `OrderedDict` is expected on read;
serialization type identity should not depend on the sysimage. Likely upstream in
SymbolicAWEModels/SymbolicUtils. **This now also interacts with the workload:**
if precompilation runs under one image and the cache was written under the other,
precompilation will rebuild and re-serialize the model — slow, and it writes to
`data/`. Measure C and D back-to-back under the *same* image.

## Reproducing

```bash
# config C (repo sysimage) — the reference
julia --startup-file=no --project -J bin/kps-image-1.12.so scratch/ttfx_breakdown.jl
# config A (stock sysimage) — note this invalidates the model cache written by C
julia --startup-file=no --project scratch/ttfx_breakdown.jl
# trace bucketing
julia --startup-file=no --project -J bin/kps-image-1.12.so \
      --trace-compile=trace_C.txt scratch/ttfx_breakdown.jl
```

The measurement script times `using V3Kite`, `init(...)`, the first `step!`, and
100 subsequent `step!`s separately. It currently lives in the session scratchpad
and should move into the repo (e.g. `test/ttfx_breakdown.jl`) so these numbers
can be tracked across dependency bumps.
