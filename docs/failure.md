# Local test failures on `bump_vsm`: `ContinuousAero` diverges at t = 0

**Branch:** `bump_vsm` (`038f8a9`) · **Investigated:** 2026-09-03
**Status: unresolved — reproduces locally, passes on CI.**

Local `Pkg.test()`:

```
Test Summary:                     | Pass  Fail  Error  Total
V3Kite.jl                         |  387     1      3    391
  Delete Cache Files Then Parking |    1            1      2
  Interface Functions / lift_drag |    5     1             6
  init damping_per_stiffness      |    1            1      2
  Parking AoA Ripple              |                 1      1
```

CI run 33782052364 on the same commit: **success**.

## Confirmed: one bug, not four

`ContinuousAero` diverges on the very first integrator step:

```
At t=0.0, dt was forced below floating point epsilon 5.0e-324. Aborting.
There is either an error in your model specification or the true solution is
unstable (or it cannot be represented in Float64 precision).
```

Isolated against `system_cabauw.yaml`. `damping_per_stiffness` is irrelevant —
the aero mode is the whole story:

| config                                          | result           |
| ----------------------------------------------- | ---------------- |
| `ContinuousAero`, no `damping_per_stiffness`     | Settling failed  |
| `ContinuousAero`, `damping_per_stiffness=0.001`  | Settling failed  |
| `AeroDirect`, `damping_per_stiffness=0.001`      | OK               |

Every failing test runs `continuous`:

- **Delete Cache Files Then Parking** and **Parking AoA Ripple** pass
  `AERO_MODE = ContinuousAero()`. Settling diverges on step 1 →
  `error("Settling diverged before completing 40 steps")` in
  `src/stabilization.jl` → `settle_failed` → `error("Settling failed")` at
  `src/interface.jl:535`. The 3-minute vs 0.6-second split between the two is
  only cold vs warm model cache; both die at t = 0.
- **init damping_per_stiffness** passes no `aero_mode`, and
  `data/kite_settings_cabauw.yaml` defaults to `continuous`. Same throw. The
  apparent `damping_per_stiffness = 0.001` correlation across the three errors
  is a red herring — that value is simply what the two parking tests also use.
- **lift_drag** uses `system_psm.yaml`, and `data/kite_settings_psm.yaml:5` is
  `aero_mode: continuous`, so `@test sim_step!(...)` at
  `test/test-interface.jl:31` returns `false` for the same reason. The five
  tests after it still pass because they read the un-stepped initial state.

## Ruled out: the VSM bump itself

The first read of this was "VSM 4.2/4.3 broke `ContinuousAero`". **That is
wrong.** CI resolved exactly the same graph and passed:

| package                   | local   | CI      |
| ------------------------- | ------- | ------- |
| VortexStepMethod          | 4.3.1   | 4.3.1   |
| SymbolicAWEModels         | 0.15.1  | 0.15.1  |
| ModelingToolkit           | 11.38.2 | 11.38.2 |
| SciMLBase                 | 3.34.0  | 3.34.0  |
| NonlinearSolve            | 4.21.1  | 4.21.1  |
| OrdinaryDiffEqCore        | 4.6.0   | 4.6.0   |
| KiteUtils                 | 0.12.2  | 0.12.2  |
| Julia                     | 1.12.7  | 1.12.7  |

CI ran the ContinuousAero parking test to completion: *Parking AoA ripple RMS:
0.00226° (baseline 0.0064°, limit 0.0096°)*. Note that RMS is ~3x below the
baseline taken under VSM 4.1.2, so 4.3.1 does move the physics — but in the
stable direction, and with wide margin.

## Ruled out: a stale model binary surviving the cache wipe

Plausible, but the timestamps say no. `bin/delete_cache_files` ran at the start
of the suite (~19:25); the scratchspace
`model_..._cont_..._kernel.bin` is dated **19:28**, i.e. rebuilt from scratch
under VSM 4.3.1 during the failing run. The run that diverged was not reading
an old binary.

Also ruled out: turbulence (`use_turbulence: 0.0` in every `sim_settings_*.yaml`,
`gui.yaml` set to `"default"`), and VSM's `use_prior_polar` (in-memory only, not
a disk cache; the 2D polars are tracked in git).

## Still open

The only difference found in the entire dependency graph:

- **SymbolicIndexingInterface 0.3.54 (local) vs 0.3.55 (CI).** Untested. It is
  the next thing to try.

Other untested local-vs-CI differences: CPU/BLAS, and CI running with
`--check-bounds=yes`.

## Genuine defects found along the way (independent of the above)

1. **`test_delete_cache_parking.jl` is vacuous on CI.** The runner starts with an
   empty scratchspace, so `bin/delete_cache_files` finds nothing, prints
   "No V3Kite cache files found.", and exits 0 — `@test success(script)` passes
   without a deletion ever happening. The regression the test exists to catch
   (a cache outliving the versions it was written against) is only ever
   exercised on a developer machine. Seeding a cache before the wipe would make
   the test mean something on CI.
2. **`bin/delete_cache_files` misses the `data/` folder.** A
   `model_v0.15.1_..._kernel.bin` and a `settle_particle_dynamics_wing.arrow`
   also accumulate in `data/` (both gitignored; the model bin was written there
   at 19:24, distinct in content from the scratchspace copy). Some code path is
   still caching to `data_path` rather than to `default_cache_path`, and the
   script cleans only the scratchspace. Not the cause of this failure, but it
   leaves exactly the kind of stale artifact the script's own header warns about.

## Reproducing

```julia
using V3Kite
using KiteUtils: set_data_path
set_data_path(v3_data_path())

# throws "Settling failed"
init(10.0, 150.0; depower_setpoint=0.25, sim_time=1.0,
     system_yaml="system_cabauw.yaml", aero_mode=ContinuousAero())

# succeeds
init(10.0, 150.0; depower_setpoint=0.25, sim_time=1.0,
     system_yaml="system_cabauw.yaml", aero_mode=AeroDirect())
```
