# Reduce time-to-start-simulation (TTFX)

## Goal

Reduce the time from starting Julia to the first simulation step, as measured by
`tic()`/`toc()` in `examples/simple_sinus.jl`. The currently observed value is
**28 s** (cache state and sysimage state of that measurement: *to be pinned down
in step 0*).

The decisive question is **not** "does a PrecompileTools workload beat nothing".
The everyday workflow already uses a system image. The decisive question is:

> **Does a PrecompileTools workload inside V3Kite measurably reduce TTFX
> *on top of* `bin/kps-image-<version>.so`?**

Everything below is organized around that comparison.

## The system image does *not* contain V3Kite — this is what makes the question live

From the header of `bin/create_sys_image`:

> It includes most of the dependencies of the examples, **but not V3Kite.jl
> itself**. This means you can still use Revise for the main package while
> benefiting from faster startup of the examples.

Concretely:

- `test/create_sys_image.jl` bakes in ~30 dependencies (ModelingToolkit,
  OrdinaryDiffEqBDF, NonlinearSolve, SymbolicAWEModels, VortexStepMethod,
  GLMakie, CSV/DataFrames/HDF5, …). **`V3Kite` is deliberately absent from that
  list.**
- Its `precompile_execution_file` is `test/test_for_precompile.jl`, which runs
  `examples/v3kite.jl` — a full simulation. So *dependency-owned* methods
  specialized on V3Kite's types **are** traced and baked in.
- V3Kite's *own* methods are not. They are compiled into V3Kite's ordinary
  pkgimage, which `bin/create_sys_image` step 5 builds by running
  `julia --project -J <sysimage> -e 'using V3Kite'`.

So sysimage and PrecompileTools are **complementary, not redundant**. The
addressable delta for a workload is: V3Kite's own methods, plus any external
`CodeInstance`s the sysimage's execution trace missed. That delta may be small —
V3Kite is a fairly thin orchestration layer over SymbolicAWEModels — but it is
not zero by construction, and its size is an empirical question.

**Cheap iteration loop (important):** because V3Kite is not in the sysimage,
testing a workload does **not** require the ~30 min / ~64 GB sysimage rebuild.
It only requires re-precompiling V3Kite against the existing image:

```bash
touch src/*.jl
julia --startup-file=no --project -J bin/kps-image-1.12.so -e 'using V3Kite'
```

## Prior art in this repo (do not re-run a failed experiment)

- `PrecompileTools` is already a dependency (`Project.toml`).
- `src/precompile.jl` already exists, with its **entire workload commented out**.
- `include("precompile.jl")` is disabled in `src/V3Kite.jl` (last line before
  `end # module`), marked `# disabled: precompilation workload`.
- Introduced by commit `81a1043 "Add PrecompileTools workload and fix example bugs"`.

**Step 1 below must establish why it was disabled.** If it was disabled because
it gave no measurable benefit on top of the sysimage, this plan stops there.

## Other baseline facts that constrain the work

- **The 28 s is a warm-cache number.** `data/*.bin` are **not** tracked in git:
  - `data/model_v0.11.1_jl1.12_….bin` — the built model, keyed by Julia version
    and SymbolicAWEModels version.
  - `data/settled_*.bin` — settled geometry (`src/stabilization.jl`, key encodes
    tape lengths, wind, tether length, gravity, system yaml).

  On a fresh clone both are absent and `init()` rebuilds them, which is far
  slower than 28 s. Every timing reported below must state its cache state.
- **MTK sets a hard ceiling.** Model evaluation goes through
  `RuntimeGeneratedFunction`s generated at *run* time. Those cannot be baked into
  a `.ji`/pkgimage *or* into a sysimage. Whatever share of the 28 s they
  represent is addressable by neither approach.

---

## Step 0 — Measure the breakdown (blocking; no code changes)

### 0a. Four configurations

| Config | Sysimage | Workload | Represents |
| --- | --- | --- | --- |
| **A** | no | no | plain `julia --project`, CI, package users |
| **B** | no | yes | what a workload buys users without a sysimage |
| **C** | yes | no | **today's actual workflow** (`bin/run_julia`) |
| **D** | yes | yes | **the proposal** |

- **Primary metric: D − C.** This decides accept/reject.
- Secondary metric: B − A. Relevant only for CI and for users who never build a
  sysimage; it must not be used to justify the change on its own.
- A and C also answer which configuration the original "28 s" came from.

Configs B and D cannot be measured until step 3 exists. Step 0 delivers A and C
plus the segment breakdown; the B/D numbers are filled in at step 4.

### 0b. Segment breakdown (measure A and C now)

The 28 s spans `using V3Kite` **and** `init(...)`. These need different fixes and
cannot be optimized as one number. Fresh Julia process, `--project=examples`,
warm `data/*.bin`, each segment timed separately:

| Segment | Already covered by sysimage (C)? | Addressable by workload (D)? |
| --- | --- | --- |
| `using` of deps (MTK, SymbolicAWEModels, VSM, GLMakie, CSV/HDF5) | Yes | No — load time, not compile time |
| `using V3Kite` itself (V3Kite pkgimage load) | No | Marginally |
| `init(...)`: V3Kite's own code (`interface.jl`, `stabilization.jl`) | No | **Yes — the main target** |
| `init(...)`: dep methods on V3Kite types | Mostly (traced via `examples/v3kite.jl`) | Redundant |
| `init(...)`: MTK `RuntimeGeneratedFunction` compilation | **No** | **No** |
| first `step!` / `next_step!` solve specialization | Mostly | Partly |

### 0c. Also record

- `julia --trace-compile=stderr` for config **C**, bucketed into (i) methods
  owned by V3Kite, (ii) methods owned by dependencies, (iii)
  `RuntimeGeneratedFunction`s. **Bucket (i) is the upper bound on what D can
  win.** If bucket (i) is a small fraction of the remaining time, stop here.
- Baseline `Pkg.precompile` wall-clock for V3Kite against the sysimage (the cost
  side of the ledger).

**Exit criterion:** the tables above filled in with real numbers and committed to
this document. The choice in step 2 follows from them, not from assumption.

## Step 1 — Recover the history of the disabled workload

Determine from git history / the author why `src/precompile.jl` was commented
out. Record the reason here. Note in particular that the commented-out setup
called `create_v3_model(config, remake_cache=true)`, i.e. it **wrote into
`data/` during precompilation** — non-hermetic, potentially into a read-only
depot, and very slow. Any new workload must not do this.

## Step 2 — Choose the approach (gated on step 0)

- **If trace-compile bucket (i) in config C is large** → a PrecompileTools
  workload is justified; go to step 3.
- **If bucket (i) is small but `using` of deps dominates config A** → the
  workload is not the lever. The higher-leverage change for non-sysimage users
  is dependency surgery: `CSV`, `DataFrames` and `HDF5` are used only by
  `src/flight_data.jl` and are irrelevant to a simulation run, so they can move
  behind a weak-dep extension (the `V3KiteMakieExt` pattern is already in place).
- **If the residual in config C is mostly RGF compilation** → neither approach
  helps. The fix belongs upstream in SymbolicAWEModels/ModelingToolkitBase.
  Document and stop.
- **If config C is already close to the floor** → consider instead adding
  `V3Kite` to the sysimage package list in `test/create_sys_image.jl`, and weigh
  that against losing Revise on the main package (which is the stated reason it
  was excluded). This is a workflow trade-off for the maintainer to decide, not a
  pure performance question.

## Step 3 — Workload implementation (only if step 2 selects it)

Constraints on the workload in `src/precompile.jl`:

- **Hermetic.** No writes to `data/`, no `remake_cache=true`, no cache
  regeneration. If the workload needs a model, it must use a temporary directory
  or a fixture committed to the repo.
- **Bounded.** Exercise the smallest call path that covers V3Kite's own hot
  types — the helpers in `sim_helpers.jl`/`interface.jl` — rather than a full
  `init()`. Anything that only re-triggers dependency-owned code is wasted work
  in config D, since the sysimage already has it.
- **Loud in CI, quiet for users.** A workload that silently `catch`es its way to
  a no-op (as the previous draft did) produces zero benefit and no signal. CI
  must fail if the workload does not run.
- Re-enable `include("precompile.jl")` in `src/V3Kite.jl` in the same change.

## Step 4 — Accept / reject

Fill in the full matrix:

| | TTFX (warm cache) | `Pkg.precompile` V3Kite |
| --- | --- | --- |
| A — no sysimage, no workload | | — |
| B — no sysimage, workload | | |
| C — sysimage only | | — |
| D — sysimage + workload | | |

**Accept only if D is at least 5 s faster than C**, *and* the added
V3Kite precompile time is under ~90 s. A win that only shows up as B − A does
**not** justify the change for this repo's workflow — record it, but reject
unless D − C also clears the bar. A 15 s TTFX win costing 4 minutes of
precompilation on every dependency bump is a bad trade — revert instead.
