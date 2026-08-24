# Plan: unify cache location to always use the scratchspace

## Background

V3Kite caches two kinds of generated artifacts — the compiled model binary
(`model_*.bin`) and the settled-geometry files (`settled_*.bin`,
`settle_particle_dynamics_wing.arrow`) — under a path chosen by
`default_cache_path(data_path)` in `src/stabilization.jl` (lines 623–651),
used whenever a caller does not pass `cache_path` explicitly to `init()` or
`settle_wing()`:

```julia
function default_cache_path(data_path)
    abs_v3_data = abspath(v3_data_path())
    in_depot = any(DEPOT_PATH) do depot
        startswith(abs_v3_data, joinpath(abspath(joinpath(depot, "packages")), ""))
    end
    in_depot || return data_path
    uuid = "4caac9c8-c726-438f-ab10-3553e918eab1"  # V3Kite, see Project.toml
    return joinpath(DEPOT_PATH[1], "scratchspaces", uuid, "v3kite_cache")
end
```

This resolves to one of two places, depending on **how V3Kite itself is
loaded** (not on the caller's own install status):

- **Pkg-installed V3Kite** → depot scratchspace,
  `~/.julia/scratchspaces/<V3Kite-uuid>/v3kite_cache`. A package directory
  under `DEPOT_PATH/packages` is read-only and gets swept by `Pkg.gc` once no
  environment references that version, so the scratchspace (keyed by UUID) is
  used instead, and survives reinstalls.
- **Dev'ed / local-checkout V3Kite** (`] dev` or a `[sources]` `path` entry,
  e.g. via SimpleKiteControllers.jl's `bin/dev`) → the checkout's own `data/`
  directory, unchanged from pre-scratchspace behaviour, so existing
  `data/settled_*.arrow` states stay in use across sessions.

`precompile.jl`'s `@compile_workload` also calls `init()` without
`cache_path`, so it writes its own warm-up artifacts to whichever of the two
locations `default_cache_path()` resolves to **at precompile time** — `data/`
for a dev checkout, the scratchspace for an installed one.

Consuming repos mirror this split. SimpleKiteControllers.jl's
`bin/delete_cache_files` cleans **both** locations: the scratchspace and,
when V3Kite is dev'ed (read from `examples/Project.toml`'s `[sources]`
entry), the checkout's `data/` too — because that is where the caches
actually land today.

## The ask

Simplify this to a single, always-scratchspace location, dropping the
dev-checkout `data/` branch.

## Two ways to read that

### Option A — cleanup script only

Change `bin/delete_cache_files` in SimpleKiteControllers.jl to search only
the scratchspace directory, leaving `default_cache_path()` and every `init()`
call site untouched.

**Effect:** for a dev'ed V3Kite (the current state — see
SimpleKiteControllers.jl's `CLAUDE.md`, "V3Kite is at v1.0.2, but sourced
from a local checkout"), runs keep writing/reading cache in the checkout's
`data/`, exactly as before. The cleanup script would simply stop finding
those files, so it silently stops doing part of its job: a stale
`model_*.bin` or `settled_*.bin` left behind after a breaking serialization
change (the `TypeError: ... SmallVec ...` failure mode the script's own
header comment documents) would no longer be swept, and the next run would
still hit the same deserialize crash the script exists to prevent.

**Verdict:** cheap, but defeats the script's purpose while V3Kite stays
dev'ed. Only sensible if paired with switching back to a Pkg-installed
V3Kite (`bin/free`) as a matter of course, or if the dev-checkout `data/`
cache is going to be cleaned by hand from now on.

### Option B — force scratchspace everywhere

Additionally change V3Kite so caching always goes to the scratchspace,
regardless of dev vs. installed — either by changing
`default_cache_path()` itself to drop the `in_depot` check, or by having
consuming repos (SimpleKiteControllers.jl's `examples/simple_fig8.jl`, etc.)
pass `cache_path` explicitly to every `init()` call.

**Effect, if `default_cache_path()` is changed:** every dev'ed checkout
(anyone's, not just this one) starts reading/writing a shared depot path
instead of its own `data/`. Two checkouts of V3Kite dev'ed side by side on
one machine would now collide on the same scratchspace cache — previously
isolated because each had its own `data/`. `precompile.jl`'s warm-up
artifacts, compiled *at package build time* against whatever
`default_cache_path()` resolves to then, would also move to scratchspace,
so this is at least internally consistent — the workload and the runtime
would agree on where to look.

**Effect, if only consuming repos pass `cache_path` explicitly:** the
precompile workload still writes to `data/` (unchanged, since it calls
`init()` internally without a caller-supplied `cache_path`), while runtime
calls from the consuming repo look in scratchspace instead. That is a
guaranteed first-run miss against the precompiled artifacts: scratchspace
starts empty, so the first `init()` pays a full settle + compile
(`precompile.jl`'s own comment: minutes) instead of the fast deserialize the
precompile workload exists to provide. SimpleKiteControllers.jl's own
`CLAUDE.md` documents this exact cost from a past incident: "the run
deserialized a model binary V3Kite's precompile workload had never compiled
against" cost ~34s of extra `init` time, root-caused and fixed by *dropping*
`cache_path` — this reintroduces the same failure mode deliberately.

**Verdict:** correct only if paired with also repointing
`precompile.jl` (i.e., changing `default_cache_path()` itself, not just
callers) — otherwise it reintroduces a bug this codebase already hit once
and documented. Even done correctly, it removes the per-checkout cache
isolation that lets two dev'ed working trees run independently, and every
existing `data/settled_*.arrow` / `data/model_*.bin` in every dev checkout
on the machine becomes orphaned dead weight (nothing reads it anymore; needs
its own manual cleanup once).

## Open questions before implementing either option

1. Is the goal "stop bin/delete_cache_files from missing files", "stop two
   dev checkouts from diverging", "make cache location predictable for
   tooling", or something else? The right option (and how big a change is
   justified) depends on which.
2. If Option B: is `default_cache_path()`'s dev-checkout branch considered
   still worth keeping for anyone (e.g. as an opt-in), or should it be
   deleted outright along with its rationale in the docstring and the
   "Default Cache Path" testset in `test/runtests.jl` (lines ~135–157)?
3. Who else dev's V3Kite locally and would be affected by losing
   per-checkout cache isolation?

## Recommendation

Do not implement Option B piecemeal (consuming-repo-only). If forcing
scratchspace is genuinely wanted, change `default_cache_path()` itself so
`precompile.jl` and every runtime caller agree, update the docstring and the
"Default Cache Path" testset in `test/runtests.jl` together, and clean out
now-orphaned `data/*.bin`/`data/*.arrow` in every affected dev checkout as
part of the same change. Otherwise, Option A alone is fine as a narrow fix to
`bin/delete_cache_files`, but only with the explicit understanding that it
stops cleaning the dev-checkout cache, not that the dev-checkout cache stops
existing.

# TODO (done)
Change `default_cache_path()` itself so
`precompile.jl` and every runtime caller agree, update the docstring and the
"Default Cache Path" testset in `test/runtests.jl` together, and clean out
now-orphaned `data/*.bin`/`data/*.arrow` in every affected dev checkout as
part of the same change.
