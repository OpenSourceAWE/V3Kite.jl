# Fix turbulence: inject it through `set.wind_vec`, not `wing.wind_disturb`

The turbulence feature as merged does not perturb the simulation at all. `PlanTurbulence.md`
step 3 picked `wing.wind_disturb` as the injection point on the assumption that it feeds the
wing's aerodynamics. It does not — not for the wing type V3Kite actually builds. This plan
replaces the injection point, removes the feedback loop the replacement would otherwise
introduce, and adds the regression test whose absence let this ship.

## Why `wind_disturb` is dead on this plant

V3Kite always builds a `PARTICLE_DYNAMICS` wing (`src/stabilization.jl:340`, `:373`,
`src/sim_helpers.jl:744`). For that wing type the aero force is a per-**point** VSM solve,
and the per-point apparent wind never sees `wind_disturb`:

- `wind_disturb` is a field of the wing body (`SymbolicAWEModels/src/system_structure/rigid_body.jl:97`)
  and enters the equations in exactly one place, `va_wing`
  (`SymbolicAWEModels/src/generate_system/scalar_eqs.jl:57-61`).
- Point apparent wind is built independently as `wind_factor(z) * wind_vec_gnd - vel`
  (`SymbolicAWEModels/src/generate_system/point_eqs.jl:98-101`). No disturbance term.
- Those point values are what reach the panels, via `set_particle_panel_va!`
  (`SymbolicAWEModels/src/aero_modes/common.jl:682-713`).
- `va_wing_b` survives only as the logged `angle_of_attack` (`scalar_eqs.jl:162`) and as the
  fallback inside `set_particle_panel_va!` when the structural↔panel mapping is missing —
  which for the V3 it never is (`vsm_refine.jl:428` asserts it is present).

So [`update_turbulence!`](src/interface.jl#L676-L690) today moves the logged AoA and nothing
else. Note `wind_disturb` is not broken in general: it works for `RIGID_DYNAMICS` wings, which
take their aero from `wing.va_b` ← `va_wing_b`. It is simply not on the path this model uses.

`set.wind_vec` **is** live: it becomes a real MTK parameter
(`SymbolicAWEModels/src/generate_system/flat_params.jl:139-145`) that `next_step!` re-syncs
from `sys_struct.set.wind_vec` on every step (`symbolic_awe_model.jl:560`), and
`sam.set === sam.sys_struct.set` by construction (`symbolic_awe_model.jl:251`, asserted at
`:301`). It is the only per-step wind hook that reaches the whole system.

## The approximation this buys, and its cost

`set.wind_vec` is a **ground** vector scaled by the height profile factor at every position, so
writing the kite's turbulent wind into it applies that same fluctuation, coherently, to every
tether and bridle point as well. With the shipped `profile_law: 0` the factor is 1 everywhere,
so the entire tether feels the gust sampled at the kite.

That is physically wrong in the opposite direction from the status quo — over-coherent rather
than absent — and it overstates fluctuating tether drag. It is accepted deliberately here: it
is the only hook that exists, and a gust field that acts on the kite beats one that acts on
nothing. `PlanTurbulence.md` rejected this route ("**Do not do that**") on the double-counting
argument; the division below removes the double-counting, and the tether coherence is what
remains. **This must be stated in the CHANGELOG, not left implicit.** The second return value
of `calc_turbulent_wind` (`v_wind_tether`) stays unused either way.

## Step 1 — store the commanded mean wind on `V3KITE`

Add to the `V3KITE` struct ([src/interface.jl:62](src/interface.jl#L62)):

```julia
"Commanded mean ground wind vector [m/s]; `set.wind_vec` is this except during the solve."
wind_vec_mean::SVec3 = zeros(SVec3)
```

Set it in `init` next to `set.wind_vec = wind_vec` ([src/interface.jl:618](src/interface.jl#L618)),
and in `step!` wherever the live-wind kwargs update `set.wind_vec`
([src/interface.jl:743-747](src/interface.jl#L743-L747)).

This is the "separate field so we don't pollute the settings" idea from the review, in the
form that costs least. The alternative — giving `am` its own `deepcopy(set)` — is the wrong
lever: it re-opens the staleness bug that `init` just closed by re-pointing `am.set` at the
live `set` ([src/interface.jl:624](src/interface.jl#L624)), and leaves two `Settings` objects
to keep in sync forever, since `am.set` also drives `calc_rho` and `calc_wind_factor` inside
the DAE.

## Step 2 — why the mean must be kept out of `set`

Assigning `set.wind_vec` triggers KiteUtils' `sync_wind!`
(`KiteUtils/src/settings.jl:373-484`), which with `use_wind_vec: true` overwrites `v_wind`,
`upwind_dir` **and** `upwind_elevation` from the vector. Leaving a turbulent value there feeds
back through four separate paths:

| path | consequence |
| :--- | :--- |
| `get_wind`: `am.set.v_wind * calc_wind_factor(...)` (`AtmosphericModels/src/windfield.jl:333,382`) | Taylor advection speed and the added mean both jitter |
| `rel_turbo(am)` snaps `am.set.v_wind` to the nearest `v_wind_gnds` entry (`windfield.jl:66-70`) | turbulence **intensity jumps discontinuously** at a bin edge |
| `calc_wind_factor4/5/6` divide by `set.v_wind` (`AtmosphericModels/src/AtmosphericModels.jl:278,281,366`) | under those profile laws the Step 3 denominator becomes turbulence-dependent |
| `step!`: `vw = something(v_wind_gnd, s.set.v_wind)`, `ud = deg2rad(s.set.upwind_dir)` ([src/interface.jl:744-745](src/interface.jl#L744-L745)) | a caller passing only one kwarg bakes the last turbulent value in as the new mean |

Hence: `set.wind_vec` holds the mean at all times **except** across the solver call.

## Step 3 — replace `update_turbulence!`

```julia
function apply_turbulence!(s::V3KITE)
    s.set.use_turbulence > 0 || return nothing
    ud = upwind_dir(s.wind_vec_mean)
    isfinite(ud) || return nothing
    pos = s.sys.wings[1].pos_w
    v_turb, _ = calc_turbulent_wind(s.am, pos, s.sys_state.time; upwind_dir = ud)
    s.set.wind_vec = v_turb / calc_wind_factor(s.am, max(pos[3], 10.0))
    return nothing
end
```

Four corrections against the sketch this plan replaces:

1. **`upwind_dir` is radians.** `set.upwind_dir` is stored in **degrees** — `step!` does
   `deg2rad(s.set.upwind_dir)` ([src/interface.jl:745](src/interface.jl#L745)). Derive it from
   `wind_vec_mean` instead, and keep the existing `isfinite` guard.
2. **`calc_turbulent_wind` returns a 2-tuple** `(v_wind, v_wind_tether)`; destructure it.
3. **The clamp must match upstream.** The mean baked into `v_turb` is evaluated at
   `max(pos[3], MIN_KITE_HEIGHT = 6.0)` (`windfield.jl:414`) and then clamped again to 10 m
   inside `get_wind` (`windfield.jl:321`), so the effective height is `max(pos[3], 10.0)`.
   Dividing by the factor at any other height leaves a residual. Harmless at `profile_law: 0`
   (factor ≡ 1), silently wrong the moment anyone switches profile law — i.e. exactly when this
   code matters. See Step 7 for removing the duplicated constant.
4. **One wing only.** `wind_vec` is global, so no per-wing loop is possible; take the first
   wing's position and let the (single-wing) V3 be the documented scope.

Delete the `wing.wind_disturb` writes entirely rather than keeping a second path for
`RIGID_DYNAMICS`. The `wind_vec` route is correct for rigid wings too (`wind_disturb` then
stays zero and `va_wing` picks the turbulence up through `wind_vec_gnd`), and one tested path
beats two.

## Step 4 — sequence it inside `step!`

`set.wind_vec` must hold the mean when `apply_turbulence!` runs (it reads `am.set.v_wind`
internally) and the turbulent value when `sim_step!` runs:

```julia
apply_turbulence!(s)
try
    sim_step!(s.sam; set_values = [torque], dt, vsm_interval) || error(...)
    update_sys_state!(s.sys_state, s)
finally
    s.set.wind_vec = s.wind_vec_mean
end
```

The `finally` matters: `sim_step!` failing must not leave a turbulent gust latched into the
settings. Restoring **after** `update_sys_state!` is deliberate — `ss.v_wind_gnd .= set.wind_vec`
(`symbolic_awe_model.jl:415`), so the log carries the instantaneous wind the kite actually saw,
which is what a turbulence run wants to plot. Say so in the `step!` docstring.

Also note `wind_vec` now carries a **vertical** component from the Mann field's `w`, so
`sync_wind!` will start writing a non-zero `set.upwind_elevation` — which `init` and `step!`
currently hardcode to `0.0` via `wind_vec_from_angles(vw, ud, 0.0)`. Feeding the mean from
`wind_vec_mean` rather than from the scalars keeps that consistent.

## Step 5 — regression tests

Nothing in `test/` asserts that turbulence changes a trajectory; `test-default_turbulence.jl`
covers only the YAML preference plumbing. That gap is the reason this shipped. Add two tests:

- **Injection point is live** (no wind field needed, so it runs anywhere `init` runs): from a
  settled model, run N steps unperturbed, then re-init and run N steps with `set.wind_vec`
  perturbed by a few percent each step, and assert the kite positions diverge beyond a
  threshold. This is the assertion that fails today against `wind_disturb` and passes after
  Step 3.
- **End-to-end turbulence**, gated on the wind field being present on disk (1.24 GB, not a CI
  artefact): N steps at `use_turbulence = 0` vs `> 0`, same assertion. Skip with an `@info`
  when the field is missing so the suite stays green on a fresh checkout.

`test/test-interface.jl:291` (`@test norm(s.set.wind_vec) ≈ 12.0`) is safe while that testset
runs at `use_turbulence = 0`, but it now encodes the mean-vs-instantaneous distinction — assert
against `s.wind_vec_mean` instead so it stays true if turbulence is ever switched on there.

## Step 6 — documentation

- Rewrite the `update_turbulence!` docstring ([src/interface.jl:663-674](src/interface.jl#L663-L674)).
  Its "only the *deviation* from that mean may be injected here" paragraph describes the
  mechanism being removed.
- Rewrite the CHANGELOG *Added* entry (`CHANGELOG.md:6-16`), which documents `wind_disturb` as
  the feature. State the tether-coherence approximation from the top of this plan.
- The `V3KITE.am` docstring and the "one shared `AtmosphericModel`" reasoning stay valid — one
  `am`, one `Settings`, unchanged.

## Step 7 — follow-up upstream (not blocking)

The two fragilities that remain are both in `AtmosphericModels`' interface, not in this fix:

- `calc_turbulent_wind`/`get_wind` read `am.set.v_wind` and `rel_turbo(am)` from mutable shared
  state, which is what forces the save/restore dance in Step 4. Keyword overrides
  (`v_wind = ...`, `rel_turb = ...`) would let the turbulence lookup be a pure function of the
  commanded mean.
- The height clamp reproduced in Step 3 duplicates an upstream internal. An accessor returning
  the mean that `calc_turbulent_wind` actually added — or a variant returning the fluctuation
  alone — would remove the duplication and the whole division.

`AtmosphericModels` compat is being bumped for this feature anyway, so both are cheap to land.
