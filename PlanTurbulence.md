# Allow to use a turbulent wind field

Use `AtmosphericModels.calc_turbulent_wind` (new in 0.3.6) to add Mann-model turbulence to the V3
kite simulation, and add a `set_default_turbulence` entry to `examples/menu2.jl`.

## Why this is not a copy of the KiteModels approach

`KiteModels.jl` writes the turbulent wind straight into the model state: `set_v_wind_ground!`
(`src/KiteModels.jl:220-236`) assigns `s.v_wind` and `s.v_wind_tether`, and the hand-written
residual reads those fields.

V3Kite has no such fields. Wind enters the ModelingToolkit DAE built by `SymbolicAWEModels`:

- `wind_vec_gnd ~ get_wind_vec(psys)` reads `set.wind_vec` (`generate_system/scalar_eqs.jl:41`),
- every wing, tether segment and point multiplies it by
  `calc_wind_factor(am, x, y, z, psys)` (`generate_system/helpers.jl:101-115`), which is
  `@register_symbolic`, **height-only and time-independent**.

Consequences that shape this plan:

1. `set.wind_vec` is a *ground* vector that gets scaled by the profile factor at every position.
   Overwriting it with the turbulent wind at the kite would double-count the profile law and apply
   the kite's turbulence to the whole tether. **Do not do that.**
2. The only per-body wind hook that exists today is `wing.wind_disturb`, a mutable `KVec3` on the
   wing struct (`system_structure/wing.jl:153`) read via `get_wind_disturb(psys, idx)` and added to
   the wing's apparent wind (`generate_system/scalar_eqs.jl:52-55`). That is the injection point.
3. The disturbance must therefore be the **deviation from the mean wind**, not the wind itself.
4. It is updated once per `step!` and held constant during the solve. This is required, not a
   shortcut: `get_wind` is a nearest-grid-point lookup, so making it a function of the DAE's `t`
   would put a discontinuous, non-differentiable term inside the Newton/BDF stages.

Tether turbulence is **out of scope**: `generate_system/segment_eqs.jl:175` has no disturbance term,
so the second return value of `calc_turbulent_wind` (`v_wind_tether`) has nowhere to go. See
"Follow-up" below.

## Step 1 — dependency

`AtmosphericModels` 0.3.6 exports `calc_turbulent_wind`. Once it is registered:

- `Project.toml`: bump `AtmosphericModels = "0.3.4"` → `"0.3.6"`.
- Refresh `Manifest-v1.12.toml.default`.

## Step 2 — one shared `AtmosphericModel`

Today two atmospheric models are built from the same settings:

- `V3KITE.am = AtmosphericModel(set)` — [src/interface.jl:69](src/interface.jl#L69)
- `sys_struct.am = AtmosphericModel(set)` — `SymbolicAWEModels/src/system_structure/system_structure_core.jl:954`

With `use_turbulence > 0` **both eagerly load a wind field**, and a wind field for the configured
grid is 1.24 GB on disk (`windfield_4050_100_500_70_1.0_8.2.npz`) — i.e. ~2.5 GB of RAM for two
copies of the same data.

Fix: in `init` ([src/interface.jl:628](src/interface.jl#L628)) pass `am = sam.sys_struct.am` to the
`V3KITE` constructor so there is exactly one instance. The `sys_struct` is fully built at that point
(also when it is deserialized from the model `.bin` cache), so the reference is valid.

## Step 3 — feed the turbulent deviation into the wing

In `step!` ([src/interface.jl:676-716](src/interface.jl#L676-L716)), after the optional live wind
update and **before** `sim_step!`:

```julia
# Turbulence: the DAE scales set.wind_vec by the height-only profile factor, so only the
# deviation from that mean may be injected — via the wing's wind_disturb parameter, which
# SymbolicAWEModels adds to the wing's apparent wind. Held constant over the step: get_wind
# is a nearest-grid-point lookup and must not be evaluated inside the implicit solve.
if s.set.use_turbulence > 0
    wing = s.sys.wings[1]
    pos = wing.pos_w
    v_turb, _ = calc_turbulent_wind(s.am, pos, s.sys_state.time; upwind_dir = upwind_dir(s))
    v_mean = calc_wind_factor(s.am, max(1.0, pos[3])) * s.set.wind_vec
    wing.wind_disturb .= v_turb - v_mean
end
```

Details that must not drift:

- **Position**: use `wing.pos_w`, the position the DAE itself uses for the wing wind, *not*
  `pos_kite(s)` ([src/interface.jl:190](src/interface.jl#L190), a bridle-point average). The mean
  subtracted here has to be the mean the DAE adds back, or the deviation does not cancel.
- **Height clamp**: `max(1.0, pos[3])` matches `SymbolicAWEModels`' `calc_wind_factor`;
  `calc_turbulent_wind` applies its own `MIN_KITE_HEIGHT` clamp internally. They differ only within
  a few metres of the ground.
- **Profile law**: call the two-argument `calc_wind_factor(am, height)`, which defaults to
  `set.profile_law` — the same value `get_wind` uses (`windfield.jl:333`). Never hardcode a law.
- **Angle units**: use `upwind_dir(s)` ([src/interface.jl:234](src/interface.jl#L234)), which returns
  **radians**, as `calc_turbulent_wind` expects. Note `set.upwind_dir` is in degrees and is converted
  with `deg2rad` at [src/interface.jl:690](src/interface.jl#L690) — do not pass it directly.
- **Time**: `s.sys_state.time` is monotonically increasing and starts at 0; `get_wind` asserts
  `t >= 0` and `z >= 5`.
- Reset `wing.wind_disturb` to zero when `use_turbulence == 0` is toggled at runtime (it is a
  persistent field on the struct).

Optionally log `norm(wing.wind_disturb)` to a spare `sys_state` slot in `update_sys_state!` so the
turbulence is visible in the plots.

## Step 4 — wind fields for the V3 wind speeds

`load_windfield` snaps to the **closest** entry of `set.v_wind_gnds` (`windfield.jl:117-121`), while
`get_wind` computes the mean wind from `set.v_wind` (`windfield.jl:333`). The current
`v_wind_gnds: [3.483, 5.324, 8.163]` are the Cabauw scenarios; the V3 examples run at 7.6, 9.51,
10.0 and 15.4 m/s, so today every one of them would silently borrow the 8.163 m/s field — turbulence
intensity too low by ~15-25 % at 9.5 m/s and roughly half at 15.4 m/s.

Therefore, in `data/settings.yaml` (and `settings_reelout.yaml` / `settings_cabauw.yaml` if they are
to support turbulence too):

1. Extend `v_wind_gnds` with the speeds the examples actually use. Suggested first pass: `9.51`
   (used by `simple_parking`, `reel_out_v3`, `steering_test_v3`), then `15.4` if the batch/fig-8
   runs need it. Each additional speed costs **1.24 GB on disk and several minutes** to generate, so
   add them deliberately rather than all at once.
2. Extend `rel_turbs` with a matching entry per speed. **These are not free parameters**: per
   `AtmosphericModels/docs/src/wind_field.md`, they are correction factors calibrated so the Mann
   field reproduces the turbulence intensity measured at Cabauw (8.5 / 9.7 / 9.8 % at 99 m). The
   measured table stops at 8.163 m/s, so values above that are an extrapolation. Procedure: pick a
   starting value by extrapolating the existing trend, generate the field, measure the resulting
   intensity with the tooling in `AtmosphericModels/examples/plot_windfield.jl` (or `test_all.jl`),
   and iterate until it matches the intended target. Record the target used, in a comment next to
   the new entries.
3. Generate the fields into V3Kite's data path (`v3_data_path()` → `data/`), which currently holds
   no `.npz` at all:

   ```julia
   set_data_path(v3_data_path())
   set = load_settings("system.yaml"; relax=true)
   set.use_turbulence = 1.0
   am = AtmosphericModel(set; nowindfield=true)
   new_windfields(am)          # all speeds in set.v_wind_gnds
   ```

   Without this, the first turbulent run falls into `new_windfield` unnoticed (`windfield.jl:109-112`
   only warns) and blocks for minutes. `data/*.npz` is already in `.gitignore`.

Note the filename also encodes `rel_sigma = set.use_turbulence` with one decimal
(`calc_full_name`, `windfield.jl:85-90`), so a non-default `use_turbulence` needs its own generated
set of files.

## Step 5 — `set_default_turbulence` menu entry

`examples/menu2.jl` builds its options as `"name = include(\"file.jl\")"` strings and runs them with
`eval(Meta.parse(...))` ([examples/menu2.jl:20-35](examples/menu2.jl#L20-L35)), so any expression
works. Add `examples/set_default_turbulence.jl` plus a `push!` next to the other explicit entries.

What it must do, and why:

- **Set `use_turbulence` to `1.0`** (the Cabauw-calibrated reference intensity — the setting is a
  *relative* intensity, not an absolute one). Only values with matching `.npz` files avoid a 1.24 GB
  regeneration, so the script should check for the file and refuse/warn rather than silently
  generating.
- **Persist it to `data/settings.yaml`.** An in-memory change is useless: every `simple_*.jl` calls
  `set_data_path(v3_data_path())` and reloads the settings before `init`. Edit the file textually
  (a YAML round-trip through `YAML.jl` drops the extensive comments in that file).
- **State that it only takes effect for the next run.** `AtmosphericModel(set)` and the wind-field
  load happen at construction, so flipping the flag after `init` does nothing.
- A matching way back to `0.0` — either a second entry or a toggle that reports the new state.

## Step 6 — verification

There is no CI test for this: a meaningful check needs a 1.24 GB wind field, which cannot live in
the test suite. Verify manually and record the numbers in the PR:

1. Run `examples/simple_parking.jl` (V_WIND = 9.51) with `use_turbulence` 0.0 and 1.0.
2. Confirm the run stays stable and note the realtime-factor change.
3. Compare the standard deviation of apparent wind speed and angle of attack between the two runs;
   with `use_turbulence = 1.0` the apparent-wind std should land near the intensity the field was
   calibrated to at the kite's height.
4. Sanity-check the sign convention: with `wind_disturb` forced to a constant, e.g. `[1, 0, 0]`, the
   apparent wind must shift east by 1 m/s.

## Documentation

- `CHANGELOG.md`: new entry — turbulent wind support at the kite, `use_turbulence` in
  `data/settings.yaml`, new `v_wind_gnds`/`rel_turbs` entries, `set_default_turbulence` menu item.
- `CLAUDE.md` / README wind section: document that turbulence is applied **at the wing only**, that
  it is piecewise constant per `step!`, and that wind fields must be generated once per
  (grid, `use_turbulence`, ground wind speed) combination.

## Follow-up (not in this plan)

Tether turbulence needs an upstream change in `SymbolicAWEModels`: a `wind_disturb` field on the
segment struct, a `get_wind_disturb(psys, segment_idx)` accessor, and an additive term in
`generate_system/segment_eqs.jl:175-179`, followed by a release. Only then can the `v_wind_tether`
return value of `calc_turbulent_wind` be used, giving parity with `KiteModels`.
