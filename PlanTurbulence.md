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

## Step 1 — dependency (done)

`AtmosphericModels` 0.3.6 exports `calc_turbulent_wind`.

- `Project.toml`: `AtmosphericModels` compat bumped `"0.3.4"` → `"0.3.6"`, so a resolve cannot
  silently land on 0.3.5, where `calc_turbulent_wind` does not exist.
- `Manifest-v1.12.toml` and `Manifest-v1.12.toml.default` both pin 0.3.6 and match each other.

## Step 2 — one shared `AtmosphericModel` (done)

Today two atmospheric models are built from the same settings:

- `V3KITE.am = AtmosphericModel(set)` — [src/interface.jl:69](src/interface.jl#L69)
- `sys_struct.am = AtmosphericModel(set)` — `SymbolicAWEModels/src/system_structure/system_structure_core.jl:954`

With `use_turbulence > 0` **both eagerly load a wind field**, and a wind field for the configured
grid is 1.24 GB on disk (`windfield_4050_100_500_70_1.0_8.2.npz`) — i.e. ~2.5 GB of RAM for two
copies of the same data.

Fix applied: the `V3KITE` field default is now `am::AtmosphericModel = sam.am`
([src/interface.jl:76](src/interface.jl#L76)). `sam.am` forwards to `sam.sys_struct.am`
(`symbolic_awe_model.jl:190-191`), so every construction site shares the DAE's instance — not just
`init`, but also the bare `V3KITE(set=..., kcu=..., sam=...)` form used in `test/test-interface.jl`.
Sharing is safe because the live `sys_struct` (and its `am`) is always built in-process: only
`full_sys` comes from the `.bin` cache, and the deserialized problem's parameters are re-pointed at
the current `sys_struct` (`model_management.jl:423`).

Verified in the REPL: `s.am === sam.sys_struct.am` and `s.set === s.am.set` are both `true`, and
`test/test-interface.jl` passes (60 + 23 + 26 tests).

## Step 3 — feed the turbulent deviation into the wing (done)

Implemented as `update_turbulence!(s::V3KITE)`
([src/interface.jl:644-679](src/interface.jl#L644-L679)), called from `step!` right after the
optional live wind update and before `sim_step!`
([src/interface.jl:703](src/interface.jl#L703)):

```julia
function update_turbulence!(s::V3KITE)
    # `upwind_dir` must be qualified here and in `step!`: the keyword argument of `step!`
    # shadows the V3Kite function of the same name.
    ud = V3Kite.upwind_dir(s)   # NaN if the wind vector has no horizontal component
    for wing in s.sys.wings
        if s.set.use_turbulence > 0 && isfinite(ud)
            pos = wing.pos_w
            v_turb, _ = calc_turbulent_wind(s.am, pos, s.sys_state.time; upwind_dir = ud)
            v_mean = calc_wind_factor(s.am, max(1.0, pos[3])) * s.set.wind_vec
            wing.wind_disturb .= v_turb .- v_mean
        else
            wing.wind_disturb .= 0.0
        end
    end
    nothing
end
```

Differences from the sketch this plan originally carried: it loops over all wings instead of
hardcoding `wings[1]`, it guards against a `NaN` upwind direction (zero horizontal wind) rather than
letting `NaN` propagate into the solve, and the zeroing branch is part of the same function.

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

`wing.pos_w` is refreshed from the integrator at the end of every step by
`update_sys_struct!` (`symbolic_awe_model.jl:474, 568`), and that function does *not* touch
`wind_disturb` — so the position read here is current and the injected disturbance survives the step.

Verified so far:

- `test/test-interface.jl` passes with the new call in `step!` (60 + 23 + 26 tests) — this exercises
  the `use_turbulence == 0` path, including the `V3Kite.upwind_dir(s)` qualification.
- The injection path itself was checked directly against the DAE: with
  `wing.wind_disturb = [1, 0, 0]` and one `sim_step!`, the residual of
  `va_b - R_b_to_w' * (v_wind - vel_w + wind_disturb)` is 5.4e-10, while dropping the
  `wind_disturb` term leaves exactly 1.0. The disturbance reaches the wing's apparent wind unchanged.
- **Still unverified**: the `use_turbulence > 0` path, which needs a wind field — see step 4.

Optionally log `norm(wing.wind_disturb)` to a spare `sys_state` slot in `update_sys_state!` so the
turbulence is visible in the plots.

## Step 4 — wind fields for the V3 wind speeds

`load_windfield` snaps to the **closest** entry of `set.v_wind_gnds` (`windfield.jl:117-121`), while
`get_wind` computes the mean wind from `set.v_wind` (`windfield.jl:333`), so a speed that is not in
the list borrows a neighbour's turbulence over its own mean wind.

Measured cost of that snap, for `settings_reelout.yaml` (turbulence intensity at 99 m, computed as
`rel_turb * calc_sigma1(am, v) / (v * calc_wind_factor(am, 99))`):

| `v_wind_gnds` | `rel_turbs` | I₉₉ |
|---:|---:|---:|
| 3.483 | 0.342 | 9.7 % |
| 5.324 | 0.465 | 10.4 % |
| 8.163 | 0.583 | 10.7 % |
| 9.51 | *0.583, snapped* | 10.1 % |
| 15.4 | *0.583, snapped* | 8.6 % |

Two things this corrects: V3Kite uses `alpha = 0.08163` rather than Cabauw's 0.234, yet the Cabauw
`rel_turbs` still land close to the intended intensity (9.7 / 10.4 / 10.7 % against the measured
8.5 / 9.7 / 9.8 %); and the snapping penalty at 9.51 m/s is only ~6 % relative — it is 15.4 m/s,
where the intensity falls to 8.6 %, that is really mis-served.

Done:

1. `v_wind_gnds` extended to `[3.483, 5.324, 8.163, 9.51]` and `rel_turbs` to
   `[0.342, 0.465, 0.583, 0.626]` in all three settings files (`settings.yaml`,
   `settings_reelout.yaml`, `settings_cabauw.yaml`), which share `alpha`, `avg_height`, `i_ref` and
   `h_ref` and therefore share one set of field files.
2. The new `rel_turb` is **not** a free parameter — per `AtmosphericModels/docs/src/wind_field.md`
   these are correction factors calibrated against measured Cabauw intensities, and the measured
   table stops at 8.163 m/s. 0.626 continues the log fit `rel_turb = 0.342 + 0.283*(ln v - 1.248)`,
   which reproduces the three calibrated points almost exactly (successive slopes 0.290 and 0.276).
   It puts I₉₉ at 10.9 % for 9.51 m/s, continuing the 9.7 → 10.4 → 10.7 % trend. A comment in each
   settings file records this.
3. Fields generated into `v3_data_path()` for 8.163 (covers the settings default `v_wind: 8.0`) and
   9.51 (the `simple_*`/`reel_out_v3`/`steering_test_v3` examples), 1.24 GB each:

   ```julia
   set_data_path(v3_data_path())
   set = load_settings("system_reelout.yaml"; relax=true)
   set.use_turbulence = 1.0
   am = AtmosphericModel(set; nowindfield=true)
   for v in (8.163, 9.51); new_windfield(am, v); end
   ```

   3.483 and 5.324 stay listed but have no pre-generated file; a run at those speeds triggers an
   on-the-fly `new_windfield` (`windfield.jl:109-112` only `@warn`s). 15.4 m/s is not covered at all
   and still snaps to 9.51.

**Grid**: V3Kite's settings files have no `grid:` key, so `set.grid` is KiteUtils' default
`[100, 4050, 500, 70]` — the **short** dimension first. This is exactly the layout the 0.3.6
`get_wind` fix handles (before it, dimension 1 was assumed to be the long, along-wind one), so the
step 1 compat floor is required for correctness here, not only for `calc_turbulent_wind`. It also
gives the basename `windfield_100_4050_500_70_*`, distinct from the `windfield_4050_100_500_70_*`
files in `AtmosphericModels/data`.

**Do not copy fields between packages.** The filename encodes only grid, `use_turbulence` and ground
wind speed — not `alpha`, `avg_height` or `i_ref`, all of which enter `calc_sigma1`. A file generated
under another package's settings would load without complaint and be silently wrong.

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
