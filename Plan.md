# address reviewer comment
Key question: Is this function in src/interface.jl needed:
 _wind_vec(v_wind_gnd, upwind_dir) -> Vector{Float64}

## First comment by Bart
You can just use set.wind_vec, which is always in sync.

## First reply
They serve different purposes — _wind_vec isn't a duplicate of set.wind_vec, it's the conversion function that produces it:

Line 453 (in init): set doesn't exist yet at this point — settle_wing is what constructs it (line 471, returning sam.set at line 480). But settle_wing needs a wind vector as one of its inputs. So _wind_vec has to convert the user-facing scalar inputs (v_wind_gnd, upwind_dir) into a 3-vector before set is available. There's no set.wind_vec to read yet.

Line 545 (in step!): here set.wind_vec already exists, but step! accepts wind updates as scalar speed/direction (v_wind_gnd, upwind_dir), and either can be omitted independently. Lines 542–544 decompose the current set.wind_vec back into speed (norm(v)) and direction (atan(...)) to fill in whichever argument wasn't given, then _wind_vec recombines speed+direction into the new vector. You can't skip this: set.wind_vec is the vector representation, but the function's public API is in terms of speed/direction, so something has to do that polar→Cartesian conversion.

So _wind_vec is genuinely needed as a helper — it's the (speed, direction) → ENU-vector transform, used both to build the initial vector before set exists and to reconstruct the vector after a partial scalar update.

## Second comment by Bart
No, not true, take a look at KiteUtils.jl. If you do set.use_wind_vec=false, you can specify set.upwind_dir and set.v_wind, and then set.wind_vec is automatically updated. The vector in Settings is automatically updated after a scalar update.

## Answers to the open questions (verified against the code and live KiteUtils 0.11.9)

Conclusion: **Bart is right, and _wind_vec can be removed** — it duplicates
`KiteUtils.wind_vec_from_angles` (exported, already a dependency).

### Where is this function used?
Exactly two call sites, both in src/interface.jl:
- interface.jl:453 in `init` — builds the initial wind vector for `settle_wing`.
- interface.jl:545 in `step!` — recombines the vector after a partial live wind
  update.

### Is it really needed at all places where it is used now?
No, at neither site:
- The "set doesn't exist yet in init" argument from the first reply is true but
  irrelevant: `wind_vec_from_angles(v_wind_gnd, upwind_dir, 0.0)` is a
  standalone KiteUtils function that needs no `Settings` instance, and it is
  algebraically identical to `_wind_vec` (both give `east = -v*sin(ud)`,
  `north = -v*cos(ud)`; default case `[v, 0, 0]` matches).
- In `step!`, the decompose step (interface.jl:542–544, `norm(v)`/`atan(...)`)
  is also redundant: with `use_wind_vec: true` (set in data/settings.yaml:65),
  KiteUtils' `setproperty!` hook keeps `set.v_wind` and `set.upwind_dir` in
  sync with `set.wind_vec` via `sync_wind!` (verified live). So `step!` can
  read `s.set.v_wind` and `deg2rad(s.set.upwind_dir)` for the omitted argument
  and call `wind_vec_from_angles` to recombine.

### Is this function only used when setting the initial conditions?
No — the `step!` site is a live per-step update. Architecturally that is fine:
`set.wind_vec` is deliberately a live field, read by the DAE every step via
`get_wind_vec`, so it isn't purely an initial condition. The background concern
applies to the *scalars*, and KiteUtils already resolves it with the two-way
`sync_wind!` mechanism.

## Caveats for the reply to Bart
1. **Degrees vs. radians.** `Settings.upwind_dir` is stored in degrees;
   V3Kite's `init`/`step!` API takes radians. Bart's exact suggestion
   (set `set.v_wind`/`set.upwind_dir` with `use_wind_vec=false`) requires
   `rad2deg` at the boundary, or changing the public API to degrees.
2. **The use_wind_vec trap.** Verified live: with `use_wind_vec=false`, a
   direct assignment `set.wind_vec = [...]` is **silently clobbered** —
   `sync_wind!` immediately recomputes it from the scalars. The direct
   assignments at interface.jl:481, interface.jl:545, stabilization.jl:180 and
   flight_data.jl:333 only work because settings.yaml has
   `use_wind_vec: true`. Adopting the `use_wind_vec=false` route would require
   converting *all* of those sites to scalar assignments; keeping
   `use_wind_vec: true` and just replacing `_wind_vec` with
   `wind_vec_from_angles` is the smaller, safer change.

## Recommended resolution
Delete `_wind_vec` and use the KiteUtils primitives, keeping
`use_wind_vec: true`:

```julia
# init (line 453)
wind_vec = wind_vec_from_angles(v_wind_gnd, upwind_dir, 0.0)

# step! (lines 541–546)
if v_wind_gnd !== nothing || upwind_dir !== nothing
    vw = something(v_wind_gnd, s.set.v_wind)          # auto-synced by KiteUtils
    ud = upwind_dir === nothing ? deg2rad(s.set.upwind_dir) : upwind_dir
    s.set.wind_vec = wind_vec_from_angles(vw, ud, 0.0)
end
```

This concedes Bart's point (the conversion already lives in KiteUtils and the
scalars are auto-synced) while keeping the vector-driven update style the rest
of the codebase relies on. Behavioral note: like the current code, this drops
any `upwind_elevation` component on update — to preserve tilted wind, pass
`deg2rad(s.set.upwind_elevation)` as the third argument instead of `0.0`.

## TODO
- [x] Apply the recommended change in src/interface.jl and delete `_wind_vec`.
- [x] Verify: `wind_vec_from_angles` matches the old `_wind_vec` to 4e-15;
      `examples/simple_parking.jl` runs the full 10 s; partial wind updates in
      `step!` (speed-only, direction-only) preserve the omitted component and
      keep the scalars in sync.
- [ ] Reply to Bart with the conclusion and the two caveats above.
