# Add a REST server for init()/step!() plus a Matlab parking client

Create `examples/rest_server.jl`: an HTTP/REST interface for driving a V3Kite
simulation from Matlab, Python or any HTTP client, plus a Matlab client
`examples/matlab/simple_parking_client.m` that mirrors
`examples/simple_parking.jl` over the wire.

## Framing

A proven design: one process, one global session, lock-serialized model
access, background init with live progress. Notes specific to V3Kite's
high-level interface:

- model type: `V3KITE`
- step call: `step!(s; rel_depower, rel_steering, set_length, ...)` (kwargs)
- steering: single `rel_steering` in `[-1, 1]`
- init extras: `system_yaml` (e.g. `"system_cabauw.yaml"`)
- trim outputs: none — client picks `rel_depower`; length setpoint is
  `l0 = s.sys_state.l_tether[1]`
- winch state: single winch: `l_tether`, `v_reelout`, `winch_force` are
  1-element vectors

## Design decisions

1. **Architecture**: one server process, ONE global `Session` (no session
   ids), a `ReentrantLock` around every model-touching endpoint (the model is
   not thread-safe), a second `msg_lock` for the `/status` message buffer,
   and the state machine

       idle → initializing → ready ⇄ (stepping)
                        ↘ failed        ↘ failed

   Init runs on a background task (`Threads.@spawn`) with a
   `CollectLogger`/`TeeLogger` pair so `@info` progress from `init` (settling
   can be slow on a cold cache) is streamed into the `/status` buffer while
   also reaching the terminal. Structure: `Session`, `CollectLogger`,
   `push_message!`, the `sanitize`/`to_float`/`getf`/`get_bool`/`get_int`
   helpers, `json_response`, `parse_body`, `truthy`, and `main`, with the
   model field typed `s::Union{V3KITE, Nothing}`.

2. **`/init` (POST, → 202)**: accepts `v_wind`, `l_tether`,
   `depower_setpoint`, `sim_time`, and additionally `system_yaml` (string).
   Defaults are the simple_parking.jl operating point: `10.0`, `150.0`,
   `0.25`, `10.0`, `"system_cabauw.yaml"`. Calls

       init(v_wind, l_tether; depower_setpoint, sim_time, system_yaml)

   `system_yaml` is used to open files server-side, so validate it with the
   same simple-filename regex used for `save_log` names (plus a required
   `.yaml` suffix) to block path traversal. 409 if an init is already
   running.

3. **Init result (published in `/status` when `state == "ready"`)**: V3Kite
   has no trim values to report, so the result dict is

   - `l0` — `s.sys_state.l_tether[1]` right after init; the client sends it
     back as `set_length` to park at constant length (exactly `l0` in
     simple_parking.jl),
   - `steps` — `s.steps` (step budget),
   - `dt` — `s.dt` (for realtime-factor reporting on the client).

4. **`/step` (POST)**: requires `rel_depower` and `set_length`; optional
   `rel_steering` (default `0.0`, validated to `[-1.0, 1.0]`), `steps`
   (batch size, default 2,
   clamped to the remaining budget) and `full_state` (bool). Each sub-step
   calls

       step!(s; rel_depower, rel_steering, set_length)

   Winch position mode only for now — `set_torque`/`speed_limit`/
   `acceleration_limit` can be added later as optional payload keys without
   breaking the protocol, since `step!` already supports them. Only the LAST
   sub-step's state is returned (top-level fields plus `n` and
   `rel_steering`); intermediates still go to the server-side logger, so
   `save_log` keeps full resolution. Step errors → state `"failed"`, 500
   with the error string. 409 when not ready or the budget is exhausted.

5. **`KiteState` projection** (V3 rename of `WingState`): only the fields
   the clients plot/control on, taken from the panels of
   simple_parking_plots.jl plus the control quantities:

   `time`, `elevation`, `azimuth`, `heading`, `v_app`, `AoA`,
   `l_tether`, `v_reelout`, `winch_force` (single winch → store
   `first(...)` as scalars), `var_15`
   (L/D wing), `var_16` (L/D eff). Non-finite floats → JSON `null` via
   `sanitize`, `JSON3.StructTypes.Struct()` for direct serialization.
   `full_state=true` returns every `SysState` field via `full_state_dict`
   (unchanged helper — `sanitize` recurses into the vector-valued fields).

6. **`/state` (GET)** and **`/save_log` (POST)**:
   `save_log(SESSION.s.logger, name)` with the same filename whitelist;
   `init` already ran `set_data_path(v3_data_path())`, so `get_data_path()`
   points at the V3 data dir and the returned `path` is
   `joinpath(get_data_path(), name) * ".arrow"`. Default name `"tmp_run"`
   so simple_parking_plots.jl can plot the REST run unmodified.

7. **Env/port naming**: `V3_REST_PORT` and `V3_REST_ACCESS_LOG` (access
   logging off by default — the Oxygen per-request log line measurably slows
   the tight step loop). Bind to `127.0.0.1` only, default port 8080,
   overridable by first CLI arg. Auto-start only when run as a script
   (`abspath(PROGRAM_FILE) == @__FILE__`).

8. **Dependencies**: add `Oxygen`, `HTTP`, `JSON3`, and `LoggingExtras` to
   `examples/Project.toml` (`Logging` is a stdlib; none of the four are
   currently present). Nothing is added to the top-level `Project.toml` —
   the server is an example, not part of the package.

## Files

### examples/rest_server.jl

Port as described above. Docstring: endpoints, batching rationale, the
state machine, the threads note (run `julia -t 4 --project=examples
examples/rest_server.jl [port]`; single-threaded works but `/status` only
fills after init finishes), and a pointer to this plan and the Matlab
client.

### examples/matlab/simple_parking_client.m

A Matlab client adapted to the V3 protocol:

- Parameters from simple_parking.jl: `V_WIND = 10.0`, `TETHER_LENGTH =
  150.0`, `DEPOWER_SETPOINT = 0.25`, `SIM_TIME = 10.0`, `PROJECT =
  "system_cabauw.yaml"`; plus `STEPS_PER_CALL = 2` and `FULL_STATE = false`.
- Flow: POST `/init` → poll `/status` (printing new messages) until
  `ready`/`failed` → read `l0`, `steps`, `dt` from `result` → loop POSTing
  `/step` with `rel_depower = DEPOWER_SETPOINT`, `set_length = l0`,
  `steps = n_req`, recording one sample per call → POST `/save_log` with
  name `"tmp_run"` → plot.
- Progress line every ~100 steps with the client-observed realtime factor.
- Plot: subplots mirroring the simple_parking_plots.jl panels from the
  recorded per-call samples — `v_reelout` [m/s], `winch_force` [N],
  `rad2deg(elevation)`, `rad2deg(heading)`, `rad2deg(AoA)`, and L/D
  (`var_15`/`var_16` in one axes). (The depower panel is a constant, skip
  it.)

### bin/run_server

Launcher script adapted to V3: `cd` out of `bin/` if
needed, use the `bin/kps-image-*.so` sysimage when present, run
`julia -t 4 --project=examples examples/rest_server.jl "$@"`, honor
`V3_REST_PORT`. Make it executable.

### examples/Project.toml

Add `Oxygen`, `HTTP`, `JSON3`, `LoggingExtras` to `[deps]` (and resolve so
`examples/Manifest.toml` picks them up).

## Verification

A server start plus init plus a 10 s parking run exceeds 90 s wall time on a
cold cache; per CLAUDE.md do not run the full loop as a verification test.
What CAN be checked cheaply from the kaimon REPL: that the server file
includes cleanly (no `main()` when included), and — with the server started
in a terminal via `bin/run_server` — a smoke test of the protocol with
`HTTP.jl` (`POST /init`, poll `/status`, a couple of `/step` calls,
`/save_log`), which doubles as a Matlab-free reference client. Finish with
the message: start the server with `bin/run_server`, then run
`examples/matlab/simple_parking_client.m` in Matlab; afterwards
`include("examples/simple_parking_plots.jl")` plots the full-resolution
server-side log.
