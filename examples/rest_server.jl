# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
REST interface for driving a V3Kite simulation from Matlab, Python or any
HTTP client. Covers the
parking example (`examples/simple_parking.jl`): `init` + `step!` with
`rel_depower`/`rel_steering`/`set_length`, plus saving the log.

`/step` advances `steps` (default 2) Julia integration steps per call and
returns only the LAST state as a compact `KiteState` (just the fields the
clients plot/control on — see its docstring) at top level, plus `n` (steps
taken) and `rel_steering`. Pass `full_state=true` to get every `SysState`
field instead of the compact projection (same top-level layout, plus
`n`/`rel_steering`). Batching amortizes the HTTP round-trip, which otherwise
dominates the loop; the intermediate states are still written to the
server-side log, so `save_log` keeps full resolution.

Run it under the examples project:

    julia -t 4 --project=examples examples/rest_server.jl [port]

or via `bin/run_server`. Multiple threads (e.g. `-t 4`, as above) let
`/status` report live `@info` progress while the (settling can be slow on a
cold cache) initialization runs on a background thread; single-threaded
still works, but the message buffer only fills once init finishes. The port
defaults to 8080 and can be set via the first command-line argument or the
`V3_REST_PORT` environment variable. The server binds to 127.0.0.1
(localhost only).

Design (see PlanREST.md): one server process, ONE global session (one
`V3KITE`), no session ids. The model is not thread-safe, so every endpoint
that touches the model takes a single `ReentrantLock`. State machine:

    idle → initializing → ready ⇄ (stepping)
                     ↘ failed        ↘ failed

See `examples/matlab/simple_parking_client.m` for a client example.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: init, step!
using KiteUtils                       # SysState, Logger, get_data_path, save_log
using Oxygen
using HTTP
using JSON3
using Logging
using LoggingExtras

AERO_MODE = ContinuousAero()

# ============================ Session state ============================ #

"""
    Session

The single global simulation session. `lock` serializes every access to the
model (`s`) and to the state-machine fields; `msg_lock` guards the `messages`
buffer alone, so `/status` polls never contend with a running init/step.
"""
mutable struct Session
    lock::ReentrantLock
    msg_lock::ReentrantLock
    state::String                          # "idle" | "initializing" | "ready" | "failed"
    messages::Vector{String}
    error::Union{String, Nothing}
    result::Union{Dict{String, Any}, Nothing}
    s::Union{V3KITE, Nothing}
    step_count::Int
    init_task::Union{Task, Nothing}
end

Session() = Session(ReentrantLock(), ReentrantLock(), "idle",
                    String[], nothing, nothing, nothing, 0, nothing)

const SESSION = Session()

# ============================ Log capture ============================= #

"""
    CollectLogger(min_level, sink)

Minimal `AbstractLogger` that forwards each formatted log message (at
`min_level` or above) to `sink(::String)`. Combined via `TeeLogger` with the
console logger so init progress goes both to the terminal and the status
buffer.
"""
struct CollectLogger <: AbstractLogger
    min_level::LogLevel
    sink::Function
end
Logging.min_enabled_level(l::CollectLogger) = l.min_level
Logging.shouldlog(::CollectLogger, level, _module, group, id) = true
Logging.catch_exceptions(::CollectLogger) = false
function Logging.handle_message(l::CollectLogger, level, message, _module, group,
                                id, file, line; kwargs...)
    l.sink(string(message))
    return nothing
end

push_message!(session::Session, msg::AbstractString) =
    lock(() -> push!(session.messages, String(msg)), session.msg_lock)

# ============================ Serialization =========================== #

# Non-finite floats (`NaN`/`Inf`) are mapped to JSON `null`, since they are
# invalid JSON and would otherwise make `JSON3.write` throw and break the
# client decoder. In Matlab a `null` decodes to `[]` (see `scalar_or_nan` in
# the client).
sanitize(x::AbstractFloat) = isfinite(x) ? x : nothing
sanitize(x::AbstractArray) = [sanitize(e) for e in x]
sanitize(x) = x

"""
    KiteState(ss::SysState)

Compact projection of a `SysState` carrying ONLY the fields the Matlab/Python
REST clients need for plotting and control — far smaller than serializing the
full `SysState`. Returned directly by `/state`, and (augmented with
`n`/`rel_steering`) by `/step`. Keep this in sync with the fields the clients
actually read; add a field here rather than falling back to the whole state.

The V3 model has a single winch, so `l_tether`/`v_reelout`/`winch_force` are
taken as scalars (`first(...)` of the underlying 1-element vectors), unlike a
multi-tether model. Non-finite floats become JSON `null` via `sanitize`.
"""
struct KiteState
    time::Union{Float64, Nothing}
    elevation::Union{Float64, Nothing}
    azimuth::Union{Float64, Nothing}
    heading::Union{Float64, Nothing}
    v_app::Union{Float64, Nothing}
    AoA::Union{Float64, Nothing}
    l_tether::Union{Float64, Nothing}
    v_reelout::Union{Float64, Nothing}
    winch_force::Union{Float64, Nothing}
    var_15::Union{Float64, Nothing}
    var_16::Union{Float64, Nothing}
end

KiteState(ss::SysState) = KiteState(
    sanitize(ss.time), sanitize(ss.elevation), sanitize(ss.azimuth),
    sanitize(ss.heading), sanitize(ss.v_app), sanitize(ss.AoA),
    sanitize(first(ss.l_tether)), sanitize(first(ss.v_reelout)),
    sanitize(first(ss.winch_force)), sanitize(ss.var_15), sanitize(ss.var_16))

# Let JSON3 serialize `KiteState` field-by-field (→ a JSON object), so
# `/state` can return it directly.
JSON3.StructTypes.StructType(::Type{KiteState}) = JSON3.StructTypes.Struct()

"""
    full_state_dict(ss::SysState) -> Dict{String, Any}

Serialize EVERY field of a `SysState` into a JSON-ready `Dict`, keyed by field
name. Used by `/step` when the client passes `full_state=true`. All `SysState`
fields are numeric scalars or float vectors, so `sanitize` (which recurses into
arrays) maps non-finite floats to JSON `null` and leaves everything else intact.
"""
full_state_dict(ss::SysState) =
    Dict{String, Any}(string(f) => sanitize(getfield(ss, f)) for f in fieldnames(typeof(ss)))

json_response(status::Int, obj) =
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(obj))

parse_body(req) = isempty(req.body) ? Dict{Symbol, Any}() : JSON3.read(String(req.body))

# Coerce a decoded JSON value to Float64, or return `nothing` if it is not a
# number (JSON `null` decodes to `nothing`, strings/objects to non-`Real`).
# `Bool` is a `Real` in Julia, but `true`/`false` are not valid control inputs,
# so reject them too. Lets the handler answer 400 instead of throwing a 500.
to_float(x::Real) = Float64(x)
to_float(::Bool)  = nothing
to_float(::Any)   = nothing

# Fetch `key` from the payload as a Float64, falling back to `default` when the
# key is absent. Throws an `ArgumentError` (which the handler turns into a 400)
# when the key is present but not a number, so bad types get a clear message
# instead of a 500. Works on both JSON3.Object and Dict.
function getf(payload, key, default)
    haskey(payload, key) || return Float64(default)
    v = to_float(payload[key])
    v === nothing && throw(ArgumentError("$key must be a number"))
    return v
end

# Fetch `key` as a `Bool`, falling back to `default` when absent. Throws an
# `ArgumentError` (→ 400) when present but not a JSON boolean.
function get_bool(payload, key, default::Bool)
    haskey(payload, key) || return default
    v = payload[key]
    v isa Bool || throw(ArgumentError("$key must be true or false"))
    return v
end

# Fetch `key` as a positive `Int`, falling back to `default` when absent. Throws
# an `ArgumentError` (→ 400) when present but not a positive whole number.
function get_int(payload, key, default::Int)
    haskey(payload, key) || return default
    v = to_float(payload[key])
    (v === nothing || !isfinite(v) || v != floor(v) || v < 1) &&
        throw(ArgumentError("$key must be a positive integer"))
    return Int(v)
end

# Fetch `key` as a `String`, falling back to `default` when absent. Throws an
# `ArgumentError` (→ 400) when present but not a JSON string.
function get_string(payload, key, default::String)
    haskey(payload, key) || return default
    v = payload[key]
    v isa AbstractString || throw(ArgumentError("$key must be a string"))
    return String(v)
end

# Restrict a user-supplied filename-like value to a safe pattern (letters,
# digits, '.', '_', '-') so it can't escape a directory via path separators or
# ".." traversal when used to build a file path server-side.
is_safe_filename(name::AbstractString) =
    occursin(r"^[A-Za-z0-9._-]+$", name) && !(name in (".", ".."))

# ============================ Init machinery ========================== #

"""
    run_init(session, params)

Body of the background init task: run `init` with a tee'd logger so progress
lines land in the status buffer, then publish the result (or the failure) into
the session under its lock.
"""
function run_init(session::Session, params)
    collector = CollectLogger(Logging.Info, msg -> push_message!(session, msg))
    tee = TeeLogger(global_logger(), collector)
    try
        kite = with_logger(tee) do
            init(params.v_wind, params.l_tether;
                 depower_setpoint = params.depower_setpoint, sim_time = params.sim_time,
                 system_yaml = params.system_yaml, aero_mode = AERO_MODE)
        end
        lock(session.lock) do
            session.s = kite
            session.step_count = 0
            session.result = Dict{String, Any}(
                "l0"    => kite.sys_state.l_tether[1],
                "steps" => kite.steps,
                "dt"    => kite.dt,
            )
            session.state = "ready"
        end
        @info "Model ready: $(kite.steps) steps at dt=$(kite.dt)s"
    catch e
        errstr = sprint(showerror, e, catch_backtrace())
        lock(session.lock) do
            session.error = errstr
            session.state = "failed"
        end
        @error "Initialization failed" exception = (e, catch_backtrace())
    end
    return nothing
end

"""
    start_init!(session, params) -> Symbol

Reset the session and spawn the background init task. Returns `:accepted`, or
`:conflict` if an init is already running.
"""
function start_init!(session::Session, params)
    conflict = false
    lock(session.lock) do
        if session.state == "initializing"
            conflict = true
        else
            session.state = "initializing"
            session.error = nothing
            session.result = nothing
            session.s = nothing
            session.step_count = 0
            lock(() -> empty!(session.messages), session.msg_lock)
        end
    end
    conflict && return :conflict
    session.init_task = Threads.@spawn run_init(session, params)
    return :accepted
end

# ============================== Endpoints ============================= #

@post "/init" function (req)
    payload = try
        parse_body(req)
    catch
        return json_response(400, Dict("error" => "invalid JSON body"))
    end
    params = try
        (v_wind           = getf(payload, :v_wind, 10.0),
         l_tether         = getf(payload, :l_tether, 150.0),
         depower_setpoint = getf(payload, :depower_setpoint, 0.25),
         sim_time         = getf(payload, :sim_time, 10.0),
         system_yaml      = get_string(payload, :system_yaml, "system_cabauw.yaml"))
    catch e
        return json_response(400, Dict("error" => sprint(showerror, e)))
    end
    if !is_safe_filename(params.system_yaml) || !endswith(params.system_yaml, ".yaml")
        return json_response(400, Dict("error" =>
            "system_yaml must be a simple filename ending in .yaml"))
    end
    if start_init!(SESSION, params) == :conflict
        return json_response(409, Dict("error" => "an init is already running"))
    end
    return json_response(202, Dict("state" => "initializing"))
end

@get "/status" function (req)
    snap = lock(SESSION.lock) do
        Dict{String, Any}("state"  => SESSION.state,
                          "error"  => SESSION.error,
                          "result" => SESSION.result)
    end
    snap["messages"] = lock(() -> copy(SESSION.messages), SESSION.msg_lock)
    return json_response(200, snap)
end

@post "/step" function (req)
    payload = try
        parse_body(req)
    catch
        return json_response(400, Dict("error" => "invalid JSON body"))
    end
    return lock(SESSION.lock) do
        if SESSION.state != "ready"
            return json_response(409, Dict("error" => "session not ready",
                                           "state" => SESSION.state))
        end
        kite = SESSION.s
        remaining = kite.steps - SESSION.step_count
        if remaining <= 0
            return json_response(409, Dict("error" => "step budget exhausted; re-init with a larger sim_time",
                                           "steps" => kite.steps))
        end
        if !haskey(payload, :rel_depower) || !haskey(payload, :set_length)
            return json_response(400, Dict("error" => "rel_depower and set_length are required"))
        end
        rel_depower = to_float(payload[:rel_depower])
        set_length  = to_float(payload[:set_length])
        if rel_depower === nothing || set_length === nothing
            return json_response(400, Dict("error" => "rel_depower and set_length must be numbers"))
        end
        # Number of Julia integration steps to advance per REST call (default 2).
        # Batching amortizes the HTTP round-trip over several steps. Clamped to
        # the remaining step budget so we never overshoot `sim_time`.
        nreq = try
            get_int(payload, :steps, 2)
        catch e
            return json_response(400, Dict("error" => sprint(showerror, e)))
        end
        # When `full_state=true`, return every `SysState` field (see
        # `full_state_dict`) instead of the compact `KiteState` projection.
        full_state = try
            get_bool(payload, :full_state, false)
        catch e
            return json_response(400, Dict("error" => sprint(showerror, e)))
        end
        nsteps = min(nreq, remaining)
        # Steering channel: hold the given `rel_steering` (0.0 = parking)
        # constant across all sub-steps. Any steering logic lives on the
        # client, which passes the value it wants applied for this call.
        rel_steering = 0.0
        if haskey(payload, :rel_steering)
            rs = to_float(payload[:rel_steering])
            rs === nothing && return json_response(400,
                Dict("error" => "rel_steering must be a number"))
            if rs < -1.0 || rs > 1.0
                return json_response(400,
                    Dict("error" => "rel_steering must be in [-1.0, 1.0]"))
            end
            rel_steering = rs
        end
        # Only the LAST sub-step's outputs are returned; the intermediate states
        # are still logged server-side (`save_log`), so full resolution is
        # preserved in the saved run while the response stays small.
        try
            for _ in 1:nsteps
                step!(kite; rel_depower, rel_steering, set_length)
                SESSION.step_count += 1
            end
        catch e
            errstr = sprint(showerror, e, catch_backtrace())
            SESSION.state = "failed"
            SESSION.error = errstr
            return json_response(500, Dict("error" => errstr))
        end
        # Return the last state's fields at top level (so the client reads
        # `resp.time`, `resp.elevation`, …), plus how many steps were taken and the
        # steering value used. By default this is the compact `KiteState`
        # projection; with `full_state=true` it is every `SysState` field.
        resp = if full_state
            full_state_dict(kite.sys_state)
        else
            ks = KiteState(kite.sys_state)
            Dict{String, Any}(string(f) => getfield(ks, f) for f in fieldnames(KiteState))
        end
        resp["n"]            = nsteps
        resp["rel_steering"] = rel_steering
        return json_response(200, resp)
    end
end

@get "/state" function (req)
    return lock(SESSION.lock) do
        if SESSION.s === nothing || SESSION.step_count == 0
            return json_response(409, Dict("error" => "no step has produced a state yet"))
        end
        return json_response(200, KiteState(SESSION.s.sys_state))
    end
end

@post "/save_log" function (req)
    payload = try
        parse_body(req)
    catch
        return json_response(400, Dict("error" => "invalid JSON body"))
    end
    name = get_string(payload, :name, "tmp_run")
    # Restrict to a simple filename so the name can't escape the data directory
    # via path separators or ".." traversal (the log path is built from it below).
    if !is_safe_filename(name)
        return json_response(400, Dict("error" =>
            "name must be a simple filename (letters, digits, '.', '_', '-' only)"))
    end
    return lock(SESSION.lock) do
        if SESSION.s === nothing
            return json_response(409, Dict("error" => "no model to save"))
        end
        mkpath(get_data_path())
        save_log(SESSION.s.logger, name)
        path = joinpath(get_data_path(), name) * ".arrow"
        return json_response(200, Dict("path" => path))
    end
end

# ============================== Startup ============================== #

"""
    truthy(s) -> Bool

Parse an on/off env-var string. Anything in `0/false/no/off` (case-insensitive)
counts as off; everything else is on.
"""
truthy(s::AbstractString) = !(lowercase(strip(s)) in ("0", "false", "no", "off"))

function main()
    port = if length(ARGS) >= 1
        parse(Int, ARGS[1])
    else
        parse(Int, get(ENV, "V3_REST_PORT", "8080"))
    end
    # Per-request access logging (Oxygen's default `[ Info: ... "POST /step" 200`
    # line) costs a measurable amount per round-trip, which dominates the tight
    # one-step-per-request client loop. It is OFF by default to keep throughput
    # up; re-enable it with V3_REST_ACCESS_LOG=1 (or `true`/`yes`/`on`).
    access_log_on = truthy(get(ENV, "V3_REST_ACCESS_LOG", "0"))
    @info "Starting V3Kite REST server on http://127.0.0.1:$port (threads: $(Threads.nthreads()))"
    if Threads.nthreads() == 1
        @warn "Running single-threaded: /status will not report progress DURING init. " *
              "Start with `julia -t auto` for live init progress."
    end
    if access_log_on
        serve(; host = "127.0.0.1", port = port)
    else
        @info "Per-request access logging disabled (V3_REST_ACCESS_LOG=0)"
        serve(; host = "127.0.0.1", port = port, access_log = nothing)
    end
end

# Only auto-start when run as a script, not when included interactively.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
