% Copyright (c) 2026 Uwe Fechner
% SPDX-License-Identifier: MPL-2.0
%
% Matlab client for the V3Kite REST interface, mirroring
% examples/simple_sinus.jl: init the V3 parking model, poll until ready, then
% track a sinusoidal heading setpoint on the settled, depowered wing. The
% heading PID itself has no server-side counterpart (the REST interface only
% takes an already-computed `rel_steering`), so it is re-implemented here as
% a direct translation of DiscretePIDs.DiscretePID (standard form, see
% https://github.com/JuliaControl/DiscretePIDs.jl) with the same gains as
% simple_sinus.jl, driving `rel_steering` each call.
%
% Unlike simple_parking_client.m, /step is called with steps=1: the PID needs
% a fresh heading measurement every physics step, so the control input can't
% be held constant over a multi-step batch the way parking's constant
% rel_steering/depower can.
%
% Start the server first (in the repo root):
%     bin/run_server
%
% then run this script in Matlab. Saves the run to "tmp_sinus" (kept separate
% from simple_parking_client.m's "tmp_run") and prints the heading tracking
% RMS error over the settled part (skipping the first period), mirroring
% simple_sinus.jl.

base = "http://127.0.0.1:8080";
opts = weboptions("Timeout", 30, "MediaType", "application/json");

% ------------------------------- 1. init ------------------------------- %
PROJECT          = "system_cabauw.yaml";  % System project (see data/system_*.yaml)
V_WIND           = 10.0;   % m/s
TETHER_LENGTH    = 150.0;  % m
DEPOWER_SETPOINT = 0.25;   % rel_depower held during the run [-]
SIM_TIME         = 120.0;  % s
MAX_HEADING      = 40.0;   % Heading setpoint amplitude [deg]
HEADING_PERIOD   = 30.0;   % Heading setpoint period [s]

% Heading PID gains (output is rel_steering, dimensionless, -1..1). Must
% match the constants in examples/simple_sinus.jl.
HEADING_P           = 1.1;
HEADING_D           = 0.15;
HEADING_D_RAMP_TIME = 30.0;  % ramp HEADING_D down to 0 over this many seconds
MAX_STEERING        = 0.175;
% (HEADING_I = false in simple_sinus.jl, i.e. the integral term is disabled;
% that is folded directly into the PID state below rather than modeled.)

fprintf("Requesting init...\n");
webwrite(base + "/init", struct( ...
    "v_wind",           V_WIND, ...
    "l_tether",         TETHER_LENGTH, ...
    "depower_setpoint", DEPOWER_SETPOINT, ...
    "sim_time",         SIM_TIME, ...
    "system_yaml",      PROJECT), opts);

% ---------------------------- 2. poll status --------------------------- %
printed = 0;                 % number of status messages already printed
result  = [];
while true
    status = webread(base + "/status", opts);
    msgs = status.messages;
    for k = (printed + 1):numel(msgs)
        fprintf("  [init] %s\n", string(msgs{k}));
    end
    printed = numel(msgs);

    if strcmp(status.state, "ready")
        result = status.result;
        break;
    elseif strcmp(status.state, "failed")
        error("Initialization failed:\n%s", status.error);
    end
    pause(0.5);
end

% -------------------- 3. read the driving setpoints -------------------- %
l0    = result.l0;
steps = double(result.steps);
dt    = result.dt;
fprintf("Ready: %d steps, l0=%.2f m, dt=%.4f s\n", steps, l0, dt);

% ------------------- 4. discrete PID (DiscretePIDs.jl) ------------------ %
% Standard-form controller matching DiscretePIDs.DiscretePID with Ti=false
% (integral disabled, so bi=ar=0 and the I state never leaves 0), wp=1,
% wd=0, N=10 (library defaults). Td is ramped down each step exactly like
% `set_Td!` in simple_sinus.jl, and ad/bd are recomputed from it every step.
pid.K    = HEADING_P;
pid.N    = 10.0;
pid.wp   = 1.0;
pid.wd   = 0.0;
pid.umin = -MAX_STEERING;
pid.umax =  MAX_STEERING;
pid.Ts   = dt;
pid.I    = 0.0;   % stays 0: Ti=false => bi=ar=0 in DiscretePIDs
pid.D    = 0.0;
pid.yold = 0.0;

max_heading_rad = deg2rad(MAX_HEADING);
angular_freq    = 2 * pi / HEADING_PERIOD;

% ----------------------------- 5. run loop ----------------------------- %
n_calls     = steps;   % one recorded sample per Julia step (steps=1 per call)
time        = zeros(n_calls, 1);
bearing     = zeros(n_calls, 1);   % rad, heading setpoint (computed client-side)
elevation   = zeros(n_calls, 1);   % rad
azimuth     = zeros(n_calls, 1);   % rad
heading     = zeros(n_calls, 1);   % rad
AoA         = zeros(n_calls, 1);   % rad
rel_steer   = zeros(n_calls, 1);   % rel_steering command [-1..1]
ld_wing     = zeros(n_calls, 1);   % var_15
ld_eff      = zeros(n_calls, 1);   % var_16

idx        = 0;      % number of Julia steps taken so far
% The REST interface has no way to read the settled heading before the first
% /step call (/state requires step_count > 0), so the first PID update uses
% y=0 as a stand-in for the (typically near-zero) settled heading; this only
% affects the very first control sample.
y_meas     = 0.0;
last_print = 0;      % step index at the last progress line
last_batch = tic;    % wall-clock timer since the last progress line
while idx < steps
    t = (idx + 1) * dt;   % time after the upcoming step
    target = max_heading_rad * sin(angular_freq * t);

    % ramp_factor(t, 0, HEADING_D_RAMP_TIME), then set_Td! on the PID.
    Td = HEADING_D * (1 - min(max(t / HEADING_D_RAMP_TIME, 0.0), 1.0));
    pid.ad = Td / (Td + pid.N * pid.Ts);
    pid.bd = pid.K * pid.N * pid.ad;

    % calculate_control!(pid, target, y_meas, 0.0)
    P = pid.K * (pid.wp * target - y_meas);
    e = pid.wd * target - y_meas;
    pid.D = pid.ad * pid.D + pid.bd * (e - pid.yold);
    v = P + pid.I + pid.D;
    rel_steering = min(max(v, pid.umin), pid.umax);
    pid.yold = e;

    resp = webwrite(base + "/step", struct( ...
        "rel_depower",  DEPOWER_SETPOINT, ...
        "rel_steering", rel_steering, ...
        "set_length",   l0, ...
        "steps",        1), opts);
    idx = idx + double(resp.n);

    time(idx)      = resp.time;
    bearing(idx)   = target;
    elevation(idx) = resp.elevation;
    azimuth(idx)   = resp.azimuth;
    heading(idx)   = resp.heading;
    AoA(idx)       = resp.AoA;
    rel_steer(idx) = resp.rel_steering;
    % L/D is NaN whenever the wing is unloaded (step!'s drag_floor gate), and
    % the server sends non-finite floats as JSON null, which jsondecode turns
    % into []. Assigning [] to one element would DELETE it and shift every
    % later sample, so map the gap back to NaN here.
    ld = resp.var_15; if isempty(ld), ld = NaN; end
    ld_wing(idx)   = ld;
    ld = resp.var_16; if isempty(ld), ld = NaN; end
    ld_eff(idx)    = ld;

    % Feed the just-measured heading back for the next iteration's PID call.
    y_meas = resp.heading;

    % Progress, mirroring the server-side @info in step!: every ~100 steps,
    % report the realtime factor over the interval (here it includes the HTTP
    % round-trip, so it reflects the client-observed throughput).
    if idx - last_print >= 100
        realtime_factor = ((idx - last_print) * dt) / toc(last_batch);
        fprintf("step %d / %d, %.2f times realtime\n", idx, steps, realtime_factor);
        last_print = idx;
        last_batch = tic;
    end
end

% ----------------------------- 6. save log ----------------------------- %
saved = webwrite(base + "/save_log", struct("name", "tmp_sinus"), opts);
fprintf("Log saved to %s\n", string(saved.path));

% -------------------- 7. tracking error over settled part --------------- %
settled = time >= HEADING_PERIOD;
if any(settled)
    track_err = rad2deg(heading(settled) - bearing(settled));
    fprintf("Heading tracking RMS error (t >= %.0f s): %.2f deg\n", ...
        HEADING_PERIOD, sqrt(mean(track_err .^ 2)));
end

% ------------------------------- 8. plot -------------------------------- %
figure;
subplot(3, 2, 1);
plot(time, rad2deg(elevation), "LineWidth", 1.2);
grid on; ylabel("elevation [deg]");
title("V3Kite sinusoidal heading tracking (via REST)");

subplot(3, 2, 2);
plot(time, rad2deg(azimuth), "LineWidth", 1.2);
grid on; ylabel("azimuth [deg]");

subplot(3, 2, 3);
plot(time, rad2deg(heading), time, rad2deg(bearing), "LineWidth", 1.2);
grid on; ylabel("heading [deg]");
legend("\psi", "\psi_{ref}");

subplot(3, 2, 4);
plot(time, 100.0 * rel_steer, "LineWidth", 1.2);
grid on; ylabel("u_s [%]");

subplot(3, 2, 5);
plot(time, rad2deg(AoA), "LineWidth", 1.2);
grid on; ylabel("AoA [deg]"); xlabel("time [s]");

subplot(3, 2, 6);
plot(time, ld_wing, time, ld_eff, "LineWidth", 1.2);
grid on; ylabel("L/D [-]"); xlabel("time [s]");
legend("L/D_{wing}", "L/D_{eff}");
