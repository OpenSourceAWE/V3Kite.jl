% Copyright (c) 2026 Uwe Fechner
% SPDX-License-Identifier: MPL-2.0
%
% Matlab client for the V3Kite REST interface, mirroring
% examples/simple_parking.jl: init the V3 parking model, poll until it is
% ready, run the parking loop one batch of steps per HTTP round-trip, save
% the log, and plot the collected quantities.
%
% Start the server first (in the repo root):
%     bin/run_server
%
% then run this script in Matlab.

base = "http://127.0.0.1:8080";
opts = weboptions("Timeout", 30, "MediaType", "application/json");

% ------------------------------- 1. init ------------------------------- %
PROJECT          = "system_cabauw.yaml";  % System project (see data/system_*.yaml)
V_WIND           = 10.0;   % m/s
TETHER_LENGTH    = 150.0;  % m
DEPOWER_SETPOINT = 0.25;   % rel_depower held during parking [-]
SIM_TIME         = 10.0;   % s
FULL_STATE       = false;  % request the full SysState from /step (else the
                           % compact KiteState projection)

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

% ----------------------------- 4. run loop ----------------------------- %
% Position mode: set_length holds the tether length at its initial value l0.
% Each /step call advances STEPS_PER_CALL Julia steps and returns only the LAST
% of those states; raising it amortizes the HTTP overhead but coarsens the
% sampling of the arrays below (the full-resolution run is still saved
% server-side).
STEPS_PER_CALL = 2;

n_calls     = ceil(steps / STEPS_PER_CALL);   % one recorded sample per call
time        = zeros(n_calls, 1);
elevation   = zeros(n_calls, 1);   % rad
heading     = zeros(n_calls, 1);   % rad
AoA         = zeros(n_calls, 1);   % rad
v_reelout   = zeros(n_calls, 1);   % m/s
winch_force = zeros(n_calls, 1);   % N
ld_wing     = zeros(n_calls, 1);   % var_15
ld_eff      = zeros(n_calls, 1);   % var_16

idx        = 0;      % number of Julia steps taken so far
rec        = 0;      % number of samples recorded so far
last_print = 0;      % step index at the last progress line
last_batch = tic;    % wall-clock timer since the last progress line
while idx < steps
    n_req = min(STEPS_PER_CALL, steps - idx);
    resp = webwrite(base + "/step", struct( ...
        "rel_depower", DEPOWER_SETPOINT, ...
        "set_length",  l0, ...
        "steps",       n_req, ...
        "full_state",  FULL_STATE), opts);
    idx = idx + double(resp.n);
    rec = rec + 1;
    % resp holds only the last sub-step s state at top level.
    time(rec)        = resp.time;
    elevation(rec)   = resp.elevation;
    heading(rec)      = resp.heading;
    AoA(rec)          = resp.AoA;
    v_reelout(rec)    = resp.v_reelout;
    winch_force(rec)  = resp.winch_force;
    ld_wing(rec)      = resp.var_15;
    ld_eff(rec)       = resp.var_16;

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

% ----------------------------- 5. save log ----------------------------- %
saved = webwrite(base + "/save_log", struct("name", "tmp_run"), opts);
fprintf("Log saved to %s\n", string(saved.path));

% ------------------------------- 6. plot ------------------------------- %
figure;
subplot(3, 2, 1);
plot(time, v_reelout, "LineWidth", 1.2);
grid on; ylabel("v_{ro} [m/s]");
title("V3Kite parking maneuver (via REST)");

subplot(3, 2, 2);
plot(time, winch_force, "LineWidth", 1.2);
grid on; ylabel("F_t [N]");

subplot(3, 2, 3);
plot(time, rad2deg(elevation), "LineWidth", 1.2);
grid on; ylabel("elevation [deg]");

subplot(3, 2, 4);
plot(time, rad2deg(heading), "LineWidth", 1.2);
grid on; ylabel("heading [deg]");

subplot(3, 2, 5);
plot(time, rad2deg(AoA), "LineWidth", 1.2);
grid on; ylabel("AoA [deg]"); xlabel("time [s]");

subplot(3, 2, 6);
plot(time, ld_wing, time, ld_eff, "LineWidth", 1.2);
grid on; ylabel("L/D [-]"); xlabel("time [s]");
legend("L/D_{wing}", "L/D_{eff}");
