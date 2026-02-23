
[WORK IN PROGRESS] 

# V3Kite.jl

Julia package for simulation and validation of the TU Delft V3 ram-air kite.

## Overview

V3Kite.jl provides calibration functions, model setup utilities, CSV replay capabilities, and simulation functions built on top of [SymbolicAWEModels.jl](https://github.com/OpenSourceAWE/SymbolicAWEModels.jl) for the TU Delft V3 kite.

## Installation
Inside the REPL:
```julia
using Pkg
Pkg.add(url="https://github.com/OpenSourceAWE/V3Kite.jl")
```

## Quick Start

```julia
using V3Kite
using GLMakie  # Optional, for visualization

# Create a simulation configuration
config = V3SimConfig(
    sim_time = 60.0,      # seconds
    fps = 60,             # frames per second
    v_wind = 10.0,        # wind speed [m/s]
    up = 40.0,            # depower percentage [0-100]
    us = 0.0,             # steering percentage [-100, 100]
    wing_type = REFINE,   # or QUATERNION
)

# Run simulation
sam, syslog, tape_data = run_v3_simulation(config)

# Visualize (requires GLMakie)
fig = plot(sam.sys_struct, syslog)
display(fig)
```

## Calibration System

V3Kite uses a base + delta calibration pattern:

### Official KCU Measurements (Base Values)

| Parameter | Base Value | Description |
|-----------|------------|-------------|
| `V3_STEERING_L0_BASE` | 1.6 m | Neutral steering tape length |
| `V3_DEPOWER_L0_BASE` | 0.2 m | Neutral depower tape length |
| `V3_STEERING_GAIN` | 1.4 m | Max differential at 100% steering |
| `V3_DEPOWER_GAIN` | 5.0 m | Depower range for 0-100% |

### Calibration Delta

The default delta (`V3_DEFAULT_DELTA = -0.2`) accounts for the empirical difference between measured KCU values and effective simulation values.

### Effective Values

```julia
V3_STEERING_L0 = V3_STEERING_L0_BASE + V3_DEFAULT_DELTA  # = 1.4 m
V3_DEPOWER_L0 = V3_DEPOWER_L0_BASE + V3_DEFAULT_DELTA    # = 0.0 m
```

### Conversion Functions

All conversion functions accept an optional `delta` parameter:

```julia
# Using default delta
L_left, L_right = steering_percentage_to_lengths(50.0)
L_depower = depower_percentage_to_length(40.0)

# Using custom delta
L_left, L_right = steering_percentage_to_lengths(50.0; delta=0.0)  # Use base values

# Inverse conversions
pct = steering_length_to_percentage(L_left, L_right)
pct = depower_length_to_percentage(L_depower)
```

## Configuration

### V3SimConfig

```julia
V3SimConfig(
    # Geometry files (relative to data directory)
    struc_yaml_path = "struc_geometry_depower0.0_tip0.4_te0.95.yaml",
    aero_yaml_path = "aero_geometry_depower0.0_tip0.4_te0.95.yaml",
    vsm_settings_path = "vsm_settings_reduced_for_coupling.yaml",

    # Simulation parameters
    sim_time = 60.0,           # Duration [s]
    fps = 60,                  # Logging frequency

    # Wind parameters
    v_wind = 10.0,             # Wind speed [m/s]
    upwind_dir = -90.0,        # Wind direction [deg]

    # Control parameters
    up = 40.0,                 # Depower [0-100%]
    us = 0.0,                  # Steering [-100, 100%]
    tether_length = 250.0,     # [m]

    # Model options
    wing_type = REFINE,        # REFINE or QUATERNION
    n_panels = 36,             # VSM panel count
    brake = true,              # Winch brake engaged
)
```

## Data Directory

V3Kite includes bundled V3 kite geometry and configuration files. Access with:

```julia
data_path = v3_data_path()
```

## Examples

Run the included example:

```bash
julia --project=examples examples/v3kite.jl
```

## Visualization Extension

When GLMakie is loaded, additional visualization functions become available:

```julia
using V3Kite
using GLMakie

# Plot wing points in body frame
fig = plot_body_frame_local(sys_struct; dir=:front)
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Related Packages

- [SymbolicAWEModels.jl](https://github.com/aenarete/SymbolicAWEModels.jl) - Symbolic kite modeling
- [VortexStepMethod.jl](https://github.com/aenarete/VortexStepMethod.jl) - Aerodynamic calculations
- [KiteUtils.jl](https://github.com/aenarete/KiteUtils.jl) - Shared utilities

## License

MPL-2.0
