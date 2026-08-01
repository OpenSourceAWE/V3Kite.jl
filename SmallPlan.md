# Plan for small improvements towards figure 8 flight

## First campaign
- use the function turn_rate_coeffs to determine c1 and c2 and delay
- don't run for longer than 30s
- don't iterate automatically
- run the script simple_fig8.jl after each change of the parameters
- run it using the ex tool from Kaimon
- include simple_fig8_plots.jl at the end of simple_fig8.jl
- NEVER run julia from the command line

## TODO - DONE -
- make the required adaptions, run once, plot the results and wait

## Next steps
- get log file from working controller and plot it