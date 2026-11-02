### Added
- `papers/beam_v3/`: a paper on the beam V3 model, with the scripts that run its simulations and draw its figures.
- `beam_rigidities(sys)` lists the span position, length, radius and rigidities of every beam element, and `plot_beam_rigidities(sys)` plots EI and GJ along the span.
- `fly_heading_sine(sam, sys, project)` flies the heading-sine manoeuvre of `examples/v3kite.jl` and returns its log and wall time.
- `build_v3_model` takes a `cache_path` for the model binary, the settled state and the settling log.
