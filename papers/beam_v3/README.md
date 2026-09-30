# A dynamic beam-element model of a leading-edge inflatable kite

The paper describing V3Kite's beam model of the TU Delft V3 kite: how its leading edge and struts are Timoshenko beams coupled to the canopy and bridle, where its structural parameters come from, and how it flies against the particle model of the same kite.

The figures and the numbers in the text are generated, not tracked. From the repository root, in Julia:

```julia
include("papers/beam_v3/run_simulations.jl")
```

This settles and flies both models, which takes about half an hour, and writes `figures/` and `results.tex`. It settles from a cache of its own under `output/`, so what it reports does not depend on a settled state another run left behind. Then compile `beam_v3.tex`:

```sh
latexmk -pdf beam_v3.tex
```
