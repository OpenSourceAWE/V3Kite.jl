### Changed
- BREAKING: `calc_heading(s::V3KITE)` returns the heading the model logs (`SysState.heading`): zero with the nose towards zenith, positive right-handed about the outward radial, so a nose to the right as seen from the ground station reads +90°. That is the opposite sign of what it returned before. The `upwind_dir_` and `neg_azimuth` keywords are gone.
