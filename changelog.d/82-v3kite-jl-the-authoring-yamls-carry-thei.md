### Added
- The PSM and beam structure geometries carry a `canopy_faces` table (`name, wing, points`) of the canopy's quads between adjacent stations, and the SurfplanAdapter writes it for the beam. The model build ignores it.
