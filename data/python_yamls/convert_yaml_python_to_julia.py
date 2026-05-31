#!/usr/bin/env python3
"""Convert the old Python structural geometry YAML to the Julia format.

By default this reads:
  - struc_geometry_python_format.yaml
  - struc_geometry_julia_format.yaml

and writes:
  - struc_geometry_julia_generated.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "struc_geometry_python_format.yaml"
DEFAULT_TEMPLATE = SCRIPT_DIR / "struc_geometry_julia_format.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "struc_geometry_julia_generated.yaml"


def flow_seq(values: list[Any] | tuple[Any, ...]) -> CommentedSeq:
    """Return a YAML flow-style sequence, recursively for nested lists."""
    seq = CommentedSeq()
    for value in values:
        if isinstance(value, (list, tuple)):
            seq.append(flow_seq(value))
        else:
            seq.append(value)
    seq.fa.set_flow_style()
    return seq


def as_float(value: Any) -> float:
    return float(value)


def clean_float(value: float) -> float:
    """Avoid long floating-point artifacts after mass accumulation."""
    return round(float(value), 12)


def as_int(value: Any) -> int:
    return int(value)


def as_name(value: Any) -> str:
    return str(value)


def load_yaml(path: Path) -> Any:
    yaml = YAML(typ="rt")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream)


def dump_yaml(path: Path, data: Any) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(data, stream)


def rows_by_name(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    headers = [as_name(header) for header in section["headers"]]
    rows: dict[str, dict[str, Any]] = {}
    for row in section["data"]:
        item = dict(zip(headers, row))
        rows[as_name(item["name"])] = item
    return rows


def old_node_to_julia_point(old_node_idx: Any) -> int:
    # Old format uses 0 for the KCU. Julia point ids are 1-based.
    return as_int(old_node_idx) + 1


def compression_fraction(linktype: Any, element_kind: str) -> float:
    linktype = as_name(linktype)
    if linktype == "default":
        return 1.0
    if linktype in {"noncompressive", "pulley"}:
        if element_kind == "wing":
            return 0.1
        if element_kind == "bridle":
            return 0.01
    raise ValueError(f"Unsupported {element_kind} linktype: {linktype!r}")


def template_point_by_type(points: list[Any], point_type: str, *, skip_idx: int | None = None) -> Any:
    for row in points:
        if as_name(row[2]) == point_type and (skip_idx is None or as_int(row[0]) != skip_idx):
            return row
    raise ValueError(f"Template has no point row of type {point_type!r}")


def make_point_row(template_row: Any, idx: int, position: list[float], extra_mass: float) -> CommentedSeq:
    row = list(template_row)
    row[0] = idx
    row[1] = flow_seq([clean_float(value) for value in position])
    row[5] = clean_float(extra_mass)
    return flow_seq(row)


def update_materials_from_python(source: dict[str, Any], target: dict[str, Any]) -> None:
    if "dyneema" not in source or "materials" not in target:
        return

    dyneema = source["dyneema"]
    updated_row = flow_seq(
        [
            "dyneema",
            as_float(dyneema["youngs_modulus"]),
            as_float(dyneema["density"]),
            as_float(dyneema["damping_per_stiffness"]),
        ]
    )

    material_rows = target["materials"]["data"]
    for idx, row in enumerate(material_rows):
        if as_name(row[0]) == "dyneema":
            material_rows[idx] = updated_row
            return
    material_rows.append(updated_row)


def build_segments_and_point_masses(source: dict[str, Any]) -> tuple[CommentedSeq, CommentedSeq, dict[int, float]]:
    wing_elements = rows_by_name(source["wing_elements"])
    bridle_elements = rows_by_name(source["bridle_elements"])

    point_extra_masses: dict[int, float] = defaultdict(float)
    segment_rows = CommentedSeq()
    pulley_rows = CommentedSeq()
    segment_rows.fa.set_block_style()
    pulley_rows.fa.set_block_style()

    segment_idx = 1
    pulley_idx = 1

    for connection in source["wing_connections"]["data"]:
        name = as_name(connection[0])
        old_i = as_int(connection[1])
        old_j = as_int(connection[2])
        element = wing_elements[name]

        l0 = as_float(element["l0"])
        stiffness = as_float(element["k"])
        damping = as_float(element["c"])
        mass = as_float(element["m"])
        compression_frac = compression_fraction(element["linktype"], "wing")

        segment_rows.append(
            flow_seq(
                [
                    segment_idx,
                    old_node_to_julia_point(old_i),
                    old_node_to_julia_point(old_j),
                    l0,
                    0.0,
                    stiffness,
                    damping,
                    compression_frac,
                ]
            )
        )
        point_extra_masses[old_i] += mass / 2.0
        point_extra_masses[old_j] += mass / 2.0
        segment_idx += 1

    pulley_mass = as_float(source.get("pulley_mass", 0.0))
    for connection in source["bridle_connections"]["data"]:
        name = as_name(connection[0])
        old_nodes = [as_int(node) for node in connection[1:]]
        element = bridle_elements[name]

        material = as_name(element["material"])
        diameter_mm = as_float(element["d"]) * 1000.0
        compression_frac = compression_fraction(element["linktype"], "bridle")

        if len(old_nodes) == 3:
            # A pulley row [name, a, b, c] is one continuous line a -> b -> c.
            # The middle node b is the pulley point and the rest length is split
            # evenly over the two generated segments.
            segment_l0 = as_float(element["l0"]) / 2.0
            first_segment_idx = segment_idx
            for old_i, old_j in zip(old_nodes, old_nodes[1:]):
                segment_rows.append(
                    flow_seq(
                        [
                            segment_idx,
                            old_node_to_julia_point(old_i),
                            old_node_to_julia_point(old_j),
                            segment_l0,
                            diameter_mm,
                            material,
                            "nothing",
                            compression_frac,
                        ]
                    )
                )
                segment_idx += 1

            pulley_rows.append(flow_seq([pulley_idx, first_segment_idx, first_segment_idx + 1, "DYNAMIC"]))
            point_extra_masses[old_nodes[1]] += pulley_mass
            pulley_idx += 1
        elif len(old_nodes) == 2:
            segment_rows.append(
                flow_seq(
                    [
                        segment_idx,
                        old_node_to_julia_point(old_nodes[0]),
                        old_node_to_julia_point(old_nodes[1]),
                        as_float(element["l0"]),
                        diameter_mm,
                        material,
                        "nothing",
                        compression_frac,
                    ]
                )
            )
            segment_idx += 1
        else:
            raise ValueError(f"Expected 2 or 3 nodes in bridle connection {connection!r}")

    return segment_rows, pulley_rows, point_extra_masses


def build_points(source: dict[str, Any], target: dict[str, Any], point_extra_masses: dict[int, float]) -> tuple[CommentedSeq, int, int]:
    template_points = target["points"]["data"]
    old_ground_idx = as_int(template_point_by_type(template_points, "STATIC")[0])

    kcu_template = template_points[0]
    wing_template = template_point_by_type(template_points, "WING")
    bridle_template = template_point_by_type(template_points, "DYNAMIC", skip_idx=as_int(kcu_template[0]))
    ground_template = template_point_by_type(template_points, "STATIC")

    point_rows = CommentedSeq()
    point_rows.fa.set_block_style()

    kcu_position = [as_float(value) for value in source.get("bridle_point_node", kcu_template[1])]
    kcu_mass = as_float(source.get("kcu_mass", kcu_template[5]))
    point_rows.append(make_point_row(kcu_template, 1, kcu_position, kcu_mass + point_extra_masses.get(0, 0.0)))

    all_old_point_ids = [0]
    for particle in source["wing_particles"]["data"]:
        old_idx = as_int(particle[0])
        position = [as_float(particle[1]), as_float(particle[2]), as_float(particle[3])]
        extra_mass = point_extra_masses.get(old_idx, 0.0)
        point_rows.append(make_point_row(wing_template, old_node_to_julia_point(old_idx), position, extra_mass))
        all_old_point_ids.append(old_idx)

    for particle in source["bridle_particles"]["data"]:
        old_idx = as_int(particle[0])
        position = [as_float(particle[1]), as_float(particle[2]), as_float(particle[3])]
        extra_mass = point_extra_masses.get(old_idx, 0.0)
        point_rows.append(make_point_row(bridle_template, old_node_to_julia_point(old_idx), position, extra_mass))
        all_old_point_ids.append(old_idx)

    new_ground_idx = max(old_node_to_julia_point(old_idx) for old_idx in all_old_point_ids) + 1
    point_rows.append(make_point_row(ground_template, new_ground_idx, [as_float(value) for value in ground_template[1]], as_float(ground_template[5])))

    return point_rows, old_ground_idx, new_ground_idx


def replace_ground_references(target: dict[str, Any], old_ground_idx: int, new_ground_idx: int) -> None:
    if old_ground_idx == new_ground_idx:
        return

    if "tethers" in target:
        for row in target["tethers"].get("data", []):
            if as_int(row[1]) == old_ground_idx:
                row[1] = new_ground_idx
            if as_int(row[2]) == old_ground_idx:
                row[2] = new_ground_idx

    if "winches" in target:
        for row in target["winches"].get("data", []):
            if len(row) > 2 and as_int(row[2]) == old_ground_idx:
                row[2] = new_ground_idx

    if "transforms" in target:
        for row in target["transforms"].get("data", []):
            if row.get("base_point_idx") == old_ground_idx:
                row["base_point_idx"] = new_ground_idx
                row.yaml_add_eol_comment(
                    f"Reference to point {new_ground_idx} (ground attachment)",
                    key="base_point_idx",
                )


def validate_julia_structure(target: dict[str, Any]) -> None:
    point_ids = {as_int(row[0]) for row in target["points"]["data"]}
    segment_ids = {as_int(row[0]) for row in target["segments"]["data"]}

    for row in target["segments"]["data"]:
        segment_idx = as_int(row[0])
        for point_idx in (as_int(row[1]), as_int(row[2])):
            if point_idx not in point_ids:
                raise ValueError(f"Segment {segment_idx} references missing point {point_idx}")

    for row in target["pulleys"]["data"]:
        pulley_idx = as_int(row[0])
        for segment_idx in (as_int(row[1]), as_int(row[2])):
            if segment_idx not in segment_ids:
                raise ValueError(f"Pulley {pulley_idx} references missing segment {segment_idx}")


def convert_python_yaml_to_julia(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    template_path: Path = DEFAULT_TEMPLATE,
) -> None:
    source = load_yaml(input_path)
    target = load_yaml(template_path)

    update_materials_from_python(source, target)
    segment_rows, pulley_rows, point_extra_masses = build_segments_and_point_masses(source)
    point_rows, old_ground_idx, new_ground_idx = build_points(source, target, point_extra_masses)

    target["points"]["data"] = point_rows
    target["segments"]["data"] = segment_rows
    target["pulleys"]["data"] = pulley_rows
    replace_ground_references(target, old_ground_idx, new_ground_idx)
    validate_julia_structure(target)

    dump_yaml(output_path, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Python-format structural YAML")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Julia-format template YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated Julia-format YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_python_yaml_to_julia(args.input, args.output, args.template)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
