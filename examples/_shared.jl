# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Shared helpers for example scripts.
"""

const V3_TE_EDGE_SEGMENT_IDS = 20:28

"""
    scale_te_edge_rest_lengths!(sys; scale=0.95, segment_ids=V3_TE_EDGE_SEGMENT_IDS)

Scale trailing-edge wire rest lengths (`l0`) in a loaded system structure.
By default, applies to V3 trailing-edge segment IDs 20:28.
"""
function scale_te_edge_rest_lengths!(sys;
    scale=0.95,
    segment_ids=V3_TE_EDGE_SEGMENT_IDS)

    scale < 0 && throw(ArgumentError("scale must be non-negative, got $scale"))
    n_segments = length(sys.segments)

    for seg_id in segment_ids
        (1 <= seg_id <= n_segments) || throw(
            ArgumentError(
                "segment id $seg_id out of range 1:$n_segments"))
        sys.segments[seg_id].l0 *= scale
    end
    return nothing
end
