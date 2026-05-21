# SPDX-FileCopyrightText: 2025 Jelle Poland, Bart van de Lint, Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

using GLMakie
using V3Kite

set_data_path(joinpath(dirname(@__DIR__), "data"))
include(joinpath(dirname(@__DIR__), "examples", "v3kite.jl"))
