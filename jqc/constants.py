# Copyright 2025 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Global constants used throughout the jqc package.
"""

# Maximum angular momentum supported
LMAX = 4

# Maximum number of primitive Gaussians per contracted Gaussian
NPRIM_MAX = 3

# Memory stride constants for optimal memory layout
# Choose strides so each shell record is at least 64B aligned for both fp32/fp64
# - COORD_STRIDE scalars per shell (x,y,z plus padding)
#   16 floats  -> 64B, 16 doubles -> 128B (>=64B)
COORD_STRIDE = 16
# - PRIM_STRIDE counts scalars for (c,e) interleaved storage; device uses pairs
#   Make scalars a multiple of 16 => pairs multiple of 8
#   8 pairs: 8*8B=64B (fp32), 8*16B=128B (fp64)
PRIM_STRIDE = ((2 * NPRIM_MAX + 15) // 16) * 16

# Tile size for block-based algorithms
TILE = 4

MAX_SMEM = 48 * 1024  # Maximum shared memory per block in bytes

__all__ = ["LMAX", "NPRIM_MAX", "COORD_STRIDE", "PRIM_STRIDE", "TILE"]
