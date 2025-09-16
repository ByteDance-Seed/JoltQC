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
COORD_STRIDE = 4  # Basis coordinate stride for memory optimization
PRIM_STRIDE = (
    (2 * NPRIM_MAX + 3) // 4 * 4
)  # Coefficient/exponent stride for memory optimization

# Tile size for block-based algorithms
TILE = 4

MAX_SMEM = 48 * 1024  # Maximum shared memory per block in bytes

__all__ = ["LMAX", "NPRIM_MAX", "COORD_STRIDE", "PRIM_STRIDE", "TILE"]
