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
Generate data for rys quadrature
"""

# Change the path of GPU4PySCF if needed
with open("/home/xiaojie/Documents/gpu4pyscf/gpu4pyscf/lib/gvhf-rys/rys_roots_dat.cu") as f:
    rys_roots_data = f.read()

import re
import numpy as np

# Match device array declarations and their full initializer
pattern = re.compile(r"__device__\s+double\s+(\w+)\s*\[\]\s*=\s*\{(.*?)\};", re.DOTALL)

arrays = pattern.findall(rys_roots_data)
parsed_data = {}

for array_name, array_body in arrays:
    # Split into sections based on nroots comments
    blocks = re.split(r"//\s*nroots\s*=\s*(\d+)", array_body)
    data_by_nroots = {}

    i = 1
    while i < len(blocks):
        nroots = int(blocks[i])
        values_block = blocks[i + 1]

        # Extract floating point values (both pos/neg, scientific)
        values = re.findall(r"[-+]?\d*\.\d+e[-+]?\d+", values_block)
        data_by_nroots[nroots] = list(map(float, values))
        i += 2

    parsed_data[array_name] = data_by_nroots

data_name = parsed_data.keys()

# Constants for ROOT_RW_DATA layout
DEGREE = 13
DEGREE1 = 14  # DEGREE + 1
INTERVALS = 40

def reshape_root_rw_data(data, nroots):
    """
    Reshape ROOT_RW_DATA from [NROOTS, 2, DEGREE1, INTERVALS] to [NROOTS, INTERVALS, DEGREE1, 2]
    This interleaves root and weight coefficients for vectorized float4/double4 loads.

    Current layout: all roots, then all weights
    New layout: root and weight coefficients interleaved for each degree
    """
    total_expected = nroots * 2 * DEGREE1 * INTERVALS
    if len(data) != total_expected:
        raise ValueError(f"Expected {total_expected} values, got {len(data)}")

    # Reshape to [nroots, 2, DEGREE1, INTERVALS]
    data_array = np.array(data).reshape(nroots, 2, DEGREE1, INTERVALS)

    # Transpose to [nroots, INTERVALS, DEGREE1, 2]
    # This interleaves root and weight coefficients
    data_transposed = np.transpose(data_array, (0, 3, 2, 1))

    # Flatten back to 1D
    return data_transposed.flatten().tolist()

# nroots = 10 is missing
for i in range(1, 10):
    with open(f"rys_root{i}.cu", "w+") as f:
        # First, create interleaved ROOT_SMALLX_DATA
        if all(var in parsed_data for var in ["ROOT_SMALLX_R0", "ROOT_SMALLX_R1", "ROOT_SMALLX_W0", "ROOT_SMALLX_W1"]):
            f.write("constexpr __device__ DataType ROOT_SMALLX_DATA[] = {\n")
            # Interleave: [R0_0, R1_0, W0_0, W1_0, R0_1, R1_1, W0_1, W1_1, ...]
            for idx in range(i):  # nroots
                r0 = parsed_data["ROOT_SMALLX_R0"][i][idx]
                r1 = parsed_data["ROOT_SMALLX_R1"][i][idx]
                w0 = parsed_data["ROOT_SMALLX_W0"][i][idx]
                w1 = parsed_data["ROOT_SMALLX_W1"][i][idx]
                f.write(f"  {r0:23.16e}, {r1:23.16e}, {w0:23.16e}, {w1:23.16e},\n")
            f.write("};\n\n")

        # Create interleaved ROOT_LARGEX_DATA
        if all(var in parsed_data for var in ["ROOT_LARGEX_R_DATA", "ROOT_LARGEX_W_DATA"]):
            f.write("constexpr __device__ DataType ROOT_LARGEX_DATA[] = {\n")
            # Interleave: [R_0, W_0, R_1, W_1, ...]
            for idx in range(i):  # nroots
                r = parsed_data["ROOT_LARGEX_R_DATA"][i][idx]
                w = parsed_data["ROOT_LARGEX_W_DATA"][i][idx]
                f.write(f"  {r:23.16e}, {w:23.16e},\n")
            f.write("};\n\n")

        for var in data_name:
            # Skip the individual SMALLX arrays since they're now in ROOT_SMALLX_DATA
            if var in ["ROOT_SMALLX_R0", "ROOT_SMALLX_R1", "ROOT_SMALLX_W0", "ROOT_SMALLX_W1"]:
                continue
            # Skip the individual LARGEX arrays since they're now in ROOT_LARGEX_DATA
            if var in ["ROOT_LARGEX_R_DATA", "ROOT_LARGEX_W_DATA"]:
                continue

            f.write("constexpr __device__ DataType ")
            f.write(f"{var}[] = {{ \n")

            # Special handling for ROOT_RW_DATA - reshape for continuous access and format
            if var == "ROOT_RW_DATA":
                reshaped_data = reshape_root_rw_data(parsed_data[var][i], i)
                # Format with DEGREE1*2 values per row for readability
                # Layout: [nroots, INTERVALS, DEGREE1, 2 (root/weight interleaved)]
                total_size = len(reshaped_data)
                idx = 0
                for root_idx in range(i):  # nroots
                    f.write(f"  // nroot={root_idx}\n")
                    for interval_idx in range(INTERVALS):
                        f.write("  ")
                        # Each row: root_coef0, weight_coef0, root_coef1, weight_coef1, ...
                        for deg_idx in range(DEGREE1):
                            if idx < total_size:
                                f.write(f"{reshaped_data[idx]:23.16e}, ")  # root coefficient
                                idx += 1
                            if idx < total_size:
                                f.write(f"{reshaped_data[idx]:23.16e}, ")  # weight coefficient
                                idx += 1
                        f.write("\n")
            else:
                for val in parsed_data[var][i]:
                    f.write(f"{val},\n")

            f.write("};\n\n")
