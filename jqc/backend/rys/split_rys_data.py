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
with open("gpu4pyscf/gpu4pyscf/lib/gvhf-rys/rys_roots_dat.cu") as f:
    rys_roots_data = f.read()

import re

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

# nroots = 10 is missing
for i in range(1, 10):
    with open(f"rys_root{i}.cu", "w+") as f:
        for var in data_name:
            f.write("__device__\n")
            f.write(f"double {var}[] = {{ \n")
            for val in parsed_data[var][i]:
                f.write(f"{val},\n")
            f.write("};\n\n")
