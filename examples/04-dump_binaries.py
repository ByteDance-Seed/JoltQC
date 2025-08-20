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

###################################################################
# This example shows how to dump .cubin files for a given molecule
# .cubin files can be reused to avoid recompilation
####################################################################

import time
import os
import pyscf
from gpu4pyscf import scf
from xqc.pyscf import jk

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = scf.RHF(mol)
mf.verbose = 1
mf.conv_tol = 1e-10
mf.max_cycle = 50

# .cubin files will be stored in ./tmp/
# .cubin files can be reused for the same GPU architecture
os.environ["CUPY_CACHE_DIR"] = "./tmp/"

start = time.process_time()

# Overwrite PySCF get_jk function
mf.get_jk = jk.generate_jk_kernel(mol) 
e_tot = mf.kernel()
print('total energy with double precision:', e_tot)

end = time.process_time()
print("CPU time:", end - start, "seconds")

from pathlib import Path
count = sum(1 for f in Path("./tmp").rglob("*") if f.is_file())
size = sum(f.stat().st_size for f in Path("./tmp").rglob("*") if f.is_file())
print(f'{count} binaries, total size: {size/1024/1024} MB')
