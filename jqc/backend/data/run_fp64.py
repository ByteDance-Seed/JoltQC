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

#######################################################################
#    This script generates fragment strategy for fp64, up to g orbitals
#######################################################################

import os
import subprocess
import threading
import queue

script_dir = os.path.dirname(__file__)
env_cmd = f'export PYTHONPATH="${{PYTHONPATH}}:{script_dir}/../../../"'

os.makedirs("logs", exist_ok=True)

n_groups = 5
commands = []
# 8-fold symmetry is applied to the fragment strategy
for i in range(n_groups):
    for j in range(i + 1):
        for k in range(i + 1):
            for l in range(k + 1):
                cmd = f" python3 generate_fragment.py {i} {j} {k} {l} fp64 > logs/{i}{j}{k}{l}_fp64.log"
                commands.append(env_cmd + " && " + cmd)

max_concurrent_jobs = 1
q = queue.Queue()
lock = threading.Lock()


# Worker function
def worker():
    while True:
        try:
            cmd = q.get_nowait()
        except queue.Empty:
            return
        with lock:
            print(f"Starting: {cmd}")
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
        with lock:
            print(f"Finished: {cmd}")
        q.task_done()


# Fill the queue
for cmd in commands:
    q.put(cmd)

# Create and start threads
threads = []
for _ in range(min(max_concurrent_jobs, len(commands))):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Wait for all jobs to finish
q.join()

print("All commands completed.")
