import os
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

for script in ["l1_l2_simulations.py", "l3_simulations.py"]:
    script_path = os.path.join(CODE_DIR, script)
    print(f"\n>>> Running {script} ...")
    subprocess.run([sys.executable, script_path], check=True)