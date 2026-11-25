import json
import numpy as np
import qiskit
import qiskit_aer
import os
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

# ==========================================
# L1: States (S1)
# ==========================================
rho_0 = DensityMatrix.from_label('0')
rho_plus = DensityMatrix.from_label('+')
rho_mix_01 = 0.5 * DensityMatrix.from_label('0') + 0.5 * DensityMatrix.from_label('1')
rho_mix_plusminus = 0.5 * DensityMatrix.from_label('+') + 0.5 * DensityMatrix.from_label('-')
rho_partly_mixed = 0.8 * DensityMatrix.from_label('0') + 0.2 * DensityMatrix.from_label('1')

purity_third_state = rho_mix_01.purity().real
print(f"Purity of third state: {purity_third_state}")

qc_s2 = random_circuit(num_qubits=3, depth=10, seed=42)
sv_as_dm = DensityMatrix(Statevector(qc_s2))
dm_result = DensityMatrix(qc_s2)

max_diff = np.max(np.abs(sv_as_dm.data - dm_result.data))
assert max_diff < 1e-10, "Statevector and DensityMatrix do not agree!"
print(f"Circuit consistency check passed with max difference: {max_diff}")

# ==========================================
# L2: Entanglement & Partial Trace (S3)
# ==========================================
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)
bell_dm = DensityMatrix(qc_bell)
bell_reduced = partial_trace(bell_dm, [1])
bell_reduced_purity = bell_reduced.purity().real
print(f"Bell reduced purity: {bell_reduced_purity}")

# ==========================================
# Generate results.json automatically
# ==========================================
results = {
    "team": ["2023A7PS0381G", "2023A7PS0478G"],
    "level_attempted": 2,
    "versions": {
        "qiskit": qiskit.__version__,
        "qiskit-aer": qiskit_aer.__version__
    },
    "L1": {"purity_third_state": float(purity_third_state)},
    "L2": {"bell_reduced_purity": float(bell_reduced_purity)},
    "L3": {}
}

# Always saves results.json in the repo root regardless of where the script is run from
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("Code executed successfully. results.json generated.")