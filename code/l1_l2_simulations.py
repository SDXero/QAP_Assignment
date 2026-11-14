import json
import numpy as np
import qiskit
import qiskit_aer
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

purity_third_state = np.trace(rho_mix_01.data @ rho_mix_01.data).real
print(f"Purity of third state: {purity_third_state}")