import json
import os
import numpy as np
import matplotlib.pyplot as plt
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    pauli_error,
    depolarizing_error,
    phase_damping_error,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# ==========================================================
# S1: Single-qubit state checks (O1 - O4) [L1]
# ==========================================================
print("=== S1: Single-Qubit State Checks ===")
rho_0 = DensityMatrix.from_label("0")
rho_plus = DensityMatrix.from_label("+")
rho_mix_01 = 0.5 * DensityMatrix.from_label("0") + 0.5 * DensityMatrix.from_label("1")
rho_mix_pm = 0.5 * DensityMatrix.from_label("+") + 0.5 * DensityMatrix.from_label("-")
rho_partly = 0.8 * DensityMatrix.from_label("0") + 0.2 * DensityMatrix.from_label("1")

s1_states = [
    ("|0>", rho_0),
    ("|+>", rho_plus),
    ("Equal mix |0>,|1>", rho_mix_01),
    ("Equal mix |+>,|->", rho_mix_pm),
    ("Partly mixed (0.8|0>,0.2|1>)", rho_partly),
]

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_p = np.array([1, 1], dtype=complex) / np.sqrt(2)
ket_m = np.array([1, -1], dtype=complex) / np.sqrt(2)

for name, rho in s1_states:
    M = rho.data

    is_hermitian = np.allclose(M, M.conj().T, atol=1e-10)
    trace_val = np.trace(M).real
    eigvals = np.linalg.eigvalsh(M)
    is_psd = np.all(eigvals >= -1e-12)
    assert is_hermitian and abs(trace_val - 1.0) < 1e-10 and is_psd, f"Invalid state: {name}"

    purity_qiskit = rho.purity().real
    purity_numpy = np.trace(M @ M).real
    assert abs(purity_qiskit - purity_numpy) < 1e-10
    bloch = np.array([np.trace(M @ X).real, np.trace(M @ Y).real, np.trace(M @ Z).real])
    bloch_len = np.linalg.norm(bloch)

    p_z = (np.real(ket_0.conj() @ M @ ket_0), np.real(ket_1.conj() @ M @ ket_1))
    p_x = (np.real(ket_p.conj() @ M @ ket_p), np.real(ket_m.conj() @ M @ ket_m))

    print(
        f"{name:28s} | purity={purity_qiskit:.4f} | r={np.round(bloch, 3)} "
        f"(|r|={bloch_len:.3f}) | Z={np.round(p_z, 3)} | X={np.round(p_x, 3)}"
    )

assert np.allclose(rho_mix_01.data, rho_mix_pm.data, atol=1e-10)
purity_third_state = rho_mix_01.purity().real
print("Q1.3 check passed: Equal mix |0>,|1> and Equal mix |+>,|-> have identical density matrices.\n")

# ==========================================================
# S2: 3-qubit, 10-gate circuit consistency (O5) [L2]
# ==========================================================
print("=== S2: 3-Qubit, 10-Gate Circuit Consistency ===")
qc_s2 = QuantumCircuit(3)
qc_s2.h(0)
qc_s2.cx(0, 1)
qc_s2.ry(0.7, 2)
qc_s2.cx(1, 2)
qc_s2.rz(1.1, 0)
qc_s2.t(1)
qc_s2.cx(2, 0)
qc_s2.s(2)
qc_s2.rx(0.4, 1)
qc_s2.h(2)
assert len(qc_s2.data) == 10

sv_as_dm = DensityMatrix(Statevector(qc_s2))
dm_result = DensityMatrix(qc_s2)
s2_max_diff = float(np.max(np.abs(sv_as_dm.data - dm_result.data)))
assert s2_max_diff < 1e-10
print(f"S2 max difference (Statevector vs DensityMatrix): {s2_max_diff:.3e}\n")

# ==========================================================
# S3: Entanglement & Partial Trace (O6) [L2]
# ==========================================================
print("=== S3: Bell & GHZ Partial Traces ===")
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)
bell_dm = DensityMatrix(qc_bell)
bell_reduced = partial_trace(bell_dm, [1])
bell_reduced_purity = bell_reduced.purity().real
print(f"Bell full purity: {bell_dm.purity().real:.4f} | reduced (1 qubit) purity: {bell_reduced_purity:.4f}")

qc_ghz = QuantumCircuit(3)
qc_ghz.h(0)
qc_ghz.cx(0, 1)
qc_ghz.cx(1, 2)
ghz_dm = DensityMatrix(qc_ghz)
ghz_rem1 = partial_trace(ghz_dm, [2])
ghz_rem2 = partial_trace(ghz_dm, [1, 2])
print(
    f"GHZ full purity:  {ghz_dm.purity().real:.4f} | "
    f"remove 1 qubit purity: {ghz_rem1.purity().real:.4f} | "
    f"remove 2 qubits purity: {ghz_rem2.purity().real:.4f}\n"
)

# ==========================================================
# S4: Noise Sweep — Aer Density Matrix vs Exact Kraus (O7) [L2]
# ==========================================================
print("=== S4: Noise Sweep (21 steps, Aer vs Exact Kraus) ===")
p_vals = np.linspace(0.0, 1.0, 21)
sim_dm = AerSimulator(method="density_matrix")

def apply_kraus_exact(rho_mat, channel, p):
    if channel == "bit_flip":
        return (1 - p) * rho_mat + p * (X @ rho_mat @ X)
    elif channel == "dephasing":
        return (1 - p) * rho_mat + p * (Z @ rho_mat @ Z)
    elif channel == "depolarising":
        return (1 - p) * rho_mat + p * (I2 / 2.0)
    raise ValueError(channel)

def aer_channel_error(channel, p):
    if channel == "bit_flip":
        return pauli_error([("X", p), ("I", 1 - p)])
    elif channel == "dephasing":
        return pauli_error([("Z", p), ("I", 1 - p)])
    elif channel == "depolarising":
        return depolarizing_error(p, 1)
    raise ValueError(channel)

channels = ["bit_flip", "dephasing", "depolarising"]
init_states = {"|0>": rho_0.data, "|+>": rho_plus.data}
s4_aer_purities = {st: {ch: [] for ch in channels} for st in init_states}
s4_kraus_purities = {st: {ch: [] for ch in channels} for st in init_states}
s4_max_diff = 0.0

for st_name, rho_init in init_states.items():
    for ch in channels:
        for p in p_vals:
            rho_k = apply_kraus_exact(rho_init, ch, p)
            s4_kraus_purities[st_name][ch].append(np.trace(rho_k @ rho_k).real)

            qc = QuantumCircuit(1)
            if st_name == "|+>":
                qc.h(0)
            qc.id(0)
            qc.save_density_matrix()

            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(aer_channel_error(ch, p), ["id"])
            res = sim_dm.run(qc, noise_model=nm).result()
            rho_aer = np.asarray(res.data(0)["density_matrix"])
            s4_aer_purities[st_name][ch].append(np.trace(rho_aer @ rho_aer).real)

            s4_max_diff = max(s4_max_diff, float(np.max(np.abs(rho_k - rho_aer))))

assert s4_max_diff < 1e-10, f"Aer and Kraus mismatch: {s4_max_diff}"
print(f"S4 max difference (Aer NoiseModel vs Exact Kraus): {s4_max_diff:.3e}")

p_fine = np.linspace(0.0, 1.0, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
styles = {"bit_flip": ("o", "Bit-flip"), "dephasing": ("s", "Dephasing"), "depolarising": ("^", "Depolarising")}

for ax, st_name in zip(axes, ["|0>", "|+>"]):
    for ch, (marker, label) in styles.items():
        ax.plot(p_vals, s4_aer_purities[st_name][ch], marker, markersize=5, label=f"{label} (Aer)")
        if (st_name == "|0>" and ch == "dephasing") or (st_name == "|+>" and ch == "bit_flip"):
            theory = np.ones_like(p_fine)
        elif ch == "depolarising":
            theory = 0.5 * (1.0 + (1.0 - p_fine) ** 2)
        else:
            theory = 1.0 - 2.0 * p_fine + 2.0 * (p_fine ** 2)
        ax.plot(p_fine, theory, "-", alpha=0.75, label=f"{label} (Theory)")
    ax.set_title(f"Initial State {st_name}")
    ax.set_xlabel("Noise probability p")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Purity Tr(rho^2)")
fig.suptitle("S4: Purity vs Noise Probability (Aer vs Hand Derivation)")
fig.tight_layout()
s4_plot_path = os.path.join(FIG_DIR, "s4_noise_sweep.png")
plt.savefig(s4_plot_path, dpi=200)
plt.close(fig)
print(f"Saved S4 plot to: {s4_plot_path}\n")

# ==========================================================
# S5: Dephasing Trajectories (O8) [L2]
# ==========================================================
print("=== S5: Dephasing Trajectories on |+> (p = 0.3) ===")
p_deph = 0.3
rho_exact_s5 = apply_kraus_exact(rho_plus.data, "dephasing", p_deph)

sim_sv = AerSimulator(method="statevector")
qc_s5 = QuantumCircuit(1)
qc_s5.h(0)
qc_s5.id(0)
qc_s5.save_statevector(pershot=True)

nm_s5 = NoiseModel()
nm_s5.add_all_qubit_quantum_error(pauli_error([("Z", p_deph), ("I", 1 - p_deph)]), ["id"])

N_list = [10, 50, 100, 500, 1000, 5000]
s5_errors = []

for N in N_list:
    res = sim_sv.run(qc_s5, noise_model=nm_s5, shots=N, seed_simulator=17).result()
    sv_list = res.data(0)["statevector"]
    avg_rho = np.zeros((2, 2), dtype=complex)
    for sv in sv_list:
        vec = np.asarray(sv)
        avg_rho += np.outer(vec, vec.conj())
    avg_rho /= N
    err = float(np.linalg.norm(avg_rho - rho_exact_s5, ord="fro"))
    s5_errors.append(err)
    print(f"N = {N:5d} | Frobenius error vs exact DM = {err:.5f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.loglog(N_list, s5_errors, "o-", label="Measured Frobenius error")
ref_curve = s5_errors[-1] * np.sqrt(N_list[-1] / np.array(N_list, dtype=float))
ax.loglog(N_list, ref_curve, "--", label="Reference 1/sqrt(N) scaling")
ax.set_xlabel("Number of trajectories N")
ax.set_ylabel("Frobenius Error ||rho_avg - rho_exact||_F")
ax.set_title("S5: Dephasing Trajectory Convergence (|+>, p=0.3)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
s5_plot_path = os.path.join(FIG_DIR, "s5_trajectories.png")
plt.savefig(s5_plot_path, dpi=200)
plt.close(fig)
print(f"Saved S5 plot to: {s5_plot_path}\n")

# ==========================================================
# Q2.4 Numerical Check: Qiskit Noise Parameters & Qubit Ordering
# ==========================================================
print("=== Q2.4: Parameter & Qubit-Ordering Checks ===")
p_test = 0.2

rho_dep_q = rho_plus.evolve(depolarizing_error(p_test, 1)).data
rho_dep_o7 = apply_kraus_exact(rho_plus.data, "depolarising", p_test)
print(f"depolarizing_error(0.2, 1) vs O7 max diff: {np.max(np.abs(rho_dep_q - rho_dep_o7)):.3e}")

gamma_equiv = 4 * p_test * (1 - p_test)
rho_pd_raw = rho_plus.evolve(phase_damping_error(p_test)).data
rho_pd_conv = rho_plus.evolve(phase_damping_error(gamma_equiv)).data
rho_z_o7 = apply_kraus_exact(rho_plus.data, "dephasing", p_test)
print(f"phase_damping_error(p=0.2) vs O7 max diff:       {np.max(np.abs(rho_pd_raw - rho_z_o7)):.3e} (DOES NOT MATCH)")
print(f"phase_damping_error(4p(1-p)=0.64) vs O7 diff:    {np.max(np.abs(rho_pd_conv - rho_z_o7)):.3e} (MATCHES)")

qc_order = QuantumCircuit(2)
qc_order.x(0)
sv_order = Statevector(qc_order)
print(f"X on qubit 0 -> probabilities dict: {sv_order.probabilities_dict()} (qubit 0 is rightmost bit)\n")

# ==========================================================
# Generate results.json in Repo Root
# ==========================================================
results = {
    "team": ["2023A7PS0381G", "2023A7PS0478G"],
    "level_attempted": 2,
    "versions": {
        "qiskit": qiskit.__version__,
        "qiskit-aer": qiskit_aer.__version__,
    },
    "L1": {"purity_third_state": float(purity_third_state)},
    "L2": {"bell_reduced_purity": float(bell_reduced_purity)},
    "L3": {},
}

output_path = os.path.join(REPO_ROOT, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"All L1 & L2 simulations passed! Wrote {output_path}")