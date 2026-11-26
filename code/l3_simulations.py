import json
import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

INSTANCE_PATH = os.path.join(REPO_ROOT, "instance.json")
RESULTS_PATH = os.path.join(REPO_ROOT, "results.json")

with open(INSTANCE_PATH, "r") as f:
    inst = json.load(f)

n = inst["n"]
gates = inst["gates"]
channel = inst["noise"]["channel"]
p = inst["noise"]["p"]
sub_A = inst["subsystem_A"]
sub_B = [q for q in range(n) if q not in sub_A]
traj_seed = inst["traj_seed"]

print(f"=== Loaded Instance (code={inst['code']}, fingerprint={inst['fingerprint']}) ===")
print(f"n={n}, gates={len(gates)}, channel={channel}, p={p}, A={sub_A}, B={sub_B}, seed={traj_seed}\n")

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def vn_entropy(rho_mat):
    """Von Neumann entropy S(rho) = -Tr(rho log2 rho)."""
    evals = np.linalg.eigvalsh(rho_mat)
    evals = evals[evals > 1e-15]
    return float(-np.sum(evals * np.log2(evals)))


def build_instance_qc():
    qc = QuantumCircuit(n)
    for gate in gates:
        name, qs = gate["g"], gate["q"]
        if name == "CNOT":
            qc.cx(qs[0], qs[1])
        else:
            getattr(qc, name.lower())(qs[0])
    return qc


def make_1q_aer_error(ch, p_val):
    if ch == "bitflip":
        return pauli_error([("X", p_val), ("I", 1.0 - p_val)])
    elif ch == "dephasing":
        return pauli_error([("Z", p_val), ("I", 1.0 - p_val)])
    elif ch == "depolarising":
        return depolarizing_error(p_val, 1)
    raise ValueError(f"Unknown channel: {ch}")


def make_1q_kraus(ch, p_val):
    if ch == "bitflip":
        ops = [np.sqrt(1.0 - p_val) * I2, np.sqrt(p_val) * X]
    elif ch == "dephasing":
        ops = [np.sqrt(1.0 - p_val) * I2, np.sqrt(p_val) * Z]
    elif ch == "depolarising":
        ops = [
            np.sqrt(1.0 - 0.75 * p_val) * I2,
            np.sqrt(0.25 * p_val) * X,
            np.sqrt(0.25 * p_val) * Y,
            np.sqrt(0.25 * p_val) * Z,
        ]
    else:
        raise ValueError(f"Unknown channel: {ch}")
    return Kraus(ops)


def build_instance_noise_model(ch, p_val):
    err_1q = make_1q_aer_error(ch, p_val)
    err_2q = err_1q.tensor(err_1q)  # independent single-qubit channel on both CNOT qubits
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(err_1q, ["h", "s", "t", "x"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx"])
    return nm


# ==========================================================
# S6: Instance Simulation — Aer Density Matrix vs Exact Kraus
# ==========================================================
print("=== S6: Instance Simulation (Aer DM vs Exact Kraus) ===")
qc_clean = build_instance_qc()

rho_noiseless = DensityMatrix(qc_clean)
rho_A_noiseless = partial_trace(rho_noiseless, sub_B)
print(
    f"Noiseless | purity={rho_noiseless.purity().real:.6f}, "
    f"S(rho)={vn_entropy(rho_noiseless.data):.6f}, "
    f"S(rho_A)={vn_entropy(rho_A_noiseless.data):.6f}"
)

nm_inst = build_instance_noise_model(channel, p)
qc_dm = qc_clean.copy()
qc_dm.save_density_matrix()
sim_dm = AerSimulator(method="density_matrix")
res_dm = sim_dm.run(qc_dm, noise_model=nm_inst).result()
rho_aer = DensityMatrix(np.asarray(res_dm.data(0)["density_matrix"]))

kraus_1q = make_1q_kraus(channel, p)
rho_kraus = DensityMatrix.from_int(0, 2 ** n)
for gate in gates:
    step_qc = QuantumCircuit(n)
    name, qs = gate["g"], gate["q"]
    if name == "CNOT":
        step_qc.cx(qs[0], qs[1])
    else:
        getattr(step_qc, name.lower())(qs[0])
    rho_kraus = rho_kraus.evolve(step_qc)
    for q in qs:
        rho_kraus = rho_kraus.evolve(kraus_1q, qargs=[q])

s6_max_diff = float(np.max(np.abs(rho_aer.data - rho_kraus.data)))
assert s6_max_diff < 1e-10, f"S6 mismatch between Aer and Kraus: {s6_max_diff}"

rho_A = partial_trace(rho_kraus, sub_B)
l3_purity = float(rho_kraus.purity().real)
l3_entropy = vn_entropy(rho_kraus.data)
l3_entropy_A = vn_entropy(rho_A.data)

print(f"Noisy     | purity={l3_purity:.6f}, S(rho)={l3_entropy:.6f}, S(rho_A)={l3_entropy_A:.6f}")
print(f"S6 max difference (Aer DM vs Exact Kraus): {s6_max_diff:.3e}")
print(f"Comparison: S(rho_A) {'<' if l3_entropy_A < l3_entropy else '>'} S(rho)\n")

# ==========================================================
# S7 & A3.1: Instance Trajectories (Aer Statevector + Seed)
# ==========================================================
print("=== S7: Instance Trajectories ===")
sim_sv = AerSimulator(method="statevector")
qc_sv = qc_clean.copy()
qc_sv.save_statevector(pershot=True)

N_list = [10, 30, 100, 300, 1000, 3000]
s7_errors = []
s7_purities = []

for N in N_list:
    res = sim_sv.run(qc_sv, noise_model=nm_inst, shots=N, seed_simulator=traj_seed).result()
    sv_list = res.data(0)["statevector"]
    V = np.array([np.asarray(sv) for sv in sv_list], dtype=complex)  # shape (N, 2^n)
    avg_rho = (V.T @ V.conj()) / N
    err = float(np.linalg.norm(avg_rho - rho_kraus.data, ord="fro"))
    est_purity = float(np.trace(avg_rho @ avg_rho).real)
    s7_errors.append(err)
    s7_purities.append(est_purity)
    print(f"N = {N:5d} | Frobenius error = {err:.6f} | est purity = {est_purity:.6f}")

log_N = np.log10(N_list)
log_err = np.log10(s7_errors)
slope, intercept = np.polyfit(log_N, log_err, 1)
C_prefactor = 10 ** intercept
print(f"A3.1 Log-log fit: slope = {slope:.4f} (expected -0.5000), prefactor C = {C_prefactor:.4f}\n")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.loglog(N_list, s7_errors, "o-", label=f"Measured error (slope={slope:.3f})")
fit_line = C_prefactor * (np.array(N_list, dtype=float) ** slope)
ref_line = s7_errors[-1] * np.sqrt(N_list[-1] / np.array(N_list, dtype=float))
ax.loglog(N_list, fit_line, ":", label="Linear log-log fit")
ax.loglog(N_list, ref_line, "--", label="Theoretical 1/sqrt(N) slope (-0.5)")
ax.set_xlabel("Number of trajectories N")
ax.set_ylabel("Frobenius Error ||rho_avg - rho_exact||_F")
ax.set_title(f"S7 / A3.1: Instance Trajectory Convergence (n={n}, {channel}, p={p})")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
s7_plot_path = os.path.join(FIG_DIR, "s7_instance_trajectories.png")
plt.savefig(s7_plot_path, dpi=200)
plt.close(fig)
print(f"Saved S7 plot to: {s7_plot_path}\n")

# ==========================================================
# S8, A3.2 & A3.3: Scaling Benchmark (Statevector vs Density Matrix)
# ==========================================================
print("=== S8: Scaling Benchmark (Fixed-Depth Circuit Family) ===")


def build_benchmark_circuit(num_qubits, layers=3):
    qc = QuantumCircuit(num_qubits)
    for _ in range(layers):
        for q in range(num_qubits):
            qc.h(q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def benchmark_method(method, n_values, repeats=3):
    sim = AerSimulator(method=method)
    times = []
    state_mems_mb = []
    py_peak_mems_mb = []
    for nq in n_values:
        qc = build_benchmark_circuit(nq, layers=3)
        if method == "statevector":
            qc.save_statevector()
            key = "statevector"
        else:
            qc.save_density_matrix()
            key = "density_matrix"

        _ = sim.run(qc, shots=1).result()

        run_times = []
        state_bytes = 0
        tracemalloc.start()
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = sim.run(qc, shots=1).result()
            arr = np.asarray(r.data(0)[key])
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
            state_bytes = arr.nbytes
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        med_time = float(np.median(run_times))
        times.append(med_time)
        state_mems_mb.append(state_bytes / (1024 ** 2))
        py_peak_mems_mb.append(peak_bytes / (1024 ** 2))
        print(
            f"{method:14s} | n={nq:2d} | median time={med_time*1000:8.3f} ms | "
            f"state mem={state_bytes/(1024**2):8.4f} MB"
        )
    return np.array(times), np.array(state_mems_mb), np.array(py_peak_mems_mb)


sv_ns = list(range(2, 19))
dm_ns = list(range(2, 11))  

sv_times, sv_mem, _ = benchmark_method("statevector", sv_ns)
dm_times, dm_mem, _ = benchmark_method("density_matrix", dm_ns)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

ax1.semilogy(sv_ns, sv_times * 1000, "o-", label="Statevector (measured)")
ax1.semilogy(dm_ns, dm_times * 1000, "s-", label="Density Matrix (measured)")
ax1.semilogy(sv_ns, sv_times[-1] * 1000 * (2.0 ** (np.array(sv_ns) - sv_ns[-1])), "--", alpha=0.6, label="~ 2^n slope")
ax1.semilogy(dm_ns, dm_times[-1] * 1000 * (4.0 ** (np.array(dm_ns) - dm_ns[-1])), ":", alpha=0.7, label="~ 4^n slope")
ax1.set_xlabel("Number of qubits n")
ax1.set_ylabel("Median runtime (ms)")
ax1.set_title("Runtime Scaling (S8)")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=8)

ax2.semilogy(sv_ns, sv_mem, "o-", label="Statevector state size (MB)")
ax2.semilogy(dm_ns, dm_mem, "s-", label="Density Matrix state size (MB)")
ax2.set_xlabel("Number of qubits n")
ax2.set_ylabel("State memory (MB)")
ax2.set_title("Memory Scaling (S8: 16*2^n vs 16*4^n bytes)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=8)

fig.suptitle("S8 / A3.2: Aer Statevector vs Density-Matrix Scaling Benchmark")
fig.tight_layout()
s8_plot_path = os.path.join(FIG_DIR, "s8_scaling_benchmark.png")
plt.savefig(s8_plot_path, dpi=200)
plt.close(fig)
print(f"\nSaved S8 plot to: {s8_plot_path}")

a_sv = float(np.median(sv_times[-4:] / (2.0 ** np.array(sv_ns[-4:]))))
a_dm = float(np.median(dm_times[-3:] / (4.0 ** np.array(dm_ns[-3:]))))
print("\n=== A3.3: Estimated Crossover n (Trajectories vs Density Matrix) ===")
print(f"Asymptotic time constants: a_sv={a_sv:.3e} s/2^n, a_dm={a_dm:.3e} s/4^n")
for eps in [0.05, 0.01, 0.005]:
    N_req = (C_prefactor / eps) ** 2
    n_cross = np.log2(N_req * (a_sv / a_dm))
    print(f"Target error eps={eps:.3f} -> N_req~{N_req:10.0f} shots -> crossover n ~ {n_cross:.1f} qubits")

# ==========================================================
# Update results.json with Level 3 outputs
# ==========================================================
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
else:
    results = {
        "team": ["2023A7PS0381G", "2023A7PS0478G"],
        "versions": {"qiskit": qiskit.__version__, "qiskit-aer": qiskit_aer.__version__},
        "L1": {},
        "L2": {},
    }

results["level_attempted"] = 3
results["L3"] = {
    "purity": l3_purity,
    "entropy": l3_entropy,
    "entropy_A": l3_entropy_A,
    "traj_estimate_purity": s7_purities[-1],
    "n_traj": N_list[-1],
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nAll Level 3 simulations passed! Updated {RESULTS_PATH}")