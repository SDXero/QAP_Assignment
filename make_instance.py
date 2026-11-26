#!/usr/bin/env python3
"""
make_instance.py -- generates your team's Level 3 instance for the
"Pure vs Mixed States" assignment.

Usage
-----
    python make_instance.py 4821

Give it your team's 4-digit number (digits only, leading zeros allowed).
The same number always gives the same instance, on any machine and any
Python 3 version, so you can regenerate your instance whenever you need it.

The script writes two files:

  instance.json        the full instance. This is the authoritative file;
                       your results.json must be about this instance.
  instance_circuit.py  ready-made Qiskit code that builds your circuit:
                           from instance_circuit import build_circuit
                           qc = build_circuit()
                       It contains the gates only, without noise.

It also prints a diagram of the circuit (qubit 0 is the top wire).

Use -o to change the name of the JSON file and -c to change the name of the
circuit file, e.g.  python make_instance.py 4821 -o my.json -c my_circuit.py

The instance contains:
  code          your team's 4-digit number
  n             number of qubits (4 to 7)
  gates         the circuit, as an ordered list of gates
  noise         the noise channel and its parameter p (see "Noise" below)
  subsystem_A   the qubits in subsystem A (the rest form subsystem B)
  traj_seed     the random seed to use for your trajectory simulations
  fingerprint   a short checksum of all of the above

Noise
-----
After every gate, apply the channel once to every qubit that the gate acted
on, as an independent single-qubit channel. A CNOT therefore gets two
channels, one on its control qubit and one on its target qubit. The channel
and p are as defined in O7 of the assignment.

Conventions
-----------
Qubit labels are circuit wires: qubit 0 is the first (top) wire of the circuit.
Qiskit orders basis states little-endian (qubit 0 is the rightmost bit of a
bitstring), so take care when reading state vectors and tracing out qubits.
A CNOT is written {"g": "CNOT", "q": [control, target]}.

Only the Python standard library is needed to generate the instance.
Qiskit is needed only to run instance_circuit.py or to call to_qiskit().
"""

import argparse
import hashlib
import json
import math
import re
import sys

SINGLE_GATES = ["H", "S", "T", "X"]
CHANNELS = ["bitflip", "dephasing", "depolarising"]

# Without noise, subsystem A must be clearly entangled with B: the reduced
# state rho_A of the noiseless circuit must have purity at most this value.
MAX_NOISELESS_PURITY_A = 0.9
MAX_ATTEMPTS = 10000


# ---------------------------------------------------------------------------
# Random numbers
#
# Python's random module only guarantees the same sequence across versions
# for random() itself; randint, choice, sample and uniform may change. To make
# instances identical on every machine and Python version, all random draws
# here are integers derived directly from SHA-256.
# ---------------------------------------------------------------------------

class _HashStream:
    """Deterministic stream of random integers from SHA-256 in counter mode."""

    def __init__(self, key):
        self._key = key
        self._counter = 0
        self._pool = 0      # unused random bits, as an integer
        self._pool_bits = 0

    def _bits(self, k):
        while self._pool_bits < k:
            block = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
            self._pool = (self._pool << 256) | int.from_bytes(block, "big")
            self._pool_bits += 256
        self._pool_bits -= k
        value = self._pool >> self._pool_bits
        self._pool &= (1 << self._pool_bits) - 1
        return value

    def below(self, n):
        """Uniform integer in [0, n), by rejection sampling (no bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        k = max(1, (n - 1).bit_length())
        while True:
            r = self._bits(k)
            if r < n:
                return r

    def randint(self, a, b):
        """Uniform integer in [a, b], both ends included."""
        return a + self.below(b - a + 1)

    def choice(self, items):
        return items[self.below(len(items))]

    def chance(self, num, den):
        """True with probability num/den, exactly."""
        return self.below(den) < num

    def two_distinct(self, n):
        """Two different integers in [0, n), in random order."""
        a = self.below(n)
        b = self.below(n - 1)
        if b >= a:
            b += 1
        return a, b


# ---------------------------------------------------------------------------
# A tiny noiseless statevector simulator (standard library only), used to
# check that subsystem A is entangled with B before noise is added.
# Basis index bit q is qubit q (little-endian, as in Qiskit).
# ---------------------------------------------------------------------------

_R = math.sqrt(0.5)
_MATRICES = {
    "H": ((_R, _R), (_R, -_R)),
    "S": ((1, 0), (0, 1j)),
    "T": ((1, 0), (0, complex(_R, _R))),  # e^{i pi/4}, exact on every platform
    "X": ((0, 1), (1, 0)),
}


def _final_state(n, gates):
    psi = [0j] * (2 ** n)
    psi[0] = 1 + 0j
    for gate in gates:
        if gate["g"] == "CNOT":
            c, t = gate["q"]
            for i in range(2 ** n):
                if (i >> c) & 1 and not (i >> t) & 1:
                    j = i | (1 << t)
                    psi[i], psi[j] = psi[j], psi[i]
        else:
            (q,) = gate["q"]
            (a, b), (c, d) = _MATRICES[gate["g"]]
            for i in range(2 ** n):
                if not (i >> q) & 1:
                    j = i | (1 << q)
                    x, y = psi[i], psi[j]
                    psi[i], psi[j] = a * x + b * y, c * x + d * y
    return psi


def _noiseless_purity_a(n, gates, cut):
    """Tr(rho_A^2) for the noiseless final state, A = qubits 0..cut-1."""
    psi = _final_state(n, gates)
    dim_a, dim_b = 2 ** cut, 2 ** (n - cut)
    # amplitude index = a + dim_a * b, since A holds the low (rightmost) bits
    m = [[psi[a + dim_a * b] for b in range(dim_b)] for a in range(dim_a)]
    purity = 0.0
    for a1 in range(dim_a):
        for a2 in range(dim_a):
            rho = sum(m[a1][b] * m[a2][b].conjugate() for b in range(dim_b))
            purity += abs(rho) ** 2
    return purity


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def _check_code(code):
    code = str(code).strip()
    if not re.fullmatch(r"\d{4}", code):
        raise ValueError("The team number must be exactly 4 digits, e.g. 4821.")
    return code


def _fingerprint(inst):
    body = {k: inst[k] for k in ("code", "n", "gates", "noise", "subsystem_A", "traj_seed")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def make_instance(code):
    """Return the instance (a dict) for a 4-digit team number."""
    code = _check_code(code)
    rng = _HashStream(f"pure-vs-mixed|{code}".encode())

    n = rng.randint(4, 7)
    cut = rng.randint(1, n - 1)  # subsystem A = qubits 0..cut-1
    channel = rng.choice(CHANNELS)
    p = rng.randint(20, 150) / 1000  # 0.020 to 0.150
    traj_seed = rng.below(2 ** 31)

    # Draw circuits until A is clearly entangled with B without noise, so
    # that S(rho_A) is not trivially zero.
    for _ in range(MAX_ATTEMPTS):
        gates = []
        for _ in range(rng.randint(12, 19)):
            if rng.chance(2, 5):
                gates.append({"g": "CNOT", "q": list(rng.two_distinct(n))})
            else:
                gates.append({"g": rng.choice(SINGLE_GATES), "q": [rng.below(n)]})
        if _noiseless_purity_a(n, gates, cut) <= MAX_NOISELESS_PURITY_A:
            break
    else:
        raise RuntimeError("Could not generate a suitable circuit. Please contact the instructor.")

    inst = {
        "code": code,
        "n": n,
        "gates": gates,
        "noise": {
            "channel": channel,
            "p": p,
            "applied": "after every gate, an independent single-qubit channel on "
                       "each qubit the gate acted on (both qubits for a CNOT)",
        },
        "subsystem_A": list(range(cut)),
        "traj_seed": traj_seed,
    }
    inst["fingerprint"] = _fingerprint(inst)
    return inst


# ---------------------------------------------------------------------------
# Output: diagram and Qiskit code
# ---------------------------------------------------------------------------

def draw(inst):
    """Plain-text diagram, one row per wire, qubit 0 on top.

    Single-qubit gates show their letter, a CNOT shows * on the control and
    + on the target, joined by |.
    """
    n, cut = inst["n"], len(inst["subsystem_A"])
    rows = [[] for _ in range(n)]
    for gate in inst["gates"]:
        if gate["g"] == "CNOT":
            c, t = gate["q"]
            lo, hi = min(c, t), max(c, t)
            for q in range(n):
                if q == c:
                    cell = "-*-"
                elif q == t:
                    cell = "-+-"
                elif lo < q < hi:
                    cell = "-|-"
                else:
                    cell = "---"
                rows[q].append(cell)
        else:
            (q0,) = gate["q"]
            for q in range(n):
                rows[q].append(f"-{gate['g']}-" if q == q0 else "---")
    lines = []
    for q in range(n):
        label = f"q{q} ({'A' if q < cut else 'B'}): "
        lines.append(label + "".join(rows[q]) + "-")
    lines.append("")
    lines.append("*  = CNOT control,  + = CNOT target,  A/B = subsystem")
    return "\n".join(lines)


def circuit_code(inst):
    """Python source for a module with build_circuit() for this instance."""
    noise = inst["noise"]
    diagram = "\n".join("    " + line for line in draw(inst).splitlines())
    lines = [
        '"""',
        f"Circuit for team {inst['code']} (instance fingerprint {inst['fingerprint']}).",
        "Generated by make_instance.py. Do not edit: regenerate it instead.",
        "",
        "This circuit contains the GATES ONLY. Your instance also has noise:",
        f"    channel = {noise['channel']}, p = {noise['p']}",
        f"    applied {noise['applied']}.",
        f"Subsystem A = qubits {inst['subsystem_A']}; subsystem B = the rest.",
        f"Trajectory seed = {inst['traj_seed']}.",
        "",
        "Diagram (qubit 0 = top wire):",
        "",
        diagram,
        "",
        "Usage:",
        "    from instance_circuit import build_circuit",
        "    qc = build_circuit()",
        '"""',
        "",
        "from qiskit import QuantumCircuit",
        "",
        "",
        "def build_circuit():",
        f"    qc = QuantumCircuit({inst['n']})",
    ]
    for k, gate in enumerate(inst["gates"]):
        if gate["g"] == "CNOT":
            c, t = gate["q"]
            lines.append(f"    qc.cx({c}, {t})".ljust(20) + f"# gate {k}: CNOT, control {c}, target {t}")
        else:
            (q,) = gate["q"]
            lines.append(f"    qc.{gate['g'].lower()}({q})".ljust(20) + f"# gate {k}: {gate['g']} on qubit {q}")
    lines += [
        "    return qc",
        "",
        "",
        'if __name__ == "__main__":',
        "    print(build_circuit().draw())",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional helpers
# ---------------------------------------------------------------------------

def load_instance(path="instance.json"):
    with open(path) as f:
        return json.load(f)


def to_qiskit(inst):
    """Build the (noise-free) Qiskit circuit for an instance."""
    from qiskit import QuantumCircuit  # imported here so generating needs no Qiskit

    qc = QuantumCircuit(inst["n"])
    for gate in inst["gates"]:
        name, qubits = gate["g"], gate["q"]
        if name == "CNOT":
            qc.cx(qubits[0], qubits[1])
        else:
            getattr(qc, name.lower())(qubits[0])
    return qc


def main():
    parser = argparse.ArgumentParser(
        description="Generate your team's instance.json and instance_circuit.py")
    parser.add_argument("code", help="your team's 4-digit number")
    parser.add_argument("-o", "--out", default="instance.json", help="JSON output file")
    parser.add_argument("-c", "--circuit", default="instance_circuit.py",
                        help="Qiskit circuit output file")
    args = parser.parse_args()
    try:
        inst = make_instance(args.code)
    except ValueError as err:
        sys.exit(f"Error: {err}")
    with open(args.out, "w") as f:
        json.dump(inst, f, indent=2)
    with open(args.circuit, "w") as f:
        f.write(circuit_code(inst))
    noise = inst["noise"]
    print(f"Team {inst['code']}: wrote {args.out} and {args.circuit}")
    print(f"  fingerprint = {inst['fingerprint']}")
    print(f"  qubits n = {inst['n']}, gates = {len(inst['gates'])}")
    print(f"  noise = {noise['channel']} with p = {noise['p']}, {noise['applied']}")
    print(f"  subsystem A = qubits {inst['subsystem_A']}, trajectory seed = {inst['traj_seed']}")
    print()
    print(draw(inst))


if __name__ == "__main__":
    main()
