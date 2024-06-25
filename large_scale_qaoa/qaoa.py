"""QAOA class for ML error mitigation and circuit construction."""

from typing import Dict, List, Optional, Tuple

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit_aer import Aer
from qiskit import quantum_info as qi
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)

from large_scale_qaoa.swap_mapping import find_path
from large_scale_qaoa.graph_utils import build_graph


class ErrorMitigationQAOA:
    """A class to run error mitigated QAOA and efficiently construct its circuits.

    Conventions: the QAOA circuits in this class are built with the following
    definitions. The aim is to minimize the energy of the cost operator
    :math:`H_c` to maximize the value of the classical cost function :math:`f(x)`.
    The variational Ansatz is made from the blocks

    ..math::

        \exp(-i\beta_k H_m)\exp(-i\gamma_k H_c)

    where the mixer operator is thus defined by

    ..math::

        H_m = -\sum_i X_i

    so that the |+> state is the ground state of :math:`H_m`. The cost operator is defined by

        H_c = \sum_{i,j} w_{i,j}Z_iZ_j

    This means that for the mixer we apply rotations of the form :math:`R_{x}(-2\beta)` and
    for each term in the cost operator we apply the gates :math:`Rzz(2\gamma w_{ij})`.
    """

    def __init__(
            self,
            shots: int,
            local_correlators: List[Tuple[str, float]],
            backend,
            path: Optional[List[int]] = None,
    ):
        """Initialize the QAOA class with the paulis that describe the graph.

        Args:
            shots: The number of shots with which to run.
            local_correlators: The Paulis for the QAOA that have been permuted
                with the SAT mapping.
            backend: The backend on which to run.
            path: The line of qubits to run. If None the code finds its own path.
                Note that this might take some time to run.
        """
        self.G = build_graph(local_correlators)
        self.N = len(self.G)
        self.shots = shots
        self.backend = backend

        # used to build the QAOA circuit in the transpiler step that goes
        # from PauliEvolution to a standard basis gate set.
        self.basis_gates = ["rz", "sx", "x", "cx"]

        # Find the qubit path on which to run.
        # This can take some time when running on large graphs.
        if path is None:
            path, _ = find_path(self.N, True, backend)

        self.path = path

        # The regressor that will be used to error mitigate the correlators.
        self.regr = None

        self.local_correlators = local_correlators
        self.initial_layout = None

        self._sampler = None

        # Variables to save data during an optimization.
        self.job_ids = []
        self.counts = []

    @property
    def sampler(self):
        """Return the primitive with which to run."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        """Set the primitive for the QAOA. To be used with sessions only."""
        self._sampler = sampler

    @staticmethod
    def replace_rz_with_barrier(qc: QuantumCircuit):
        """A transpiler pass to replace rz with a barrier."""
        dag = circuit_to_dag(qc)
        replacement = QuantumCircuit(1)
        replacement.barrier()

        for node in dag.op_nodes():
            if node.op.name == "rz" and len(node.op.params) == 1:
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))

        return dag_to_circuit(dag)

    def create_qaoa_circ_pauli_evolution(
            self,
            theta,
            superposition=True,
            random_cut=None,
            transpile_circ: bool = False,
            remove_rz: bool = False,
            apply_swaps: bool = True,
    ):
        """Main circuit construction method.

        This function uses the line swap strategy to route the QAOA circuit.
        Here, we assume that the Paulis have already been permuted with the
        SAT mapping. If this is not the case, then the circuit will simply
        be deeper. This function works by routing a single layer of QAOA. When
        building `p>1` QAOA it alternates between applying the depth-one
        routed layers in order and in reversed order. In both cases it ensures
        that decision variable `i` is measured and saved in classical bit `i`.

        Args:
            theta: The QAOA angles.
            superposition: If True we initialized the qubits in the `+` state.
            random_cut: A random cut, i.e., a series of 1 and 0 with the same length
                as the number of qubits. If qubit `i` has a `1` then we flip its
                initial state from `+` to `-`.
            transpile_circ: If True, we transpile the circuit to the backend.
            remove_rz: If True then the rz gates in the cost Hamiltonian part
                of the circuit will be replaced with barriers. This makes it
                possible to efficiently simulate the circuit.
            apply_swaps: If True, the default, then we apply the swap pass manager.
                This can be set to false for noiseless simulators only.
        """
        gamma = theta[: len(theta) // 2]
        beta = theta[len(theta) // 2:]
        p = len(theta) // 2

        # First, create the Hamiltonian of 1 layer of QAOA
        hc_evo = QuantumCircuit(self.N)
        op = qi.SparsePauliOp.from_list(self.local_correlators)
        gamma_param = Parameter("g")
        hc_evo.append(PauliEvolutionGate(op, gamma_param), range(self.N))

        # This will allow us to recover the permutation of the measurements that the swap introduce.
        hc_evo.measure_all()

        edge_coloring = {(idx, idx + 1): idx % 2 for idx in range(self.N)}

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(
                    SwapStrategy.from_line([i for i in range(self.N)]),
                    edge_coloring,
                ),
            ]
        )

        if apply_swaps:
            hc_evo = pm_pre.run(hc_evo)

        # Now transpile to sx, rz, x, cx basis
        hc_evo = transpile(hc_evo, basis_gates=self.basis_gates)

        # Replace Rz with zero rotations in cost Hamiltonian if desired
        if remove_rz:
            hc_evo = self.replace_rz_with_barrier(hc_evo)

        # Compute the measurement map (qubit to classical bit). we will apply this for p % 2 == 1.
        if p % 2 == 1:
            meas_map = self.make_meas_map(hc_evo)
        else:
            meas_map = {idx: idx for idx in range(self.N)}

        hc_evo.remove_final_measurements()

        qc = QuantumCircuit(self.N)

        if superposition:
            # Initial superpositions of all solutions, i.e. ground state of H_B (mixing Hamiltonian)
            qc.h(range(self.N))

        if random_cut is not None:
            if superposition:
                raise ValueError(
                    "Applying random cuts with superposition True has no effect."
                )

            for idx, coin_flip in enumerate(random_cut):
                if coin_flip == 1:
                    qc.x(idx)

        for i in range(p):
            bind_dict = {} if remove_rz else {gamma_param: gamma[i]}
            bound_hc = hc_evo.assign_parameters(bind_dict)
            if i % 2 == 0:
                qc.append(bound_hc, range(self.N))
            else:
                qc.append(bound_hc.reverse_ops(), range(self.N))

            qc.rx(-2 * beta[i], range(self.N))

        creg = ClassicalRegister(self.N)
        qc.add_register(creg)

        for qidx, cidx in meas_map.items():
            qc.measure(qidx, cidx)

        if transpile_circ:
            qc = transpile(
                qc, self.backend, initial_layout=self.path, optimization_level=2
            )

        return qc

    @staticmethod
    def make_meas_map(circuit: QuantumCircuit) -> dict:
        """Return a mapping from qubit index (the key) to classical bit (the value).

        This allows us to account for the swapping order.
        """
        creg = circuit.cregs[0]
        qreg = circuit.qregs[0]

        meas_map = {}
        for inst in circuit.data:
            if inst.operation.name == "measure":
                meas_map[qreg.index(inst.qubits[0])] = creg.index(inst.clbits[0])

        return meas_map

    def get_local_expectation_values_from_counts(
            self,
            counts: Dict[str, int],
            all_to_all: bool = False,
    ):
        """Compute the expectation value of Z and ZZ.

        The expectation value of Z is computed for each node in the graph.
        The expectation value of ZZ is computed between each of the N(N-1)/2
        edges if `all_to_all` is `True` or between the edges of the graph
        if `all_to_all` is `False`.

        Args:
            counts: A counts dictionary of bitstring: num. times measured.
            all_to_all: Compute the correlators between all the edges if True.
        """
        local_exp_z = [0] * self.N
        if all_to_all:
            local_exp_zz = [0] * (self.N * (self.N - 1) // 2)
        else:
            local_exp_zz = [0] * len(self.G.edges)

        for bitstring, count in counts.items():
            bits = [int(bit) for bit in bitstring[::-1]]

            for i in range(self.N):
                local_exp_z[i] += count * (2 * bits[i] - 1)

            if all_to_all:
                idx = 0
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        local_exp_zz[idx] += self.G[i][j]['weight'] * (count * (2 * bits[i] - 1) * (2 * bits[j] - 1))
                        idx += 1
            else:
                for idx, (i, j, data) in enumerate(self.G.edges(data=True)):
                    local_exp_zz[idx] += data['weight'] * (count * (2 * bits[i] - 1) * (2 * bits[j] - 1))

        num_shots = sum(counts.values())
        local_exp_z = [val / num_shots for val in local_exp_z]
        local_exp_zz = [val / num_shots for val in local_exp_zz]

        return local_exp_z, local_exp_zz

    def evaluate_local_exp_on_device(self, theta, all_to_all=False):
        # Create QAOA circuit

        circ_swap = self.create_qaoa_circ_pauli_evolution(
            theta,
            superposition=True,
            random_cut=None,
            transpile_circ=True,
        )

        # Run on device
        watch_dog, success, counts_device = 0, False, dict()
        while watch_dog < 100 and not success:
            watch_dog += 1
            try:
                if self._sampler is not None:
                    job = self._sampler.run(circ_swap, shots=self.shots)
                    result = job.result()
                    counts_device = result.quasi_dists[0].binary_probabilities()
                else:
                    job = self.backend.run(circ_swap, shots=self.shots)
                    result = job.result()
                    counts_device = result.get_counts()

                success = True
                self.job_ids.append(job.job_id())
                self.counts.append(counts_device)
            except:
                pass

        if watch_dog >= 100 and not success:
            raise ValueError(f"{watch_dog} jobs in optimization failed.")

        local_exp_z, local_exp_zz = self.get_local_expectation_values_from_counts(
            counts_device, all_to_all
        )

        return local_exp_z, local_exp_zz

    def cost_noisy(self, theta):
        return sum(self.evaluate_local_exp_on_device(theta)[1])

    def cost_exact(self, theta):
        # Create QAOA circuit
        qc = self.create_qaoa_circ_pauli_evolution(
            theta,
            superposition=True,
            random_cut=None,
            transpile_circ=False,
        )

        # Transpile for the noiseless simulator
        noiseless_simulator = Aer.get_backend("qasm_simulator")

        # Run on the noiseless simulator
        result_exact = noiseless_simulator.run(qc, shots=self.shots).result()
        counts_exact = result_exact.get_counts()

        # Compute local expectation values from counts
        _, local_exp_ZZ = self.get_local_expectation_values_from_counts(counts_exact)

        return sum(local_exp_ZZ)
