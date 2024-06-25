"""Perform a naive light-cone simulation of QAOA."""

from typing import Dict, List, Tuple
import networkx as nx
import numpy as np

from qiskit import ClassicalRegister

from qiskit_aer import AerSimulator

from large_scale_qaoa.qaoa import ErrorMitigationQAOA


class LightConeQAOA:
    """Naive light-cone computation of depth-two QAOA.

    This simulation is described in Appendix B of https://arxiv.org/abs/2307.14427.
    The simulation is done without noise other than sampling noise.
    """

    def __init__(self, graph: nx.Graph, shots: int = 4096):
        """Initialize the simulation class.

        Args:
            graph: The graph we wish to simulate.
            shots: The number of shots that the QASM simulator will use.
        """
        self._graph = graph
        self.shots = shots

    def depth_two_qaoa(self, theta: List[float]) -> float:
        """Make the circuit for the light cone QAOA."""

        circuits, coeffs = [], []
        for u, v, data in self._graph.edges(data=True):
            circ = self.make_radius_two_circuit((u, v), theta).decompose()
            circuits.append(circ)
            coeff = data["weight"] if "weight" in data else 1.0
            coeffs.append(coeff)

        results = AerSimulator(method="automatic").run(circuits, shots=self.shots).result()

        observable = 0.0
        for counts, coeff in zip(results.get_counts(), coeffs):
            observable += coeff * self._sum_zz(counts)

        return observable

    def _sum_zz(self, counts: [Dict, str]) -> float:
        """Compute the expectation value of ZZ."""
        exp_zz = counts.get("00", 0.0)
        exp_zz -= counts.get("01", 0.0)
        exp_zz -= counts.get("10", 0.0)
        exp_zz += counts.get("11", 0.0)

        return exp_zz / sum(counts.values())

    def make_radius_two_circuit(self, edge: Tuple[int, int], theta: List[float]):
        """Create the circuit for the given edge."""
        ego1 = nx.generators.ego_graph(self._graph, edge[0], radius=2)
        ego2 = nx.generators.ego_graph(self._graph, edge[1], radius=2)

        # Edges we need to consider.
        edges = set(ego1.edges).union(set(ego2.edges))
        paulis, src_edge = self.make_sub_correlators(edges, edge)

        qaoa = ErrorMitigationQAOA(
            shots=4096,
            local_correlators=paulis,
            backend=AerSimulator(method="automatic"),
            path=list(range(len(paulis[0]))),
        )

        qaoa.basis_gates = ["rz", "sx", "h", "rzz", "x"]

        circuit = qaoa.create_qaoa_circ_pauli_evolution(theta, apply_swaps=False)
        circuit.remove_final_measurements()

        # Measure the qubits of the source edge.
        creg = ClassicalRegister(2)
        circuit.add_register(creg)
        circuit.measure(src_edge[0], 0)
        circuit.measure(src_edge[1], 1)

        return circuit

    def make_sub_correlators(self, edges, source_edge) -> Tuple[List[str], Tuple]:
        """Build a list of Pauli strings that identify where to place Rzz gates.

        From the edges we build correlators, to give to `ErrorMitigationQAOA`.
        These correlators are shorter than the correlators of the full graph.
        This is because we consider only the edges that are in the light-cone.
        We construct these reduced correlators by starting from full length
        pauli strings and then remove idle qubits.
        First, construct an array where each row is a correlator.
        Each I is a 0 and each Z is a 1. Therefore, each row sums to two.
        and the columns that do not sum to 0 have non-idle qubits.

        Args:
            edges: A set of edges for which we construct correlators that do not
                contain idle qubits.
            source_edge: The edge for which we want to compute the local correlator.

        Returns:
            A list of Pauli correlators as well as the index of the source edge. The
            index of the source edge is needed so that we know which qubits in the
            quantum circuit we should measure.
        """
        g_len = len(self._graph)

        correlators, masks, coeffs = [], [], []
        for edge in edges:
            u, v = edge[0], edge[1]
            mask = [0] * g_len
            paulis = ["I"] * g_len
            mask[u], mask[v] = 1, 1
            paulis[u], paulis[v] = "Z", "Z"
            correlators.append(paulis)
            masks.append(mask)
            coeff = self._graph[u][v]["weight"] if "weight" in self._graph[u][v] else 1.0
            coeffs.append(coeff)

        num_correl = len(correlators)
        correlators = np.array(correlators)

        indices = np.sum(masks, axis=0) > 0  # Columns that sum to 0 are idle qubits

        # Identify the new indices of the source edge so that we can later place measurements
        src_idx1 = source_edge[0] - sum(not idx for idx in indices[0:source_edge[0]])
        src_idx2 = source_edge[1] - sum(not idx for idx in indices[0:source_edge[1]])

        # The new length of the correlators is the number of columns that do not sum to 0.
        new_len = sum(indices)
        indices = np.tile(indices, num_correl).reshape(num_correl, len(indices))

        filtered_correl = correlators[indices].reshape((num_correl, new_len))

        paulis = []
        for pauli, coeff in zip(filtered_correl, coeffs):
            paulis.append(("".join(pauli)[::-1], coeff))

        return paulis, (src_idx1, src_idx2)
