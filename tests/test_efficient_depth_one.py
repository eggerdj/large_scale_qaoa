import networkx as nx
import numpy as np
from unittest import TestCase

from qiskit.providers.aer import AerSimulator

from large_scale_qaoa.qaoa import ErrorMitigationQAOA
from large_scale_qaoa.efficient_depth_one import energy


class TestEfficientDepthOne(TestCase):
    """Test the efficient depth on implementation.

    See also the depth_one_tests notebook.
    """

    @staticmethod
    def build_max_cut_graph(paulis: list[tuple[str, float]]) -> nx.Graph:
        """Create a graph by parsing the pauli strings.

        TODO: This could be outsourced from the tests and moved elsewhere.

        Args:
            paulis: A list of Paulis given as tuple of Pauli string and
                coefficient. E.g., `[("IZZI", 1.0), ("ZIZI", 1.0)]`. Each
                pauli is guaranteed to have two Z's.

        Returns:
            A networkx graph.
        """
        wedges = []
        for pauli_str, coeff in paulis:
            wedges.append([idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"] + [{"weight": coeff}])

        return nx.DiGraph(wedges)

    def test_energy_landscape(self):
        """Test the energy landscape over a few points."""
        paulis = [
            ("IIZZ", 1.0),
            ("IZIZ", 1.0),
            ("ZZII", 1.0),
            ("ZIIZ", 1.0),
        ]

        n_shots = 2 ** 14
        qaoa = ErrorMitigationQAOA(n_shots, paulis, AerSimulator(method="automatic"))

        w_graph = self.build_max_cut_graph(paulis)
        adj_mat = nx.adjacency_matrix(w_graph).toarray()
        adj_mat = adj_mat + adj_mat.T

        betas, gammas = np.linspace(0, np.pi, 10), np.linspace(0, np.pi, 10)
        energies1 = np.zeros((len(betas), len(gammas)))
        energies2 = np.zeros((len(betas), len(gammas)))

        for i, beta in enumerate(betas):
            for j, gamma in enumerate(gammas):
                energies1[i, j] = qaoa.cost_noisy([gamma, beta])
                energies2[i, j] = energy(adj_mat, gamma, beta)

        max_range = np.max(energies1) - np.min(energies1)
        max_diff = np.max(np.abs(energies1 - energies2))

        # Allow a tolerance due to the QASM nature of the circuit simulation.
        self.assertTrue(max_diff < (2 * max_range / np.sqrt(n_shots)))
