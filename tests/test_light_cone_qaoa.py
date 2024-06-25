"""Tests for the light-cone QAOA code."""

import networkx as nx
import numpy as np
import pytest

from qiskit_aer import AerSimulator

from large_scale_qaoa.light_cone_qaoa import LightConeQAOA
from large_scale_qaoa.graph_utils import build_paulis
from large_scale_qaoa.qaoa import ErrorMitigationQAOA


class TestLightConeQAOA:

    @pytest.fixture()
    def light_cone(self):
        """Create a graph to work with."""
        edges = [(i, i + 1) for i in range(10)]
        yield LightConeQAOA(nx.from_edgelist(edges))

    def test_sub_correlators(self, light_cone: LightConeQAOA):
        """Test that the construction of sub-correlators works."""
        edges = [(3, 4), (4, 5), (5, 6)]
        paulis, src_edge = light_cone.make_sub_correlators(edges, (4, 5))
        expected = [("ZZII", 1.0), ("IZZI", 1.0), ("IIZZ", 1.0)]

        assert set(paulis) == set(expected)
        assert src_edge == (1, 2)

    @pytest.mark.parametrize("graph", [nx.random_regular_graph(3, 12, seed=2), nx.complete_graph(n=5)])
    @pytest.mark.parametrize("theta", [[1, 1, 0, 1], [1, 2, 0, -0.5], [0, 0, 0, 0], [-1, -1, -2, -2]])
    def test_circuit(self, graph: nx.Graph, theta: list[float]):
        """Test that light-cone QAOA gives the same results as standard QAOA."""

        paulis = build_paulis(graph)

        lc_qaoa = LightConeQAOA(graph, shots=100000)
        em_qaoa = ErrorMitigationQAOA(100000, paulis, AerSimulator(method="automatic"))

        circ = em_qaoa.create_qaoa_circ_pauli_evolution(theta).decompose()
        counts = AerSimulator(method="automatic").run(circ, shots=100000).result().get_counts()

        _, exp_zz = em_qaoa.get_local_expectation_values_from_counts(counts)

        normal_val = sum(exp_zz)
        light_cone_val = lc_qaoa.depth_two_qaoa(theta)

        assert abs(light_cone_val - normal_val) <= 0.1

    @pytest.mark.parametrize("graph", [nx.random_regular_graph(3, 12, seed=2), nx.complete_graph(n=5)])
    @pytest.mark.parametrize("theta", [[1, 1, 0, 1], [1, 2, 0, -0.5], [0, 0, 0, 0], [-1, -1, -2, -2]])
    def test_circuit_weighted(self, graph: nx.Graph, theta: list[float]):
        """Test that weighted light-cone QAOA gives the same results as standard QAOA."""

        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = np.random.normal()

        paulis = build_paulis(graph)

        lc_qaoa = LightConeQAOA(graph, shots=100000)
        em_qaoa = ErrorMitigationQAOA(100000, paulis, AerSimulator(method="automatic"))

        circ = em_qaoa.create_qaoa_circ_pauli_evolution(theta).decompose()
        counts = AerSimulator(method="automatic").run(circ, shots=100000).result().get_counts()

        _, exp_zz = em_qaoa.get_local_expectation_values_from_counts(counts)

        normal_val = sum(exp_zz)
        light_cone_val = lc_qaoa.depth_two_qaoa(theta)

        assert abs(light_cone_val - normal_val) <= 0.1
