from unittest import TestCase
import networkx as nx
import numpy as np

from large_scale_qaoa.graph_utils import build_graph, build_paulis


class TestGraphRoundTrip(TestCase):
    """Test that we can convert between graph and Paulis."""

    @staticmethod
    def _test_edge_equality(g: nx.Graph, h: nx.Graph):
        """Test equality of edges."""
        if len(g.edges) != len(h.edges):
            return False

        g_set = set(g.edges)
        for u, v, data in h.edges(data=True):
            edge = (u, v)
            if edge not in g_set and edge[::-1] not in g_set:
                return False
            else:
                weight_h = data['weight'] if 'weight' in data else 1
                weight_g = g[u][v]['weight'] if 'weight' in g[u][v] else 1
                if not weight_h == weight_g:
                    return False

        return True

    def test_round_trip(self):
        """Test that we can easily round-trip Pauli the graphs."""

        for seed in range(5):
            graph1 = nx.random_regular_graph(3, 10, seed=seed)
            graph2 = build_graph(build_paulis(graph1))

            self.assertTrue(self._test_edge_equality(graph1, graph2))

    def test_weighted_round_trip(self):
        """Test that we can easily round-trip weighted Pauli the graphs."""

        for seed in range(5):
            graph1 = nx.random_regular_graph(3, 10, seed=seed)
            for (u, v) in graph1.edges():
                graph1.edges[u, v]['weight'] = np.random.normal()

            graph2 = build_graph(build_paulis(graph1))

            self.assertTrue(self._test_edge_equality(graph1, graph2))
