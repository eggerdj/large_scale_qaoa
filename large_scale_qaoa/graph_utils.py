import networkx as nx
from typing import List, Tuple


def build_graph(paulis: List[Tuple[str, float]]) -> nx.Graph:
    """Create a graph by parsing the pauli strings.

    Args:
        paulis: A list of Paulis given as tuple of Pauli string and
            coefficient. E.g., `[("IZZI", 1.0), ("ZIZI", 1.0)]`. Each
            pauli is guaranteed to have two Z's.

    Returns:
        A networkx graph.
    """
    graph = nx.Graph()
    edges = []
    for pauli_str, weight in paulis:
        edge = [idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"]
        edges.append((edge[0], edge[1], weight))
    graph.add_weighted_edges_from(edges)
    return graph


def build_paulis(graph: nx.Graph) -> List[Tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_graph`
    """
    pauli_list = []
    for u, v, data in graph.edges(data=True):
        paulis = ["I"] * len(graph)
        paulis[u], paulis[v] = "Z", "Z"
        coeff = data["weight"] if "weight" in data else 1.0
        pauli_list.append(("".join(paulis)[::-1], coeff))

    return pauli_list
