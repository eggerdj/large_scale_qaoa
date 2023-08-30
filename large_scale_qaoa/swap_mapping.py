"""Functions to find a good line of qubits on a device."""

from typing import List, Optional, Tuple

import rustworkx as rx
from qiskit.transpiler import CouplingMap


def evaluate_path(path: List[int], backend, coupling_map) -> float:
    """Evaluate the fidelity of a path.

    The fidelity is the product of the two-qubit gates on the path.
    """
    two_qubit_fidelity = {}
    props = backend.properties()

    if "cx" in backend.configuration().basis_gates:
        gate_name = "cx"
    elif "ecr" in backend.configuration().basis_gates:
        gate_name = "ecr"
    else:
        raise ValueError("Could not identify two-qubit gate")

    for edge in coupling_map:
        try:
            cx_error = props.gate_error(gate_name, edge)
        except:
            cx_error = props.gate_error(gate_name, edge[::-1])

        two_qubit_fidelity[tuple(edge)] = 1 - cx_error

    if not path or len(path) == 1:
        return 0.0

    fidelity = 1.0
    for idx in range(len(path) - 1):
        fidelity *= two_qubit_fidelity[(path[idx], path[idx + 1])]

    return fidelity


def find_path(length: int, use_fidelity: bool, backend) -> Tuple[Optional[List[int]], int]:
    """Finds the best path based on the two-qubit gate error.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        A path and the number of evaluated paths.
    """
    coupling_map = CouplingMap(backend.configuration().coupling_map)
    coupling_map.make_symmetric()

    paths, size = [], coupling_map.size()

    for node1 in range(size):
        for node2 in range(node1 + 1, size):
            paths.extend(
                rx.all_simple_paths(
                    coupling_map.graph,
                    node1,
                    node2,
                    min_depth=length,
                    cutoff=length,
                )
            )

    if len(paths) == 0:
        return None, 0

    if not use_fidelity:
        return paths[0], len(paths)

    fidelities = [
        evaluate_path(path, backend, coupling_map.get_edges()) for path in paths
    ]

    # Return the best path sorted by fidelity
    return min(zip(paths, fidelities), key=lambda x: -x[1])[0], len(paths)
