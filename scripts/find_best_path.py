"""Find good paths on the device."""

import sys
from datetime import datetime
import json

from qiskit.transpiler import CouplingMap

from qiskit_ibm_provider import IBMProvider

from large_scale_qaoa.swap_mapping import find_path, evaluate_path


if __name__ == "__main__":
    """Find a good path on the device and save it.
    
    Args:
        argv[1]: The name of the backend for which to find the best path.
        argv[2]: The number of qubits that the line should contain.
    """
    num_qubits = int(sys.argv[2])

    provider = IBMProvider()
    backend = provider.get_backend(str(sys.argv[1]))

    coupling_map = CouplingMap(backend.configuration().coupling_map)
    coupling_map.make_symmetric()

    path, num_paths = find_path(num_qubits, True, backend)
    fidelity = evaluate_path(path, backend, coupling_map)

    date = datetime.now().strftime("%Y-%m-%d")
    data = {
        "path": path,
        "fidelity": fidelity,
        "date": date,
        "backend name": backend.name,
        "number of paths": num_paths,
    }

    name = f"../data/qubit_paths/{backend.name}_{len(path)}nodes_{date}.json"
    with open(name, "w") as out_file:
        json.dump(data, out_file)
