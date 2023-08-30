"""A set of functions to efficiently simulate a depth-one QAOA."""

import numpy as np


def mixer(beta: float):
    """The mixer operator.

    Args:
        beta: The rotation angle of the mixer.
    """
    emb = np.exp(-1.0j * beta)
    eb = np.exp(1.0j * beta)

    mixer = np.array(
        [
            [0.5 * eb + 0.5 * emb, 0.5 * (eb - emb)],
            [0.5 * (eb - emb), 0.5 * emb + 0.5 * eb],
        ],
        dtype=complex,
    )

    return mixer


def correlator(i: int, j: int, graph: np.array, gamma: float, beta: float):
    """Computes the correlator <ZiZj>

    Args:
        i: The first index of the correlator.
        j: The second index of the correlator.
        graph: Graph in adjacency matrix format.
        gamma: The gamma of the QAOA.
        beta: The beta of the QAOA.
    """
    n = len(graph)

    # Initial state of the qubits as equal superposition.
    qi = np.array(
        [
            [np.sqrt(0.5)],
            [np.sqrt(0.5)],
        ],
        dtype=complex,
    )

    qj = np.array(
        [
            [np.sqrt(0.5)],
            [np.sqrt(0.5)],
        ],
        dtype=complex,
    )

    # Apply the U1 gates that come from qubit k neq i,j
    wi, wj = 0.0, 0.0
    for k in range(n):
        if k in {i, j}:
            continue

        wi += graph[i, k]
        wj += graph[j, k]

    qi[1] *= np.exp(1.0j * wi * gamma / 2)
    qj[1] *= np.exp(1.0j * wj * gamma / 2)

    # Switch to density matrices and compute the effect of
    # the two-qubit gate that comes from qubit k neq i,j
    rhoi, rhoj = qi * qi.T.conj(), qj * qj.T.conj()
    rhoij = np.kron(rhoi, rhoj)

    for k in range(n):
        if k in {i, j}:
            continue

        if graph[i, k] == 0.0 and graph[j, k] == 0.0:
            continue

        phasei = np.exp(-1.0j * gamma * graph[i, k])
        phasej = np.exp(-1.0j * gamma * graph[j, k])
        u1 = np.diag([1.0, phasej, phasei, phasei * phasej])

        rhoij = 0.5 * rhoij + 0.5 * np.dot(u1, np.dot(rhoij, u1.conj().T))

    # Apply the two-qubit term between `i` and `j`
    if graph[i, j] != 0.0:
        phase = np.exp(1.0j * gamma / 2.0 * graph[i, j])
        Uij = np.diag([1.0, phase, phase, 1.0])
        rhoij = np.dot(Uij, np.dot(rhoij, Uij.conj().T))

    # Apply the mixer operator
    mixeri = mixer(beta)
    mixerj = mixer(beta)

    mixerij = np.kron(mixeri, mixerj)
    rhoij = np.dot(mixerij, np.dot(rhoij, mixerij.conj().T))

    return np.real(rhoij[0, 0] - rhoij[1, 1] - rhoij[2, 2] + rhoij[3, 3])


def energy(graph: np.array, gamma: float, beta: float):
    """Evaluate the energy of a given gamma and beta by summing the correlators."""
    n = len(graph)
    energy_ = 0
    for i in range(n):
        for j in range(0, i):
            w_ij = graph[i, j]
            if w_ij != 0.0:
                energy_ += w_ij * correlator(i, j, graph, gamma, beta)

    return energy_
