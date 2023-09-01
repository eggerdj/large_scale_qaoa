
"""Helper functions to learn the map between noisy data and error mitigated data."""

from functools import lru_cache
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit import quantum_info as qi


@lru_cache(maxsize=None)
def local_exp(angle, q1, q2):
    """Slower implementation than above but speed is not crucial here."""
    circ = QuantumCircuit(2)
    if q1 == 1:
        circ.x(0)
    if q2 == 1:
        circ.x(1)
    circ.rx(angle, [0, 1])

    # Take the first row, it is the evolution of |00>
    state = qi.Operator(circ).data[:, 0]

    # The ZZ op, i.e., [1, -1, -1, 1], is applied in the difference.
    return sum(np.abs(state[idx]) for idx in [0, 3]) - sum(np.abs(state[idx]) for idx in [1, 2])


def get_local_exp_zz_fast(correlator_indices, flip_list: np.array, theta):
    """Computes the expectation values of the correlators.

    We assume the following:
    - The qubits start in either 0 or 1, given by the flip list.
    - The cost operator has Rz rotations replaced by barriers. This implies
      that it is essentially the identity.
    - The mixer rotations therefore sum up.
    - We compute the correlator of ZZ after this circuit.

    Notes:
        We should not start in an equal superposition since applying X to
        |+> has no effect. Furthermore, applying Y does not help much because
        |+> and |-> lie along X and are thus insensitive to the effect of the
        mixer. Since we replace the Rz gates by identities the whole layer of the
        cost operator is the identity. We can therefore sum up the rotations
        in the mixer and apply them individually.

    Args:
        correlator_indices:
        flip_list: A numpy array that has the same length as the number of
            qubits. Each entry is either 1 or 0 which indicates whether an
            X gate has been applied or not, respectively, on the |0> state.

    Returns:
        A list with the same length as `correlator_indices`.
    """
    beta_total = -2 * np.sum(theta[len(theta) // 2:])

    return [local_exp(beta_total, flip_list[i], flip_list[j]) for i, j in correlator_indices]


def get_data(p, qaoa_obj, num_training: int = 1):
    """Get training data for the NN

    Args:
        p: The depth of the QAOA.
        qaoa_obj: The QAOA object to generate circuits.
        num_training: The number of points to generate in one job.

    Returns:
        A list of X and Y values to train a neural network as well as the job id.
    """
    # Get random parameter and assign them to the circuit parameters

    circuits, thetas, random_cuts = [], [], []
    for _ in range(num_training):
        theta = 2 * np.pi * np.random.rand(2 * p)
        random_cut = np.random.randint(0, 2, qaoa_obj.N)

        circ_swap = qaoa_obj.create_qaoa_circ_pauli_evolution(
            theta, superposition=False, random_cut=random_cut, transpile_circ=True, remove_rz=True,
        )

        circ_swap.metadata = {
            "theta": theta.tolist(),
            "random_cut": theta.tolist(),
        }

        circuits.append(circ_swap)
        thetas.append(theta)
        random_cuts.append(random_cut)

    # Run on device
    counts = []
    if qaoa_obj.sampler is not None:
        job = qaoa_obj.sampler.run(circuits, shots=qaoa_obj.shots)
        result = job.result()
        for idx in range(num_training):
            counts.append(result.quasi_dists[idx].binary_probabilities())
    else:
        job = qaoa_obj.backend.run(circuits, shots=qaoa_obj.shots)
        result = job.result()
        for idx in range(num_training):
            counts.append(result.get_counts(idx))

    x_values, y_values = [], []
    for idx in range(num_training):
        # Compute observables from the counts (input data)
        all_z_noisy, all_zz_noisy = qaoa_obj.get_local_expectation_values_from_counts(
            counts[idx], all_to_all=True
        )

        # Compute the target
        graph_zz_exact = get_local_exp_zz_fast(
            np.array(qaoa_obj.G.edges()), random_cuts[idx], thetas[idx]
        )

        x_values.append(all_z_noisy + all_zz_noisy)
        y_values.append(graph_zz_exact)

    return x_values, y_values, job.job_id()
