"""A script to run the noise learning on large graphs."""

from datetime import datetime
import json
import networkx as nx
import numpy as np
import os
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
import sys
import uuid

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options

from qiskit_experiments.framework import ExperimentEncoder

from large_scale_qaoa.ml_denoising import get_data
from large_scale_qaoa.qaoa import ErrorMitigationQAOA


if __name__ == "__main__":
    """main code to rnu the error mitigation.

    Args:
        argv[1]: The name of the backend on which to run with a session.
        argv[2]: The date on which the best path was found. This us used to
            identify the file from which to load the line of qubits to use on the device.
            The date should be in format `yyyy-mm-dd`.
        argv[3]: The name of the file where the SAT mapped graph is stored. This file name
            should have the format `{X}nodes/graph_{Y}layers_{Z}seed.json`.
    """
    backend_name = str(sys.argv[1])
    path_date = str(sys.argv[2])
    graph_file = str(sys.argv[3])

    # Load graph
    graph_file = "../data/graphs/" + graph_file
    data = json.load(open(graph_file, "r"))
    graph = nx.from_edgelist(data["Original graph"])

    # Load pre-computed path of qubits to run on
    path_data = json.load(open(f"../data/qubit_paths/{backend_name}_{len(graph)}nodes_{path_date}.json", "r"))

    # Prepare file names
    uuid_tag = str(uuid.uuid4())[:8]
    date_tag = datetime.now().strftime('%Y%m%d')

    file_path = f"../data/results/{len(graph)}nodes/{date_tag}_{uuid_tag}_{backend_name}"

    service = QiskitRuntimeService()
    backend = service.get_backend(backend_name)

    # QAOA error mitigation
    em_qaoa = ErrorMitigationQAOA(
        shots=1024,
        path=path_data["path"],
        local_correlators=data["paulis"],
        backend=backend,
    )

    # TQA initialization parameters
    dt = 0.75
    p = 2
    grid = np.arange(1, p + 1) - 0.5
    init_params = np.concatenate((1 - grid * dt / p, grid * dt / p))
    training_data_size = 3000  # Num training samples
    train_job_size = 150  # Number of circuits to put in one training job.

    # Size of the NN hidden layer. As rule, we use the mean of input and output.
    # Output: we have N k / 2 edges in a RR3 graph
    # Input: we have N (N - 1) / 2 edges and N nodes in a RR3 graph
    n_hidden = int((em_qaoa.N * 3) / 4 + em_qaoa.N * (em_qaoa.N - 1) / 4 + em_qaoa.N / 2)

    # Create a file to save the training data.
    training_data_file = file_path + "_training_data.json"
    if not os.path.exists(training_data_file):
        with open(training_data_file, "w") as out_file:
            json.dump({"x": [], "y": [], "jobs": []}, out_file)

    saved_training_data = json.load(open(training_data_file, "r"))

    # Create a file to save the QAOA optimization data
    qaoa_data_file = file_path + "_qaoa_data.json"
    if not os.path.exists(qaoa_data_file):
        with open(qaoa_data_file, "w") as out_file:
            json.dump(
                {"mitigated_cost": [],
                 "noisy_cost": [],
                 "parameters": [],
                 "graph_file": graph_file,
                 "path": em_qaoa.path,
                 },
                out_file,
            )

    saved_qaoa_data = json.load(open(qaoa_data_file, "r"))

    # Save the properties of the backend for future reference
    # Can be loaded to dict with ExperimentDecoder().decode(...)
    props_file = file_path + "_backend_properties.json"
    with open(props_file, "w") as out_file:
        json.dump(ExperimentEncoder().encode(backend.properties().to_dict()), out_file)

    watch_dog = 0

    options = Options()
    options.transpilation.skip_transpilation = True

    with Session(service=service, backend=backend) as session:

        # The QAOA class takes care of transpilation for us.
        sampler = Sampler(options=options)
        em_qaoa.sampler = sampler

        print("Obtain training data from device!\n")

        # Generate training data in a robust way.
        while len(saved_training_data["x"]) < training_data_size:

            watch_dog += 1
            if watch_dog > 2 * training_data_size:
                raise ValueError(
                    f"Failed to get {training_data_size} learning samples. "
                    f"Max retires {watch_dog} exceeded."
                )

            try:
                X_is, Y_is, job_id = get_data(p, em_qaoa, train_job_size)

                for idx in range(len(X_is)):
                    saved_training_data["x"].append(X_is[idx])
                    saved_training_data["y"].append(Y_is[idx])
                    saved_training_data["jobs"].append(job_id)

                # Save each point as we generate them.
                with open(training_data_file, "w") as out_file:
                    json.dump(saved_training_data, out_file)
            except:
                pass

        # Train the NN using the training data
        regr = MLPRegressor(
            random_state=1,
            learning_rate="adaptive",
            batch_size=5,
            hidden_layer_sizes=n_hidden,
            activation="logistic",
            max_iter=3000,
        ).fit(saved_training_data["x"], saved_training_data["y"])
        em_qaoa.regr = regr

        # Save the params and loss curve of the regressor
        regressor_data_file = file_path + "_regressor_data.json"
        regressor_data = {}
        regressor_data.update(regr.get_params())
        regressor_data["loss_curve"] = regr.loss_curve_
        with open(regressor_data_file, "w") as out_file:
            json.dump(regressor_data, out_file)

        em_qaoa.shots = 4096


        def mitigated_cost(theta):
            """Error mitigated objective value"""
            local_exp_z, local_exp_zz = em_qaoa.evaluate_local_exp_on_device(
                theta, all_to_all=True
            )

            return sum(em_qaoa.regr.predict([local_exp_z + local_exp_zz])[0])


        def save_data(x):
            """Save data at each iteration to a file."""
            mit_cost = mitigated_cost(x)
            noisy_cost = em_qaoa.cost_noisy(x)

            saved_qaoa_data["mitigated_cost"].append(mit_cost)
            saved_qaoa_data["noisy_cost"].append(noisy_cost)
            saved_qaoa_data["parameters"].append([_ for _ in x])

            print(mit_cost, noisy_cost, x)

            with open(qaoa_data_file, "w") as qaoa_file:
                json.dump(saved_qaoa_data, qaoa_file)


        print("Performing minimization\n")

        result = minimize(
            mitigated_cost,
            x0=init_params,
            method="COBYLA",
            callback=save_data,
            options={"maxiter": 100},
        )

        saved_qaoa_data["result"] = {
            "x": result.x.tolist(),
            "message": result.message,
            "success": int(result.success),
            "status": result.status,
            "fun": result.fun,
            "nfev": result.nfev,
            "maxcv": result.maxcv,
        }

        saved_qaoa_data["counts"] = em_qaoa.counts
        saved_qaoa_data["job_ids"] = em_qaoa.job_ids

        with open(qaoa_data_file, "w") as qaoa_file:
            json.dump(saved_qaoa_data, qaoa_file)

        session.close()
