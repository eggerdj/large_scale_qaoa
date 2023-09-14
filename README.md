# Large scale QAOA
This repository contains the code associate to the paper by S. H. Sack and 
D. J. Egger, *"Large-scale quantum approximate optimization on non-planar 
graphs with machine learning noise mitigation"*, [arXiv:2307.14427](https://arxiv.org/abs/2307.14427) (2023).
This paper shows how to run non-planar graphs with up to forty nodes on IBM Quantum hardware.
This is done using Trottarized Quantum Annealing parameter initialization [1], a SAT initial mapping [2], 
and swap networks as described in Ref. [3]. 

## Installation

You can install this repository in editable mode with
```commandline
pip install -e .
```
Please note that requirements will not be managed for you.
You will have to install them yourself.
Note that in addition to `numpy` and `qiskit` you will need

* Qiskit optimization with CPLEX to solve max-cut problems `pip install 'qiskit-optimization[cplex]'`
* Qiskit Experiments to serialize and deserialize objects.
* sklearn for the error mitigation.

## Authors

This code was written by Stefan H. Sack and Daniel J. Egger.
The repository is the code associated to a paper and will thus
not be further developed.
Note that a repository containing best practices in quantum approximate optimization
is currently being developed, based partly on this code, and will be available soon.
This repository also contains tools such as the SAT mapping which is not included in this
repository.
Instead, we provide the graphs with and without the SAT initial mapping.

## Tutorials

This repository provides the following tutorials:

* [Simulations](https://github.ibm.com/DEG/large_scale_qaoa_paper/blob/main/notebooks/simulations.ipynb) an example notebook showing how to run the methods of the paper on a simulator.
* [Result plotting](https://github.ibm.com/DEG/large_scale_qaoa_paper/blob/main/notebooks/plot_results.ipynb) to plot the data that has been saved.

## Scripts

This repository provides the scripts with which the data in the large-scale QAOA paper were gathered. These scripts include

* [Find best path](https://github.ibm.com/DEG/large_scale_qaoa_paper/blob/main/scripts/find_best_path.py) which should be run first to find the best line of qubits with which to run.
* [Large-scale QAOA](https://github.ibm.com/DEG/large_scale_qaoa_paper/blob/main/scripts/large_scale_qaoa.py) which performs the training of a neural network for noise mitigation and then runs QAOA on a given graph.

## References

[1] S. H. Sack & M. Serbyn, *"Quantum annealing initialization of the quantum approximate optimization algorithm"*, [Quantum **5**, 491](https://doi.org/10.22331/q-2021-07-01-491) (2021).

[2] A. Matsuo, S. Yamashita, D. J. Egger, *"A SAT approach to the initial mapping problem in SWAP gate insertion for commuting gates"*, [IEICE](https://doi.org/10.1587/transfun.2022EAP1159) (2023).

[3] J. Weidenfeller, *et al.*, *"Scaling of the quantum approximate optimization algorithm on superconducting qubit based hardware"*, [Quantum **6**, 870](https://doi.org/10.22331/q-2022-12-07-870) (2022).
