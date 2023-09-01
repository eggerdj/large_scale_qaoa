# Large scale QAOA
This repository contains the code associate to the paper by S. H. Sack and 
D. J. Egger, *"Large-scale quantum approximate optimization on non-planar 
graphs with machine learning noise mitigation"*, [arXiv:2307.14427](https://arxiv.org/abs/2307.14427) (2023).
This paper shows how to run non-planar graphs with up to forty nodes on cross-resonance based hardware.
This is done using Trottarized Quantum Annealing parameter initialization [1], a SAT initial mapping [2], 
and swap networks as described in Ref. [3]. 

## Installation

You can install this repository in editable mode with
```commandline
pip install -e .
```
Please note that requirements will not be managed for you.
You will have to install them yourself.
Note that among others you will need

* Qiskit optimization with CPLEX to solve max-cut problems `pip install 'qiskit-optimization[cplex]'`
* Qiskit Experiments to serialize and deserialize objects.

# Authors

This code was written by Stefan H. Sack and Daniel J. Egger.
The repository is the code associated to a paper and will thus
not be further developed.
Note that a repository containing best practices in quantum approximate optimization
is currently being developed, based partly on this code, and can be found at **HERE**.
This repository also contains tools such as the SAT mapping which is not included in this
repository.
Instead, we provide the graphs with and without the SAT initial mapping.

## References

[1] S. H. Sack & M. Serbyn, *"Quantum annealing initialization of the quantum approximate optimization algorithm"*, [Quantum **5**, 491](https://doi.org/10.22331/q-2021-07-01-491) (2021).

[2] A. Matsuo, S. Yamashita, D. J. Egger, *"A SAT approach to the initial mapping problem in SWAP gate insertion for commuting gates"*, [IEICE](https://doi.org/10.1587/transfun.2022EAP1159) (2023).

[3] J. Weidenfeller, *et al.*, *"Scaling of the quantum approximate optimization algorithm on superconducting qubit based hardware"*, [Quantum **6**, 870](https://doi.org/10.22331/q-2022-12-07-870) (2022).