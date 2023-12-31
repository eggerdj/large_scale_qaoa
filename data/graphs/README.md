# Graph data

This folder contains the random-three-regular graph data used in the experiments.
Each folder contains 100 graphs in json format.
Each graph is generated with Netowrkx with `nx.random_regular_graph(d=3, n=num_nodes, seed=seed)`.
Here, `num_nodes` is the number of nodes of the graph.
The `seed` used for each graph is contained in the file name.
The number of SWAP layers needed to map the graph to a line of qubits using a line swap network after a SAT initial mapping is also contained in the file name.
The files contain the following:

* **Original graph** This is the graph generated by Networkx with the function `nx.random_regular_graph`.
  The graph is given as an edge list.
* **SAT mapping** is a permutation of the index of the nodes in the graph that minimizes the number of layers of SWAP gates of a line swap network needed to embed the graph in a line of qubits.
  This quantity is given as dict.
  The SAT mapping is found following the approach of Matsuo *et al.*, "A SAT approach to the initial mapping problem in SWAP gate insertion for commuting gates" [IEICE (2023)](https://doi.org/10.1587/transfun.2022EAP1159).
* **Paulis** This is a list of `[Pauli, coefficient]`. 
  Each edge `(i, j)` in a graph gives rise to a Pauli operator with a `Z` at index `i` and `j`.
  The `coefficient` is always `1` since we are not considering weighted problems. 
  Following Qiskit, the pauli strings are little Endian.
  Furthermore, the Paulis are given *after* the SAT mapping has been applied to the `Original graph`.
  Therefore, the Paulis can be used to directly build the cost-operator of QAOA.
* **min swap layers** This is the minimum number of swap layers that is needed to embed the graph in a line of qubits following a line swap network.
* **nx seed** is the seed provided to Networkx to generate the graph.

The paper by Sack & Egger studied one graph at each size.
These graphs were chosen to minimize the number of swap layers.