Synella Gonzales & Anthony Panetta

Files:  main.py
        BayesNet.py

This was built using Python 3.7.0 and no external libraries.

We chose Option B for Part 2, with no limit on number of parents.

Run with: python3 main.py [network_file] [query_file] [num_samples]
or similar for your environment

The Bayesian Network is represented by a list of Nodes.
Each Node records its incoming edges (i.e. its parents) as well as
its conditional probability table and information about its status.
The BayesNet constructor parses the network file and the class
includes functions to assign statuses to its nodes to form a query
and perform rejection and likelihood weighted sampling. The Node
class also contains functions to help the BayesNet functions.
