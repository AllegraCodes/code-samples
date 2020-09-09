"""
Part 6
Main function that reads in arguments from the command line.
Command line arguments: network file, query file, and # samples.

Synella Gonzales & Anthony Panetta
Python 3.7.0
"""

import sys
from BayesNet import BayesNet


if __name__ == "__main__":
    num_args = len(sys.argv) - 1
    if num_args != 3:
        print("Expected arguments: network_file, query_file, and num_samples")
        exit(1)

    # get network, query, and the number of samples
    network_file = sys.argv[1]
    query_file = sys.argv[2]
    num_samples = int(sys.argv[3])

    # construct the network
    print("Constructing network in", network_file)
    net = BayesNet(network_file)

    # assign node status
    print("Assigning statuses in", query_file)
    net.assign_nodes_status(query_file)

    # perform rejection sampling
    print("Rejection sampling...")
    rs = net.rejection_sampling(num_samples)

    # perform likelihood weighted sampling
    print("Likelihood weighted sampling...")
    lws = net.likelihood_sampling(num_samples)

    # report
    print("Estimated probability with", num_samples, "samples: ")
    print("Rejection:", rs)
    print("Weighted:", lws)
