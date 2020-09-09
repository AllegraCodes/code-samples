"""
Part 1
Includes BayesNet and Node classes to form Bayesian networks
and functions to build and sample the networks.

Synella Gonzales & Anthony Panetta
Python 3.7.0
"""

import random


class BayesNet:
    def __init__(self, filename):
        """
        Part 2B (extra credit)
        Constructs the Bayesian network described in the file filename
        :param filename: the file to parse
        :return: the Bayesian network
        """
        self.nodes = []  # the nodes in this net; each node contains its incoming edges
        self.query = None  # the query variable
        self.evidence = []  # the evidence variables

        # make the nodes
        parent_stash = []  # keep track of the parents while making the nodes
        with open(filename, 'r') as file:  # each line describes a node
            line_number = 0
            for line in file:  # make each node
                breaks = line.split("[")  # split string into parts
                node_name = breaks[0].split(":")[0]  # get node name
                parents = breaks[1].split("]")[0].split(" ")  # split parent block
                probability = breaks[2].split("]")[0].split(" ")  # split probability block

                # convert probabilities to a number
                prob_vals = []
                for val in probability:
                    prob_vals.append(float(val))

                # stash parents
                current_parents = []
                for p in parents:
                    current_parents.append(p)
                parent_stash.append(current_parents)

                # finally, make the node
                self.nodes.append(Node(node_name, prob_vals))
                line_number += 1

        # nodes are created, now add edges, i.e. parents
        for node, parent_names in zip(self.nodes, parent_stash):
            parent_list = []  # holds the Node objects that the names refer to
            for name in parent_names:  # look for each parent
                for n in self.nodes:  # check each node until it is found
                    if n.name == name:  # we found the right node
                        parent_list.append(n)
                        break  # move on to the next parent
            node.parents = parent_list

    def assign_nodes_status(self, filename):
        """
        Part 3
        Assigns the status of each node in the network.
        After assignment the nodes are sorted for sampling.
        Possible statuses are:  evidence observed true (t),
                                evidence observed false (f),
                                query variable (? or q),
                                neither evidence nor query (-)
        :param filename: name of file containing node statuses (separated by commas)
                         the statuses must be given in the same order as the nodes
        """
        with open(filename) as file:
            line = file.read()
        statuses = line.split(',')
        for s, n in zip(statuses, self.nodes):
            n.assign_status(s)
            if s == 't' or s == 'f':  # keep track of evidence
                self.evidence.append(n)
            elif s == '?' or s == 'q':  # keep track of the query
                self.query = n
        self.topological_sort()

    def topological_sort(self):
        """
        Sorts the nodes of this graph so parents are before children.
        """
        ordered = []
        while len(self.nodes) > 0:
            node = self.nodes.pop(0)
            # if any parent is unordered move on
            all_parents_ordered = True  # initialize true, look for failure
            for pn in node.parents:
                if self.nodes.count(pn) > 0:  # the parent node is still unordered
                    all_parents_ordered = False
                    break
            if all_parents_ordered:  # all parents (if any) are already in ordered, so this node is ready too
                ordered.append(node)
            else:  # a parent is still unordered, so put this node back
                self.nodes.append(node)
        # replace in order
        for node in ordered:
            self.nodes.append(node)

    def rejection_sampling(self, num_samples):
        """
        Part 4
        Function that does rejection sampling
        on the Bayesian network.
        :param num_samples: number of samples
        :return: est. probability
        """
        trues = 0
        total = 0

        for i in range(num_samples):
            # generate a sample
            for node in self.nodes:
                node.sample()

            # reject or accept the sample
            rejected = False
            for node in self.evidence:
                if node.value != node.evidence_value:  # reject
                    rejected = True
                    break

            # count the sample
            if not rejected:
                total += 1
                if self.query.value:
                    trues += 1

        if total == 0:  # don't divide by 0
            total = 1
        return trues / total

    def likelihood_sampling(self, num_samples):
        """
        Part 5
        Performs likelihood-weighted sampling on the network
        :param num_samples: the number of samples to take
        :return: the approximate probability that the query variable is true
        """
        true_weight = 0
        total_weight = 0
        for i in range(num_samples):
            sample, weight = self.weighted_sample()
            total_weight += weight
            if sample:
                true_weight += weight
        return true_weight / total_weight

    def weighted_sample(self):
        """
        Part 5 helper function
        Generates a single weighted sample from the Bayesian network
        The weight is the product of the probabilities of the evidence
        :return: tuple containing a sample value for the query and the sample weight
        """
        weight = 1
        query = False

        for node in self.nodes:  # nodes must be ordered so parents are before children
            if node.is_evidence:  # weight evidence variables by their probability
                weight *= node.probability(node.evidence_value)
            else:  # assign non-evidence variables based on their probability
                sample = node.sample()
                if node.is_query:
                    query = sample
        return query, weight


class Node:
    def __init__(self, name, probabilities):
        self.name = name
        self.cpt = probabilities
        self.parents = []
        self.is_query = False
        self.is_evidence = False
        self.evidence_value = None
        self.value = None

    def assign_status(self, status):
        """
        Part 3 helper function
        Assigns the status of this node
        :param status: the status of this node
                        t = evidence observed true
                        f = evidence observed false
                        ? or q = query
        """
        if status == "t":
            self.is_evidence = True
            self.evidence_value = True
            self.value = True
        elif status == "f":
            self.is_evidence = True
            self.evidence_value = False
            self.value = False
        elif status == "?" or status == "q":
            self.is_query = True

    def probability(self, t=True):
        """
        Finds the probability this variable is true given its parents
        This is given by the CPT
        If t is false then find the probability of being false
        :param t: find the probability that this node = t
        :return: probability this variable is t
        """
        prob_true = self.cpt[self.cpt_index()]
        if t:
            return prob_true
        else:
            return 1 - prob_true

    def sample(self):
        """
        Samples this variable based on its conditional probability and sets the value
        :return: True or False, whichever is sampled
        """
        index = self.cpt_index()
        prob_true = self.cpt[index]
        unit_sample = random.random()  # sample of uniform distribution over [0, 1)
        val = unit_sample < prob_true
        self.value = val
        return val

    def cpt_index(self):
        """
        Finds the index to use in the CPT based on the values of its parents.
        This index is the value of the binary number where each digits represents a parent
        :return: index to use in self.cpt to get the proper probability
        """
        index = 0  # build this based on values of parents
        for parent, place in zip(self.parents, range(len(self.parents))):
            digit = parent.value  # true=1, false=0
            index += digit * (2 ** place)  # binary place value
        return index
