import numpy as np
from functools import reduce
from utils import dot_T


class Node:
    def __init__(self, name):
        self.name = name
        self.cardinality = None
        self.likelihood = None
        self.priors = None
        self.belief = None
        self.parents = []
        self.children = []
        self.m = None

    def add_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def __str__(self):
        return self.name

    def message_to_parent(self, parent):
        """
        returns marginalized out parent message:
            - in m: group all entries by receiver parent values (all with 0 together, all with 1 together)
            - use other values in groups to get likelihood and messages from other parents
            - multiply those values in each group element
            - sum each group
        """
        likelihood = self.get_likelihood()
        parents_priors = np.array([p.message_to_child(self) for p in self.parents if p != parent])
        parent_i = self.parents.index(parent)

        stack = np.vstack([np.dot(self.m.take(r, axis=parent_i).transpose(), parents_priors.prod(axis=0)) for r in range(parent.cardinality)])

        return np.dot(stack, likelihood)

    def message_to_child(self, child):
        children_messages = np.array([c.message_to_parent(self) for c in self.children if c != child])
        if len(children_messages) > 0:
            unnormalized = (children_messages * self.get_priors()).prod(axis=0)
            message = unnormalized/unnormalized.sum()
            return message
        return self.get_priors()

    def get_likelihood(self):
        if self.likelihood is not None:
            return self.likelihood

        incoming_children_messages = np.array([c.message_to_parent(self) for c in self.children])
        return incoming_children_messages.prod(axis=0)

    def get_priors(self):
        if self.priors is not None:
            return self.priors

        parents_messages = [p.message_to_child(self) for p in self.parents]
        priors = reduce(np.dot, [self.m.transpose()]+parents_messages)
        return priors

    def get_belief(self):
        if self.belief is not None:
            return self.belief

        unnormalized = self.get_likelihood() * self.get_priors()
        return unnormalized/unnormalized.sum()
