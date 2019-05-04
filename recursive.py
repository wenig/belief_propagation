%load_ext autoreload
%autoreload 2

import numpy as np
from node import Node


rain = Node("rain")
rain.cardinality = 2
rain.priors = np.array([0.8, 0.2]) #  no=0 yes=1

sprinkler = Node("sprinkler")
sprinkler.cardinality = 2
sprinkler.priors = np.array([0.9, 0.1]) #  no=0 yes=1

m = np.zeros((2, 2, 2)) #  rain, sprinkler, holmes' grass
m[1, 1, 1] = 1
m[0, 1, 1] = 0.9
m[0, 1, 0] = 0.1
m[1, 0, 1] = 1
m[0, 0, 0] = 1
holmes = Node("holmes")
holmes.cardinality = 2
holmes.m = m
holmes.likelihood = np.array([1, 1])

m = np.zeros((2, 2)) # rain, watson's grass
m[1, 1] = 1
m[0, 1] = 0.2
m[0, 0] = 0.8
watson = Node("watson")
watson.cardinality = 2
watson.m = m
watson.likelihood = np.array([1, 1])


holmes.add_parent(rain)
holmes.add_parent(sprinkler)
watson.add_parent(rain)

holmes.message_to_parent(rain)
holmes.message_to_parent(sprinkler)

rain.message_to_child(holmes)
sprinkler.message_to_child(holmes)

holmes.get_priors()
watson.get_priors()

# Holmes grass is wet

holmes.likelihood = np.array([0, 1])
holmes.message_to_parent(rain)
holmes.message_to_parent(sprinkler)

watson.get_belief()

# Watson's grass is also wet

watson.likelihood = np.array([0, 1])

sprinkler.get_belief()
