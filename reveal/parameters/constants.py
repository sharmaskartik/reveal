import numpy as np
import torch.nn.functional as F

#Network Strucure Types
NETWORK_TYPE_FEEDFORWARD = 0
ALL_NETWORK_TYPES = np.array([NETWORK_TYPE_FEEDFORWARD])

#Problem Types
PROBLEM_TYPE_REGRESSION = 0
PROBLEM_TYPE_CLASSIFICATION = 1

ALL_PROBLEM_TYPES = np.array([PROBLEM_TYPE_REGRESSION, PROBLEM_TYPE_CLASSIFICATION])

#Activation Functions
activation_functions = {}
activation_functions["relu"] = F.relu
activation_functions["tanh"] = F.tanh
activation_functions["sigmoid"] = F.sigmoid


#parameter validation constants
VALIDATION_FAILED = -1
VALIDATION_SUCCEEDED = 1


#Verbose levels
VERBOSE_NONE = -1
VERBOSE_DETAILED_INFO = 0
VERBOSE_MED_INFO = 1
VERBOSE_MIN_INFO = 2
VERBOSE_WARNING = 3
VERBOSE_ERROR  = 4
VERBOSE_SEVERE = 5
