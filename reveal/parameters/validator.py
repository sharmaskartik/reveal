import reveal.parameters.constants as constants
import numpy as np

def validate_problem_type(p_type):
    #check if ptype is present in list of constants.ALL_PROBLEM_TYPES
    idxs = np.where(constants.ALL_PROBLEM_TYPES == p_type)[0]

    #if it exists , length of idxs should be greater than 0
    if len(idxs) > 0:
        return constants.VALIDATION_SUCCEEDED
    else:
        print("params['problem_type'] is not valid. It should be one of the following")
        print(constants.ALL_PROBLEM_TYPES)
        print("These are defined in reveal.parameters.constants.py")
        return constants.VALIDATION_FAILED


def validate_network_type(network_type):

    #check if network_type is present in list of constants.ALL_NETWORK_TYPES
    idxs = np.where(constants.ALL_NETWORK_TYPES == network_type)[0]

    #if it exists , length of idxs should be greater than 0
    if len(idxs) > 0:
        return constants.VALIDATION_SUCCEEDED
    else:
        print("params['network_type] is not valid. It should be one of the following")
        print(constants.ALL_NETWORK_TYPES)
        print("These are defined in reveal.parameters.constants.py")
        return constants.VALIDATION_FAILED


def validate_list_of_activavtion_functions(list_of_activation_functions):

    status = constants.VALIDATION_SUCCEEDED
    for i, activation_function in enumerate(list_of_activation_functions):
        #check if activation_function is present in list of constants.activation_functions
        val = constants.activation_functions.get(activation_function, -1)

        #if it doesn't exist in the dictionary, val would be -1
        if val == -1:
            status = constants.VALIDATION_FAILED
            print("Activation function: ", activation_function," at position ", i ," in params['activation_fs'] is not valid.")

    if status == constants.VALIDATION_FAILED:
        print("These should be one of the following")
        print(list(constants.activation_functions.keys()))

    return status

def validate_loss_function(loss_function):
    if loss_function is None:
        print("params['loss_function'] was not set")
        return constants.VALIDATION_FAILED
    else:
        return constants.VALIDATION_SUCCEEDED

def validate_optimizer(optimizer):
    if optimizer is None:
        print("params['optimizer'] was not set")
        return constants.VALIDATION_FAILED
    else:
        return constants.VALIDATION_SUCCEEDED

def validate_net_structures(net_structures):
    if net_structures is None:
        print("params['net_structures'] was not set")
        return constants.VALIDATION_FAILED
    else:
        return constants.VALIDATION_SUCCEEDED

def validate_int_parameter(param, param_name):
    if isinstance(param, int):
        return constants.VALIDATION_SUCCEEDED
    else:
        print(param_name + " must be integer")
        return constants.VALIDATION_FAILED


def _check_parameters(params):

    statuses = []

    stat = validate_problem_type(params['problem_type'])
    statuses.append(stat)

    stat = validate_net_structures(params['net_structures'])
    statuses.append(stat)

    stat = validate_network_type(params['network_type'])
    statuses.append(stat)

    stat = validate_list_of_activavtion_functions(params['activation_fs'])

    statuses.append(stat)

    stat = validate_loss_function(params['loss_function'])
    statuses.append(stat)

    stat = validate_optimizer(params['optimizer'])
    statuses.append(stat)

    stat = validate_int_parameter(params['repetition'],"params['repetition']")
    statuses.append(stat)

    stat = validate_int_parameter(params['batch_size'],"params['batch_size']")
    statuses.append(stat)

    stat = validate_int_parameter(params['epochs'],"params['epochs']")
    statuses.append(stat)




    statuses = np.array(statuses)
    idxs = np.where(statuses == constants.VALIDATION_FAILED)[0]
    if len(idxs) > 0:
        return constants.VALIDATION_FAILED
    else:
        return constants.VALIDATION_SUCCEEDED
