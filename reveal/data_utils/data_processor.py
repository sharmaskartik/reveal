import numpy as np
import torch

def _standardize_unstandardize(X):
    """
      The function returns standardize and un_standardize functions for ndarray X.

      The intention for doing so is to call this function for training data and save
      the same means and standard deviation. So, calling the returned functions
      for testing data will use the saved means and standard deviation

      Usage: standardize_f, unstandardize_f = _standardize_unstandardize(X)

      params:
            X : a numpy ndarray of size n_samples x n_features.

      return:
            standardize_f(X1)  : a function that can be used to standardize
            ndarray X1

            unstandardize_f(X1): a function that can be used to un_standardize
            ndarray X1

            Both of these operations are done based on means and standard deviation
            of X.
      """

    #get means and standard deviation
    means = X.mean(dim=0)
    stds = X.std(dim=0)

    #define the functions
    def standardize(origX):
        return (origX - means) / stds

    def un_standardize(stdX):
        return stds * stdX + means

    #return the functions
    return standardize, un_standardize

def _standardize_train_test(partitions, classification = False):

    """
      The function returns the standardized partitions (a list of data and
      targets) and the un_standardize_target_f function. un_standardize_target_f
      is a function intented to be used for unstandardizing the network outputs.

      If classification flag is set to True, the function will NOT standardize
      target arrays and the return un_standardize_target_f will be None

      Usage:
      list_of_partitions, un_standardize_target_f = _standardize_train_test(partitions, classification)

      params:
            partitions : a list of 4 data partitions ::
            [data_matrix_train, target_matrix_train, data_matrix_test, target_matrix_test]

            classification : a boolean flag :: if set to True, the function will
            NOT standardize target arrays and the return un_standardize_target_f
            will be None

      return
            list_of_partitions : a list of 4 standardized data partitions in
            same order as input

            un_standardize_target_f : un_standardize function for targets
      """

    #extract partitions in individual variables
    Xtrain, Ttrain, Xtest, Ttest = partitions

    #get standardize and un_standardize functions
    std_x, unstd_x = _standardize_unstandardize(Xtrain)
    std_t, unstd_t = _standardize_unstandardize(Ttrain)

    #standardize both train and test data partition
    Xtrain = std_x(Xtrain)
    Xtest = std_x(Xtest)

    if classification:
        #set un_standardize_target_f to None if classification is set to True
        unstd_t = None
    else:
        # standardize target only if classification flag is set to False
        Ttrain = std_t(Ttrain)

    return [Xtrain, Ttrain, Xtest, Ttest], unstd_t


def _convertFromNumpy(list_of_np_arrays, tensor_type):

        """
          The function converts all numpy arrays in list_of_np_arrays into
          tensors of type tensor_type

          Usage: list_of_tensors = _convertFromNumpy(list_of_np_arrays, tensor_type):

          params:
                list_of_np_arrays : a list of numpy arrays
                tensor_type : datatype for result tensors

          return
                list_of_tensors : a list of tensors in same order as input
          """

    converted = []
    for numpy_array in list_of_np_arrays:
        converted.append(torch.from_numpy(numpy_array).type(tensor_type))
    return converted
