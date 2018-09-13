import numpy as np
import torch

def _standardize_unstandardize(X):
    means = X.mean(dim=0)
    stds = X.std(dim=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

def _standardize_train_test(partitions):

    Xtrain, Ttrain, Xtest, Ttest = partitions

    std_x, unstd_x = _standardize_unstandardize(Xtrain)
    std_t, unstd_t = _standardize_unstandardize(Ttrain)

    Xtrain = std_x(Xtrain)
    Xtest = std_x(Xtest)

    Ttrain = std_t(Ttrain)
    #Ttest = std_t(Ttest)

    return [Xtrain, Ttrain, Xtest, Ttest], unstd_t

def _convertFromNumpy(data, data_type):
    converted = []
    for tensor in data:
        converted.append(torch.from_numpy(tensor).type(data_type))
    return converted
