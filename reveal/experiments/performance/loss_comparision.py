import reveal.data_utils.partition_data as partition
import reveal.data_utils.data_processor as dp
from reveal.networks.feedforward import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def compareActivations(X, T, list_of_activations, n_iterations, batch_size):

    net = TwoLayerFF([X.shape[1],10,10,T.shape[1]], F.relu)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print(net)

    X, T = dp._convertFromNumpy([X, T], torch.FloatTensor)

    partitions = partition.partition(X, T, [0.8, 0.2], shuffle = True)

    partitions, unstd_t = dp._standardize_train_test(partitions)

    Xtrain, Ttrain, Xtest, Ttest = partitions

    n_train_samples = Xtrain.size()[0]
    for epoch in range(n_iterations):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(0, n_train_samples, batch_size):
            if n_train_samples - i < batch_size:
                end = n_train_samples
            else:
                end = i + batch_size

            X_iter = Xtrain[i:end,:].unsqueeze(0)
            T_iter = Ttrain[i:end,:].unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = net(X_iter)
            loss = criterion(outputs, T_iter)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(running_loss)
        running_loss = 0.0

    print('Finished Training')


    print('beginning testing')


    output_test, _ = net(Xtest.unsqueeze(0))
    output_test = unstd_t(output_test)
    mse = criterion(output_test, Ttest.unsqueeze(0)).item()
    print("test error", mse)
