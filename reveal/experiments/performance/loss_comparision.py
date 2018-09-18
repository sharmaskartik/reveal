import reveal.data_utils.partition_data as partition
import reveal.data_utils.data_processor as dp
import reveal.parameters.validator as validator

#from reveal.networks.feedforward import *
import reveal.networks.resolve_network_factory as rnf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LossComparision():


    def __init__(self, *args, **kwargs):
        super(LossComparision, self).__init__()

        self.params = {}
        self.params['repetition'] = 10
        self.params['epochs'] = 10
        self.params['batch_size'] = 10

        self.params['problem_type'] = None
        self.params['net_structures'] = None
        self.params['network_type'] = None
        self.params['activation_fs'] = None
        self.params['loss_function'] = None
        self.params['optimizer'] = None


    def compare_loss(self, X, T):

        #check if the parameters are valid
        stat = validator._check_parameters(self.params)

        if stat < 0:
            return

        net_struc = self.params['net_structures'][1]

        net_factory  = rnf.resolve_network_factory(self.params['network_type'])
        net_class = net_factory(net_struc)
        complete_net_struc = [X.shape[1]] + net_struc +[T.shape[1]]

        activation_f = self.params['activation_fs'][1]

        net = net_class(complete_net_struc, activation_f)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        print(net)

        X, T = dp._convertFromNumpy([X, T], torch.FloatTensor)

        partitions = partition.partition(X, T, [0.8, 0.2], shuffle = True)

        partitions, unstd_t = dp._standardize_train_test(partitions)

        Xtrain, Ttrain, Xtest, Ttest = partitions

        n_train_samples = Xtrain.size()[0]

        batch_size = self.params['batch_size']
        for epoch in range(self.params['epochs']):  # loop over the dataset multiple times

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
