import reveal.data_utils.partition_data as partition
import reveal.data_utils.data_processor as dp
import reveal.parameters.validator as validator

#from reveal.networks.feedforward import *
import reveal.networks.resolve_network_factory as rnf
import reveal.parameters.constants as constants

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy

class LossComparision():


    def __init__(self, *args, **kwargs):
        super(LossComparision, self).__init__()

        self.params = {}
        self.params['repetition'] = 10
        self.params['epochs'] = 10
        self.params['batch_size'] = 10
        self.params['verbosity'] = constants.VERBOSE_MIN_INFO

        self.params['problem_type'] = None
        self.params['net_structures'] = None
        self.params['network_type'] = None
        self.params['activation_fs'] = None
        self.params['loss_function'] = None
        self.params['optimizer'] = None

    def getLoss(self, optimizer, criterion, net, X, T, unstd_t):
        optimizer.zero_grad()
        output, _ = net(X.unsqueeze(0))
        output = unstd_t(output)
        loss = criterion(output, T.unsqueeze(0)).item()
        return loss

    def compare_loss(self, X, T):

        #check if the parameters are valid
        stat = validator._check_parameters(self.params)

        if stat < 0:
            return

        criterion = nn.MSELoss()
        batch_size = self.params['batch_size']

        print("*********************************")
        print("Beginning Experiment")
        print("*********************************")

        X, T = dp._convertFromNumpy([X, T], torch.FloatTensor)

        for i, net_struc in enumerate(self.params['net_structures']):

            net_factory  = rnf.resolve_network_factory(self.params['network_type'])

            net_class = net_factory(net_struc)
            complete_net_struc = [X.shape[1]] + net_struc +[T.shape[1]]

            print("\n\n-------------------------")
            print("Network Structure ", net_struc)
            print("----------------------------\n\n")

            for repetition in range(self.params['repetition']):

                print("Repetition ", repetition + 1)
                print("\n\n----------------------------\n\n")

                #partition data for new structure*
                #all activation functions will run on same partitions
                partitions = partition.partition(X, T, [0.8, 0.2], shuffle = True)
                partitions, unstd_t = dp._standardize_train_test(partitions)
                Xtrain, Ttrain, Xtest, Ttest = partitions

                if self.params['problem_type'] == constants.PROBLEM_TYPE_REGRESSION:
                    #un_standardized Ttrain is required to calculate the MSE after each epoch
                    unstd_Ttrain = unstd_t(copy.copy(Ttrain))
                else:
                    unstd_Ttrain = Ttrain

                n_train_samples = Xtrain.size()[0]

                for activation_f in self.params['activation_fs']:

                    print("Activation Function ", activation_f)
                    print("\n\n----------------------------\n\n")

                    net = net_class(complete_net_struc, activation_f)
                    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
                        running_loss = 0.0

                        #calculate testing error for epoch
                        loss_train = self.getLoss(optimizer, criterion, net, Xtrain, unstd_Ttrain, unstd_t)
                        print("Training error for epoch %i : \t %i " %(epoch, loss_train))

                        #calculate testing error for epoch
                        loss_test = self.getLoss(optimizer, criterion, net, Xtest, Ttest, unstd_t)
                        print("Testing error for epoch %i : \t %i " %(epoch, loss_test))

                        print("================================================\n")

                        #epoch loop ends here

                print('Finished Training')
                # activation function loop ends here

            print('Finished Training')
            #repetition loop ends here
        # net_structures loop ends here

        print('Finished Training')
