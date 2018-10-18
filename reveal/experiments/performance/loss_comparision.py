import reveal.data_utils.partition_data as partition
import reveal.data_utils.data_processor as dp
import reveal.parameters.validator as validator
from reveal.util.verbosity import Verbosity
from reveal.structures.loss_experiment_structure import LossExperimentResults
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
        self.params['repetitions'] = 10
        self.params['epochs'] = 10
        self.params['batch_size'] = 10
        self.params['verbosity'] = constants.VERBOSE_MIN_INFO

        self.params['problem_type'] = None
        self.params['net_structures'] = None
        self.params['network_type'] = None
        self.params['activation_fs'] = None
        self.params['loss_function'] = None
        self.params['optimizer'] = None
        self.params['use_cuda'] = False

    def getLoss(self, optimizer, criterion, net, X, T, unstd_t):
        optimizer.zero_grad()
        output, _ = net(X.unsqueeze(0))
        output = unstd_t(output)
        loss = criterion(output, T.unsqueeze(0))
        return loss

    def _init_variables(self):

        results = LossExperimentResults(net_structures = self.params['net_structures'],
                    repetitions = self.params['repetitions'],
                    activation_fs =self.params['activation_fs'])

        verbosity = Verbosity(self.params['verbosity'])

        criterion = nn.MSELoss()
        batch_size = self.params['batch_size']

        return results, verbosity, criterion, batch_size

    def compare_loss(self, X, T):

        #initiatilize results, verbosity, criterion objects
        results, verbosity, criterion, batch_size = self._init_variables()

        #check if the parameters are valid
        validator._check_parameters(self.params)

        verbosity.print(constants.VERBOSE_MED_INFO,
                            "*********************************",
                            "Beginning Experiment",
                            "*********************************")

        #Convert numpy arrays to Tensors
        X, T = dp._convertFromNumpy([X, T], torch.FloatTensor, self.params['use_cuda'])

        ###########################################################
        #                LOOP FOR NETWORK STRUCTURES
        ###########################################################

        for n_net_struc, net_struc in enumerate(self.params['net_structures']):

            #resolve network factory based on network type
            net_factory  = rnf.resolve_network_factory(self.params['network_type'])

            #create network object using the factory
            net_class = net_factory(net_struc)

            # add input layer and ouput layer to hidden layers
            complete_net_struc = [X.shape[1]] + net_struc +[T.shape[1]]

            verbosity.print(constants.VERBOSE_MED_INFO,
                                "\n-------------------------",
                                "Network Structure: \t" + str(net_struc),
                                "----------------------------\n")

            ###########################################################
            #                    LOOP FOR REPETITION
            ###########################################################
            for repetition in range(self.params['repetitions']):

                verbosity.print(constants.VERBOSE_MED_INFO,
                                    "Repetition: %i"%(repetition + 1),
                                    "----------------------------\n",
                                    tabs = 1)

                #partition data for current net structure
                #all activation functions will run on same partitions
                partitions = partition.partition(X, T, [0.8, 0.2], shuffle = True)
                partitions, unstd_t = dp._standardize_train_test(partitions)
                Xtrain, Ttrain, Xtest, Ttest = partitions

                if self.params['problem_type'] == constants.PROBLEM_TYPE_REGRESSION:
                    #un_standardized Ttrain is required to calculate the training loss after each epoch
                    unstd_Ttrain = unstd_t(copy.copy(Ttrain))
                else:
                    unstd_Ttrain = Ttrain

                n_train_samples = Xtrain.size()[0]

                ###########################################################
                #            LOOP FOR ACTIVATION FUNCTIONS
                ###########################################################
                for activation_f in self.params['activation_fs']:

                    verbosity.print(constants.VERBOSE_MED_INFO,
                                        "Activation Function: %s"%str(activation_f),
                                        "============================\n",
                                        tabs = 2)

                    #create network object
                    net = net_class(complete_net_struc, constants.activation_functions[activation_f])

                    #put net on GPU is self.params['use_cuda' is true]
                    if torch.cuda.is_available() and self.params['use_cuda']:
                        net = net.cuda()

                    #initiatilize optimizer
                    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


                    ###########################################################
                    #                        LOOP FOR EPOCHS
                    ###########################################################
                    for epoch in range(self.params['epochs']):  # loop over the dataset multiple times

                        running_loss = 0.0
                        for i in range(0, n_train_samples, batch_size):
                            if n_train_samples - i < batch_size:
                                end = n_train_samples
                            else:
                                end = i + batch_size

                            # add a dimention for batch
                            X_iter = Xtrain[i:end,:].unsqueeze(0)
                            T_iter = Ttrain[i:end,:].unsqueeze(0)
                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward + backward + optimize
                            outputs, _ = net(X_iter)
                            loss = criterion(outputs, T_iter)
                            loss.backward()
                            optimizer.step()

                        #calculate testing error after training on  current epoch
                        loss_train = self.getLoss(optimizer, criterion, net, Xtrain, unstd_Ttrain, unstd_t)

                        #calculate testing error  after training on current epoch
                        loss_test = self.getLoss(optimizer, criterion, net, Xtest, Ttest, unstd_t)

                        if epoch == 0:
                            losses = torch.FloatTensor((loss_train, loss_test)).view(-1, 2)
                        else:
                            losses = torch.cat((losses, torch.FloatTensor((loss_train, loss_test)).view(-1, 2)))

                        verbosity.print(constants.VERBOSE_DETAILED_INFO ,
                                        "Training error for epoch %i : \t %i " %(epoch, loss_train),
                                        tabs = 3)

                        verbosity.print(constants.VERBOSE_DETAILED_INFO ,
                                        "Testing error for epoch %i : \t %i " %(epoch, loss_test),
                                        "================================================\n",
                                        tabs = 3)
                    #epoch loop ends here

                    results.update_results(n_net_struc, repetition, activation_f, losses)

                    #print('Activation function done')

                # activation function loop ends here

                #print('Repetition done')
            #repetition loop ends here


            #print('net structure done')
        # net_structures loop ends here

        results.calculate_means()
        results.print_results(verbosity)
        #print("training finished")
        return results
        #end of method
