import reveal.parameters.constants as constants
from reveal.util.verbosity import Verbosity
import torch
import reveal.util.io as io

class LossExperimentResults():

    def __init__(self, *args, **kwargs):

        super(LossExperimentResults, self).__init__()

        obj = kwargs.get("obj", None)

        if obj is None:
            self.net_structures = kwargs.get("net_structures", None)
            self.activation_fs = kwargs.get("activation_fs", None)
            self.repetitions = kwargs.get("repetitions", None)
            self.results = {}

            if self.net_structures is None:
                raise Exception('LossExperimentResultsMean constructor called without net_structures')

            if self.activation_fs is None:
                raise Exception('LossExperimentResultsMean constructor called without activation_fs')

            if self.repetitions is None:
                raise Exception('LossExperimentResultsMean constructor called without repetition')
            #initialize the results dictionaries
            for i, net_structure in enumerate(self.net_structures):
                self.results[i] = {}
                for repetition in range(self.repetitions):
                    self.results[i][repetition] = {}
                    for activation_f in self.activation_fs:
                        self.results[i][repetition][activation_f] = None

        else:
            self.net_structures = obj["net_structures"]
            self.activation_fs = obj["activation_fs"]
            self.results = obj["results"]
            self.repetitions = obj["repetitions"]



        self.read_me = ' This is an object of class LossExperimentResults \n\
        Members: \n \
        net_structures: List of Network Structures in the experiment \n \
        activation_fs: List of activation_f Functions in the experiment \n \
        repetitions: # of time each combination was run as an independent experiment \n \
        \n\
        results: A dictionary of dictionaries of losses for each combination organised as \n \
        {"net_structure_index": {"dictionary_of_reuslts_for_this_network_structure"}} \n\
        \n\
        dictionary_of_reuslts_for_network_structure = \n\
        {"#repetition : dictionary_of_results_for_this repetition"}\n\
        \n\
        dictionary_of_reuslts_for_repetition = \n\
        {"activation_function : n X 2 array of losses"}\n\
        where n are epochs\n\
        col 0 are training losses and col 1 is losses on test set\n\
        '



    def update_results(self, net_structure, repetition, activation_f, losses):

        self.results[net_structure][repetition][activation_f] = losses


    def print_results(self, verbosity = Verbosity(constants.VERBOSE_DETAILED_INFO)):
        for i, net_structure in enumerate(self.net_structures):
            verbosity.print(constants.VERBOSE_DETAILED_INFO,
            "=======================================",
            net_structure,
            "=======================================")
            for repetition in range(self.repetitions):
                verbosity.print(constants.VERBOSE_DETAILED_INFO,
                "\t------------------------------------------------",
                "\t"+str(repetition),
                "\t------------------------------------------------")

                for activation_f in self.activation_fs:
                    verbosity.print(constants.VERBOSE_DETAILED_INFO,
                    "\t\t------------------------------------------------",
                    "\t\t"+activation_f,
                    "\t\t------------------------------------------------",
                    self.results[i][repetition][activation_f])

    def get_average_over_repetitions(self):

        results = LossExperimentResultsMean(net_structures = self.net_structures, activation_fs = self.activation_fs)

        #accumulate results for all repetitions for each activation_f function
        for i, net_structure in enumerate(self.net_structures):
            for repetition in range(self.repetitions):
                for j, activation_f in enumerate(self.activation_fs):
                    results.update_results(i, activation_f, self.results[i][repetition][activation_f])

        results.convert_results_to_mean()
        return results

    def save(self, filePath):
        io.save(filePath, self.__dict__)

    def load(self, filePath):
        return io.load(filePath)

class LossExperimentResultsMean():

    def __init__(self, *args, **kwargs):

        super(LossExperimentResultsMean, self).__init__()

        obj = kwargs.get("obj", None)

        if obj is None:
            self.net_structures = kwargs.get("net_structures", None)
            self.activation_fs = kwargs.get("activation_fs", None)
            self.results = {}

            if self.net_structures is None:
                raise Exception('LossExperimentResultsMean constructor called without net_structures')
            if self.activation_fs is None:
                raise Exception('LossExperimentResultsMean constructor called without activation_fs')


            #initialize the results dictionaries
            for i, net_structure in enumerate(self.net_structures):
                self.results[i] = {}
                for activation_f in self.activation_fs:
                    self.results[i][activation_f] = None

        else:
            self.net_structures = obj["net_structures"]
            self.activation_fs = obj["activation_fs"]
            self.results = obj["results"]




        self.read_me = ' This is an object of class LossExperimentResultsMean \n\
        Members: \n \
        net_structures: List of Network Structures in the experiment \n \
        activation_fs: List of activation_f Functions in the experiment \n \
        \n\
        results: A dictionary of dictionaries of losses for each combination organised as \n \
        {"net_structure_index": {"dictionary_of_reuslts_for_this_network_structure"}} \n\
        \n\
        dictionary_of_reuslts_for_network_structure = \n\
        {"activation_function : n X 2 array of mean of losses over repetitions"}\n\
        where n are epochs\n\
        col 0 are training losses and col 1 is losses on test set\n\
        '

    def update_results(self, net_structure, activation_f, losses):

        #ininitialize dictionay if key doesn't exist for the net structure
        if self.results[net_structure][activation_f] is None:
            self.results[net_structure][activation_f] = losses.unsqueeze(0)

        else:
            self.results[net_structure][activation_f] = torch.cat((self.results[net_structure][activation_f], losses.unsqueeze(0)))

    def convert_results_to_mean(self):
        for i, net_structure in enumerate(self.net_structures):
            for activation_f in self.activation_fs:
                self.results[i][activation_f] = torch.mean(self.results[i][activation_f], dim = 0)

    def print_results(self, verbosity = Verbosity(constants.VERBOSE_DETAILED_INFO)):

        for i, net_structure in enumerate(self.net_structures):
            verbosity.print(constants.VERBOSE_DETAILED_INFO,
            "=======================================",
            net_structure,
            "=======================================")
            for activation_f in self.activation_fs:
                verbosity.print(constants.VERBOSE_DETAILED_INFO,
                "\t\t------------------------------------------------",
                "\t\t"+activation_f,
                "\t\t------------------------------------------------",
                self.results[i][activation_f])

    def save(self, filePath):
        io.save(filePath, self.__dict__)

    def load(self, filePath):
        return io.load(filePath)
