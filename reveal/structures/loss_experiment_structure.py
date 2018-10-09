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
                for activation_f in self.activation_fs:
                    self.results[i][activation_f] = { "all": None, "mean": None}
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
        {"activation_function : "all" : r X n X 2 array of losses" \n\
                                "means": n X 2 array of mean of losses}\n\
        where r is # repetitions and n is #epochs\n\
        col 0 are training losses and col 1 is losses on test set\n\
        '



    def update_results(self, net_structure, repetition, activation_f, losses):

        losses = losses.unsqueeze(0)
        results = self.results[net_structure][activation_f].get("all", None)
        if results is None:
            self.results[net_structure][activation_f]["all"] = losses
        else:
            self.results[net_structure][activation_f]["all"] = torch.cat((self.results[net_structure][activation_f]["all"], losses))


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
                "\t\t All ",
                self.results[i][activation_f]["all"].size(),
                "\t\t Means ",
                self.results[i][activation_f]["means"].size())

    def calculate_means(self):
        for i, net_structure in enumerate(self.net_structures):
            for activation_f in self.activation_fs:
                self.results[i][activation_f]["means"] = torch.mean(self.results[i][activation_f]["all"], dim = 0)

    def save(self, filePath):
        io.save(filePath, self.__dict__)

    def load(self, filePath):
        return io.load(filePath)
