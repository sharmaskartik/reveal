import reveal.parameters.constants as constants
from reveal.util.verbosity import Verbosity
import torch
import reveal.util.io as io

class LossExperimentResults():

    def __init__(self, *args, **kwargs):

        super(LossExperimentResults, self).__init__()

        obj = kwargs.get("obj", None)

        if obj is None:
            self.net_structures = self.convert_networks_to_strings(kwargs.get("net_structures", None))
            self.activation_fs = kwargs.get("activation_fs", None)
            self.repetitions = kwargs.get("repetitions", None)
            self.results = {}
            self.convert_networks_to_strings(self.net_structures)
            if self.net_structures is None:
                raise Exception('LossExperimentResultsMean constructor called without net_structures')

            if self.activation_fs is None:
                raise Exception('LossExperimentResultsMean constructor called without activation_fs')

            if self.repetitions is None:
                raise Exception('LossExperimentResultsMean constructor called without repetition')

            #initialize the results dictionaries
            for net_structure in self.net_structures:
                self.results[net_structure] = {}
                for activation_f in self.activation_fs:
                    self.results[net_structure][activation_f] = { "all": None, "mean": None}
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
        {"net_structure_index": {"dictionary_of_results_for_this_network_structure"}} \n\
        \n\
        dictionary_of_results_for_network_structure = \n\
        {"activation_function" : "all" : r X n X 2 tensor of losses" \n\
                                "means": n X 2 tensor of mean of losses}\n\
        where r is # repetitions and n is #epochs\n\
        col 0 are training losses and col 1 is losses on test set\n\
        '



    def update_results(self, net_structure_idx, repetition, activation_f, losses):

        net_structure = self.net_structures[net_structure_idx]

        losses = losses.unsqueeze(0)
        results = self.results[net_structure][activation_f].get("all", None)
        if results is None:
            self.results[net_structure][activation_f]["all"] = losses
        else:
            self.results[net_structure][activation_f]["all"] = torch.cat((self.results[net_structure][activation_f]["all"], losses))


    def print_results(self, verbosity = Verbosity(constants.VERBOSE_DETAILED_INFO)):

        for net_structure in self.net_structures:
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
                self.results[net_structure][activation_f]["all"].size(),
                "\t\t Means ",
                self.results[net_structure][activation_f]["means"].size())

    def calculate_means(self):
        for net_structure in self.net_structures:
            for activation_f in self.activation_fs:
                self.results[net_structure][activation_f]["means"] = torch.mean(self.results[net_structure][activation_f]["all"], dim = 0)

    def save(self, filePath):
        io.save(filePath, self.__dict__)

    def load(self, filePath):
        return io.load(filePath)

    def convert_networks_to_strings(self, networks):

        '''
        function converts networks (a list of lists of elements) to a list of strings
        '''
        if networks is None:
            return None

        network_names = []

        for network in networks:
            name = ", ".join(str(i) for i in network)
            name = "[" + name + "]"
            network_names.append(name)

        return network_names
