import reveal.parameters.constants as constants
from reveal.util.verbosity import Verbosity

class LossExperimentResultStructure():

    def __init__(self, net_structures, repetitions, activations):
        super(LossExperimentResultStructure, self).__init__()
        self.net_structures = net_structures
        self.repetitions = repetitions
        self.activations = activations
        self.results = {}

        self.read_me = ' This is an object of class LossExperimentResultStructure \n\
        Members: \n \
        net_structures: List of Network Structures in the experiment \n \
        activations: List of Activation Functions in the experiment \n \
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

        #initialize the results dictionaries
        for i, net_structure in enumerate(net_structures):
            self.results[i] = {}
            for repetition in range(repetitions):
                self.results[i][repetition] = {}
                for activation in activations:
                    self.results[i][repetition][activation] = {}


    def update_results(self, net_structure, repetition, activation, losses):

        #ininitialize dictionay if key doesn't exist for the net structure
        self.results[net_structure][repetition][activation] = losses


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

                for activation in self.activations:
                    verbosity.print(constants.VERBOSE_DETAILED_INFO,
                    "\t\t------------------------------------------------",
                    "\t\t"+activation,
                    "\t\t------------------------------------------------",
                    self.results[i][repetition][activation])


                
