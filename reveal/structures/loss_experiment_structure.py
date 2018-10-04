import reveal.parameters.constants as constants

class LossExperimentResultStructure():

    def __init__(self, net_structures, repetitions, activations):
        super(LossExperimentResultStructure, self).__init__()
        self.net_structures = net_structures
        self.repetitions = repetitions
        self.activations = activations
        self.results = {}

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

    def print_results(self, verbosity):
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
