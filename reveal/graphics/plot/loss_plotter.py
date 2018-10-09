import matplotlib.pyplot as plt
import numpy as np

class LossPlotter():

    def __init__(self, loss_experiment_results):

        super(LossPlotter, self).__init__()

        self.net_structures = loss_experiment_results.net_structures
        self.activation_fs = loss_experiment_results.activation_fs
        self.repetitions = loss_experiment_results.repetitions
        self.results = loss_experiment_results.results

        self.n_net_structures = len(self.net_structures)
        self.n_activation_fs  = len(self.activation_fs)


    def show_plots_for_repetitions(self, reps = 2, alpha = 1, show = False, save = False):

        #create n_net_structures X 2 * n_activation_fs subplots
        #create 2 subplots for each activation - for training and testing losses

        fig, axs =  plt.subplots(self.n_net_structures, 2 * self.n_activation_fs, figsize = (5 * self.n_net_structures, 10 * self.n_activation_fs), sharex = True, sharey = True)

        train_idx = 0
        test_idx = 1

        for n_net_struc, net_structure in enumerate(self.net_structures):

            for n_activation_f, activation_f in enumerate(self.activation_fs):

                losses = self.results[n_net_struc][activation_f]["all"]
                losses = losses.numpy()

                for i in range(reps):

                    #training losses
                    ax = axs[n_net_struc][2* n_activation_f + train_idx]
                    ax.plot(losses[i, :, train_idx], label = "rep "+str(i), alpha = alpha)
                    ax.set_title(str(n_net_struc) + "  " + activation_f + " train")
                    ax.legend()

                    #testing losses
                    ax = axs[n_net_struc][2* n_activation_f + test_idx]
                    ax.plot(losses[i, :, test_idx], label = "rep "+str(i), alpha = alpha)
                    ax.set_title(str(n_net_struc) + "  " + activation_f + " test")
                    ax.legend()

        if show:
            plt.show(block = False)

        return fig, axs
