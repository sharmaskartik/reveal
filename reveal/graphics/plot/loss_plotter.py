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

        self.train_label = "Training Loss"
        self.test_label = "Testing Loss"
        self.x_axis_label = "#Iterations"
        self.y_axis_label = "Losses"

        self.TYPE_ALL_REPETITIONS = 0
        self.TYPE_MEAN_ACTIVATIONS_TOGETHER = 1
        self.TYPE_MEAN_ACTIVATIONS_SEPARATE = 2


        self.train_idx = 0
        self.test_idx = 1

    def _get_subplots(self, type):

        if type == self.TYPE_ALL_REPETITIONS:

            #create n_net_structures X 2 * n_activation_fs subplots
            #create 2 subplots for each activation - for training and testing losses

            fig, axs =  plt.subplots(self.n_net_structures, 2 * self.n_activation_fs, figsize = (5 * self.n_net_structures, 10 * self.n_activation_fs), sharex = True, sharey = True)

        elif(type == self.TYPE_MEAN_ACTIVATIONS_TOGETHER):
            #create n_net_structures X 2 - for training and testing losses

            fig, axs =  plt.subplots(self.n_net_structures, 2 , figsize = (5 * self.n_net_structures, 10), sharex = True, sharey = True)

        elif(type == self.TYPE_MEAN_ACTIVATIONS_SEPARATE):
            #create n_net_structures X n_activation_fs subplots
            fig, axs =  plt.subplots(self.n_net_structures, self.n_activation_fs , figsize = (5 * self.n_net_structures, 10), sharex = True, sharey = True)


        return fig, axs

    def get_plots(self, type, reps = None, alpha = 1 , show = False, save = False):

        fig, axs = self._get_subplots(type)

        for n_net_struc, net_structure in enumerate(self.net_structures):

            for n_activation_f, activation_f in enumerate(self.activation_fs):

                if type == self.TYPE_ALL_REPETITIONS:
                    losses = self.results[net_structure][activation_f]["all"]
                else:
                    losses = self.results[net_structure][activation_f]["means"]

                losses = losses.numpy()

                if type == self.TYPE_ALL_REPETITIONS:
                    self._plot_losses_for_repetitions(axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, reps = reps, alpha = 1)
                else:
                    self._plot_mean_losses(axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, type, alpha = 1)

        if show:
            plt.show()

        return plt, axs

    def _plot_losses_for_repetitions(self, axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, reps = None, alpha = 1):

        if reps is None or reps > losses.shape[0]:
            reps = losses.shape[0]

        for i in range(reps):
            label = "Rep "+str(i)

            #training losses
            ax = axs[n_net_struc][2* n_activation_f + self.train_idx]
            loss = losses[i, :, self.train_idx]
            title = net_structure + "_" + activation_f + "_" + self.train_label
            self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)


            #testing losses
            ax = axs[n_net_struc][2* n_activation_f + self.test_idx]
            loss = losses[i, :, self.test_idx]
            title = net_structure + "_" + activation_f + "_" + self.test_label
            self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)


    def _plot_mean_losses(self, axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, type, alpha = 1):


        #training losses
        loss = losses[:, self.train_idx]

        if type == self.TYPE_MEAN_ACTIVATIONS_TOGETHER:
            label = activation_f
            ax = axs[n_net_struc][self.train_idx]
            title = net_structure + "_" +self.train_label
        else:
            label = self.train_label
            ax = axs[n_net_struc][n_activation_f]
            title = net_structure + "_" + activation_f + "_" + self.train_label

        self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)


        #testing losses
        loss = losses[:, self.test_idx]

        if type == self.TYPE_MEAN_ACTIVATIONS_TOGETHER:
            label = activation_f
            ax = axs[n_net_struc][self.test_idx]
            title = net_structure + "_" + self.test_label
        else:
            label = self.test_label
            ax = axs[n_net_struc][n_activation_f]
            title = net_structure + "_" + activation_f + "_" + self.test_label

        self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)



    def plot_losses_on_axis(self, ax, loss, label, x_axis_label, y_axis_label, title, alpha):
        ax.plot(loss, label = label, alpha = alpha)
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.legend()
