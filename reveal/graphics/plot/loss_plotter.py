import matplotlib.pyplot as plt
import numpy as np

class LossPlotter():

    def __init__(self, loss_experiment_results):
        '''
        Creates an object of LossPlotter.
        Parameters:

        loss_experiment_results : An object of LossExperimentResults class
                                  This object serves as the data for the plots
        '''

        super(LossPlotter, self).__init__()

        self.net_structures = loss_experiment_results.net_structures
        self.activation_fs = loss_experiment_results.activation_fs
        self.repetitions = loss_experiment_results.repetitions
        self.results = loss_experiment_results.results

        self.n_net_structures = len(self.net_structures)
        self.n_activation_fs  = len(self.activation_fs)

        self.params = {}

        self.train_label = "Training Loss"
        self.test_label = "Testing Loss"
        self.x_axis_label = "#Iterations"
        self.y_axis_label = "Losses"

        self.TYPE_ALL_REPETITIONS = 0
        self.TYPE_MEAN_ACTIVATIONS_TOGETHER = 1
        self.TYPE_MEAN_ACTIVATIONS_SEPARATE = 2


        self.train_idx = 0
        self.test_idx = 1

    def _init(self):
        '''
        Initialize the plot labels
        '''
        self.train_label = self.params.get("train_label", self.train_label)
        self.test_label = self.params.get("test_label", self.test_label)
        self.x_axis_label = self.params.get("x_axis_label", self.x_axis_label)
        self.y_axis_label = self.params.get("y_axis_label", self.y_axis_label)

    def _get_subplots(self, type):


        '''
        return the figures and axises for the subplots which based on type of plot

        for TYPE_ALL_REPETITIONS: n_net_structures X 2 * n_activation_fs subplots
                                (traing and testing loss for each activation function)

        for TYPE_MEAN_ACTIVATIONS_TOGETHER: n_net_structures X 2 subplots
                                (traing and testing loss for each activation function)


        for TYPE_MEAN_ACTIVATIONS_SEPARATE: n_net_structures X n_activation_fs subplots

        '''
        if type == self.TYPE_ALL_REPETITIONS:

            #create n_net_structures X 2 * n_activation_fs subplots
            #create 2 subplots for each activation - for training and testing losses

            fig, axs =  plt.subplots(self.n_net_structures, 2 * self.n_activation_fs,
            figsize = (5 * self.n_net_structures, 10 * self.n_activation_fs), sharex = True, sharey = True)

        elif(type == self.TYPE_MEAN_ACTIVATIONS_TOGETHER):
            #create n_net_structures X 2 - for training and testing losses

            fig, axs =  plt.subplots(self.n_net_structures, 2 ,
            figsize = (5 * self.n_net_structures, 10), sharex = True, sharey = True)

        elif(type == self.TYPE_MEAN_ACTIVATIONS_SEPARATE):
            #create n_net_structures X n_activation_fs subplots
            fig, axs =  plt.subplots(self.n_net_structures, self.n_activation_fs ,
            figsize = (5 * self.n_net_structures, 10), sharex = True, sharey = True)


        return fig, axs


    def get_plots(self, type, reps = None, alpha = 1 , show = False, save = False):
        '''

        return the figures and axises for the subplots based on type of plots

        PARAMETERS:

        type:
            TYPE_ALL_REPETITIONS: Creates n_net_structures X 2 * n_activation_fs subplots
                                  (traing and testing loss for each activation function)

                                  Each subplot has losses for all repetition for
                                  a particular combination of network structure and
                                  activation function.

            TYPE_MEAN_ACTIVATIONS_TOGETHER: Creates n_net_structures X 2 subplots
                                (traing and testing loss for each activation function)

                                First subplot in a row has training losses for all
                                activations functions for a particular structure and
                                the second subplot are the testing losses for same
                                network.


            TYPE_MEAN_ACTIVATIONS_SEPARATE: n_net_structures X n_activation_fs subplots

                                Each subplot in a row are training and testing losses
                                for an activations function. Each row  corresponds to
                                a particular network structure

        reps:
            Controls how many repetitions to plot for each subplot.
            Only has an effect for TYPE_ALL_REPETITIONS. Default = All

        alpha:
            controls the transparency of the lines in the subplot (0 - 1)
            0: transparency, 1: solid. Default = 1

        show:
            A boolean flag the controls whether or not to show the plots. The call is
            blocking. Default = False

        save:
            A boolean flag that controls whether or not to save the plot as an
            image file on disk. Default = False

        RETURN figure, axises

        USAGE:

        lp = LossPlotter(results)
        lp.params["train_label"] = "Training MSE"
        lp.params["test_label"] = "Testing MSE"
        lp.params["y_axis_label"] = "MSE"

        fig, ax = lp.get_plots(lp.TYPE_ALL_REPETITIONS, reps = 200, alpha = 0.4, show = True)

        fig, ax = lp.get_plots(lp.TYPE_MEAN_ACTIVATIONS_TOGETHER, alpha = 0.4, show = True)
        fig, ax = lp.get_plots(lp.TYPE_MEAN_ACTIVATIONS_SEPARATE, alpha = 0.4, show = True)
        '''

        self._init()
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

                elif type == self.TYPE_MEAN_ACTIVATIONS_TOGETHER:
                    self._plot_mean_losses_separate(axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, type, alpha = 1)
                    
                elif type == self.TYPE_MEAN_ACTIVATIONS_SEPARATE:
                    self._plot_mean_losses_separate(axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, type, alpha = 1)

        if show:
            plt.show()

        return plt, axs

    def _plot_losses_for_repetitions(self, axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, reps = None, alpha = 1):

        if reps is None or reps > losses.shape[0] or reps <= 0:
            reps = losses.shape[0]

        for i in range(reps):
            label = None

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

    def _plot_mean_losses_separate(self, axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, type, alpha = 1):
        #training losses
        loss = losses[:, self.train_idx]
        label = self.train_label
        ax = axs[n_net_struc][n_activation_f]
        title = net_structure + "_" + activation_f + "_" + self.train_label
        self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)


        #testing losses
        loss = losses[:, self.test_idx]
        label = self.test_label
        ax = axs[n_net_struc][n_activation_f]
        title = net_structure + "_" + activation_f + "_" + self.test_label

        self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)


    def _plot_mean_losses_together(self, axs, losses, n_net_struc, net_structure, n_activation_f, activation_f, type, alpha = 1):


        #training losses
        loss = losses[:, self.train_idx]
        label = activation_f
        ax = axs[n_net_struc][self.train_idx]
        title = net_structure + "_" +self.train_label

        self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)


        #testing losses
        loss = losses[:, self.test_idx]
        label = activation_f
        ax = axs[n_net_struc][self.test_idx]
        title = net_structure + "_" + self.test_label

        self.plot_losses_on_axis(ax, loss, label, self.x_axis_label, self.y_axis_label, title, alpha)



    def plot_losses_on_axis(self, ax, loss, label, x_axis_label, y_axis_label, title, alpha):
        if label is None:
            ax.plot(loss, alpha = alpha)
        else:
            ax.plot(loss, label = label, alpha = alpha)
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.legend()
