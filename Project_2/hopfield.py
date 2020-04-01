'''hopfield.py
Simulates a Hopfield network
CS443: Computational Neuroscience
YOUR NAMES HERE
Project 2: Content Addressable Memory
'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import random 


class HopfieldNet():
    '''A binary Hopfield Network that assumes that input components are encoded as bipolar values
    (-1 or +1).
    '''
    def __init__(self, data, orig_width, orig_height):
        '''HopfieldNet constructor

        Parameters:
        -----------
        data: ndarray. shape=(N, M). Each data sample is a length M bipolar vector, meaning that
            components are either -1 or +1. Example: [-1, -1, +1, -1, +1, ...]
        orig_width : int. Original width of each image before it was flattened into a 1D vector.
            If data are not images, this can be set to the vector length (number of features).
        orig_height : int. Original height of each image before it was flattened into a 1D vector.
            If data are not images, this can be set to 1.

        TODO:
        Initialize the following instance variables:
        - self.num_samps
        - self.num_neurons: equal to # features
        - self.orig_width, self.orig_height
        - self.energy_hist: Record of network energy at each step of the memory retrieval process.
            Initially an empty Python list.
        - self.wts: handled by `initialize_wts`
        '''
        self.num_samps= data.shape[0]
        self.wts = self.initialize_wts(data)
        self.orig_height = orig_height
        self.orig_width = orig_width
        self.energy_hist= []
        self.num_neurons= data.shape[1]

    def initialize_wts(self, data):
        '''Weights are initialized by applying Hebb's Rule to all pairs of M components in each
        data sample (creating a MxM matrix) and summing the matrix derived from each sample
        together.


        Parameters:
        -----------
        data: ndarray. shape=(N, M). Each data sample is a length M bipolar vector, meaning that
            components are either -1 or +1. Example: [-1, -1, +1, -1, +1, ...]

        Returns:
        -----------
        ndarray. shape=(M, M). Weight matrix between the M neurons in the Hopfield network.
            There are no self-connections: wts(i, i) = 0 for all i.

        NOTE: It might be helpful to average the weights over samples to avoid large weights.
        '''
        # for i in range(len(data)):
        #     wts = sum()


        #old wts (np.sum(data.T @ data)) / (self.num_samps)
        self.wts = ((data.T @ data)) / (self.num_samps)
        np.fill_diagonal(self.wts, 0)
        return self.wts

    def energy(self, netAct):
        '''Computes the energy of the current network state / activation

        See notebook for refresher on equation.

        Parameters:
        -----------
        netAct: ndarray. shape=(num_neurons,)
            Current activation of all the neurons in the network.

        Returns:
        -----------
        float. The energy.
        '''
        netAct = netAct[np.newaxis, :]
        # print(np.shape(netAct))
        # print(np.shape(self.wts))
        #en = (-.5)*(np.sum(np.sum(netAct.T @ self.wts @ netAct)))
        en = (-.5)*(np.sum(np.sum(netAct @ self.wts @ netAct.T)))
        return en 

    def predict(self, data, update_frac=0.1, tol=1e-15, verbose=False, show_dynamics=False):
        '''Use each data sample in `data` to look up the associated memory stored in the network.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features)
            Each data sample is a length M bipolar vector, meaning that components are either
            -1 or +1. Example: [-1, -1, +1, -1, +1, ...].
            May or may not be the training set.
        update_frac: float. Proportion (fraction) of randomly selected neurons in the network
            whose netAct we update on every time step.
            (on different time steps, different random neurons get selected, but the same number)
        tol: float. Convergence criterion. The network has converged onto a stable memory if
            the difference between the energy on the current and previous time step is less than `tol`.
        verbose: boolean. You should only print diagonstic info when set to True. Minimal print outs
            otherwise.
        show_dynamics: boolean. If true, plot and update an image of the memory that the network is
            retrieving on each time step.

        Returns:
        -----------
        ndarray. shape=(num_test_samps, num_features)
            Retrieved memory for each data sample, in each case once the network has stablized.

        TODO:
        - Process the test data samples one-by-one, setting them to as the initial netAct then
        on each time step only update the netAct of a random subset of neurons
        (size determined by `update_frac`; see notebook for refresher on update equation).
        Stop this netAct updating process once the network has stablized, which is defined by the
        difference betweeen the energy on the current and previous time step being less than `tol`.
        - When running your code with `show_dynamics` set to True from a notebook, the output should be
        a plot that updates as your netAct changes on every iteration of the loop.
        If `show_dynamics` is true, create a figure and plotting axis using this code outside the
        main update loop:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        Inside, plot an image of the current netAct and make the title the current energy. Then after
        your plotting code, add the following:
            display(fig)
            clear_output(wait=True)
            plt.pause(<update interval in seconds>)  # CHANGE THIS

        NOTE: Your code should work even if num_test_samps=1.
        '''
        if np.ndim(data) < 2:
            data = np.expand_dims(data, axis=0)

        for i in range(self.num_samps):
            netAct = data[i].copy() #copy of data object

            if len(self.energy_hist) < 2: #how do you deal with nergy comparisons when list is empty? 
                continue

            while self.energy_hist[-1] - self.energy_hist[-2] > tol: #check energy tolerance level
                #select random fraction of neurons
                numCells = int(round(update_frac * self.num_samps)) #round indicies 

                #get rand indices between 1 and numcells, without replacement 
                inds = np.random.randint(0, numCells)

                #update net activity of this fraction 
                netAct[inds]= np.sign(np.sum(netAct * self.wts[:,inds])) #check dimensions

                currEnergy = self.energy(netAct) #calculate energy 
                self.energy_hist.append(currEnergy)

            if show_dynamics == true: #plotting code, use code from notebook 

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(netAct)
                plt.set_title("Energy" + self.energy_hist[-1])
                display(fig)
                clear_output(wait=True)
                plt.pause(2)













