from drifttest import ImageDrift

import torch 
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from torchvision import transforms
from torch import nn

class DriftExperiment: 
    def __init__(self, name, parameterlist): # pass a list of tuple

        self.name = name
        self.save_folder = os.path.sep.join(["./experiments", name])
        os.mkdir(self.save_folder)
        print("Created Folder.")
        torch.set_grad_enabled(False)
        print("Loading Training Data...")
        self.training_data = torch.load(os.path.sep.join(["tensors", "cocotrain.pt"]))
        print("Training Data Loaded.")
        self.driftransforms = {
            "default" : nn.Identity(),
            "gaussianBlur" : transforms.GaussianBlur(kernel_size=21, sigma=3),
            "jitter" : transforms.ColorJitter(brightness=.5, hue=.3)
        }
        self.detectors = {}
        self.drift = {}
        self.mmd = {}
        self.pval = {}
        for run, drift, params in parameterlist:
            self.detectors[run] = ImageDrift(self.training_data, **params)
            self.mmd[run] = []
            self.pval[run] = []
            self.drift[run] = drift

        print("Loading Test Data...")
        self.test = torch.load(os.path.sep.join(["tensors", "cocotest.pt"]))
        print("Test Data Loaded.")
        self.perform_experiment()

    
    def random_samples(self, run, n, p):
        torch.manual_seed(33)

        perm = torch.randperm(self.test.shape[0])
        selected_samples = self.test[perm, ...][:n, ...]
        driftcutoff = math.ceil(n * p)
        print(driftcutoff)
        drift = selected_samples[:driftcutoff, ...]
        no_drift = selected_samples[driftcutoff:, ...]
        
        driftTransform = self.driftransforms[self.drift[run]]
        drift = driftTransform(drift)
        final = torch.cat([no_drift, drift])[torch.randperm(n), ...]
        
        for x in range(n):
            response = self.detectors[run].request(final[x, ...])
            if response["enough-data"]:
                self.mmd[run].append(response["mmd"])
                self.pval[run].append(response["p-value"])

    def basic_drift_sim(self, run):
        print("Sample 0%...")
        self.random_samples(run, 200, 0)
        print("Sample 25%...")
        self.random_samples(run, 200, .25)
        print("Sample 50%...")
        self.random_samples(run, 200, .5)
        print("Sample 75%...")
        self.random_samples(run, 200, .75)
        print("Sample 100%...")
        self.random_samples(run, 200, 1)

    def default_sim(self, run):
        print("Sample 0%...")
        self.random_samples(run, 1000, 0)

    def save_experiment(self):
        mmd = []
        pval = []
        print("Creating Datasets...")
        for run in self.detectors: 
            index = np.arange(1, len(self.mmd[run]) + 1) * self.detectors[run].window_size
            mmd.append(pd.DataFrame({run : self.mmd[run]}, index = index))
            pval.append(pd.DataFrame({run : self.pval[run]}, index = index))
        mmd = pd.concat(mmd, axis=1)
        pval = pd.concat(pval, axis=1)
        print("Created Datasets")
        # Save Data
        print("Saving Datasets...")
        mmd.to_csv(os.path.sep.join([self.save_folder, "mmd.csv"]))
        pval.to_csv(os.path.sep.join([self.save_folder, "pval.csv"])) 
        print("Saved Datasets.")
        print("Plotting Data...")
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(self.name)
        fig.set_figheight(7)
        fig.set_figwidth(7)
        # Plot data
        for title, run in mmd.T.iterrows():
            run = run.dropna().sort_index()
            ax1.plot(run, label=title, **{'marker': 'o'})
            ax1.set(xlabel="data points", ylabel="mmd")
            ax1.legend()

        for title, run in pval.T.iterrows():
            run = run.dropna().sort_index()
            ax2.plot(run, label=title, **{'marker': 'o'})
            ax2.set(xlabel="data points", ylabel="p value")
            ax2.legend()

        ax2.axhline(y=0.05, linestyle="dashed")
        plt.savefig(os.path.sep.join([self.save_folder, self.name + ".png"]))
        print("Data plotted.")

    def perform_experiment(self):
        print("Starting the experiment...")
        for run in self.detectors:
            self.basic_drift_sim(run)
        print("Experiment done.")
        self.save_experiment()

DriftExperiment("Windowsize_Resnetpca_GaussianBlur", [
    ("windowsize100", "gaussianBlur", {
        "dimred" : "resnetpca",
        "window_size" : 100,
        "dimred_size" : 100
        }),
    ("windowsize50", "gaussianBlur", {
        "dimred" : "resnetpca",
        "window_size" : 50,
        "dimred_size" : 100
        }),
    ("windowsize200", "gaussianBlur", {
        "dimred" : "resnetpca",
        "window_size" : 200,
        "dimred_size" : 100,
        })
])
