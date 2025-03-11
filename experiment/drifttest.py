import torchdrift
from matplotlib import pyplot as plt
from torch.linalg import svd
import torch
import torchvision.models as models
from enum import Enum
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models 
import time
import torchvision 
from torch.utils.data import DataLoader 
import pandas as pd 

# torch.save(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), "resnet50.pt")

class ImageDrift:

    def __init__(self, training_data, *, window_size=100, dimred_size=10, preprocessing="default", dimred="none", p_value_threshhold=.05):
        self.dimred : str = dimred
        self.dimred_size=dimred_size
        self.window_size = window_size
        self.p_value_threshhold : float = p_value_threshhold
        self.preprocess = self.init_preprocess_training(preprocessing)
        self.init_training_data(training_data)
        self.init_transform()
        self.testdata = torch.empty([0] + list(self.tensors.shape[1:]))
        

    def init_preprocess_training(self, preprocessing):
        return transforms.Compose([# transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def init_training_data(self, training_data):
        tensors = self.preprocess(training_data)
        if "resnet" in self.dimred:
            self.resnet = models.resnet50()
            self.resnet.load_state_dict(torch.load("resnet50.pt"))
            self.resnet.eval()
            self.tensors = self.resnet(tensors)
        else:
            self.tensors = torch.flatten(tensors, start_dim=1)
        if "pca" in self.dimred:
            self.pca = torchdrift.reducers.PCAReducer(self.dimred_size)
            self.pca.fit(self.tensors)
            self.tensors = self.pca(self.tensors)
        print(self.tensors.shape)
        self.drift_detect = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=False)
        self.drift_detect.fit(self.tensors)


    def init_transform(self):
        # Expecting data in Tensorform
        operators = []
        if "resnet" in self.dimred:
            operators.append(self.resnet)
        else:
            operators.append(nn.Flatten())
        
        if "pca" in self.dimred:
            operators.append(self.pca)
        # operators.append(self.drift_detect)
        self.transform= nn.Sequential(*operators)

    def add_data(self, data):
        data = self.transform(torch.unsqueeze(self.preprocess(data), 0))
        self.testdata = torch.cat([self.testdata, data])

    def predict(self):
        result = {}
        if self.testdata.shape[0] < self.window_size:
            result["drift-detected"] = False
            result["enough-data"] = False
            return result
        result["enough-data"] = True
        result["mmd"] = self.drift_detect(self.testdata).item()
        result["p-value"] = self.drift_detect.compute_p_value(self.testdata).item()
        result["drift-detected"] = result["p-value"]  < self.p_value_threshhold
        self.testdata = torch.empty([0] + list(self.tensors.shape[1:]))
        return result


    def iso_map(self):
        import sklearn.manifold
        mapping = sklearn.manifold.Isomap()
        ref = mapping.fit_transform(self.tensors.numpy())

        test = mapping.transform(self.testdata.numpy())
        plt.scatter(ref[:, 0], ref[:, 1])
        plt.scatter(test[:, 0], test[:, 1])
        
    def request(self, data):
        self.add_data(data)
        return self.predict()
        
torch.set_grad_enabled(False)
