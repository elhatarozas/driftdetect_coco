import numpy as np 
from matplotlib import pyplot as plt
import torch

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = np.squeeze(img.numpy())
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


train = torch.load("cocotest.pt")

imshow(train[1])


