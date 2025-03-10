import torch 
import matplotlib.pyplot as plt
import numpy as np 
import torchvision.transforms as transforms
from torchvision.utils import make_grid

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = np.squeeze(img.numpy())
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

coco = torch.load("cocotest.pt")

print(coco.shape)
torch.manual_seed(24)

coco = coco[torch.randperm(coco.size()[0])][:1, ...]

drift1 = transforms.GaussianBlur(kernel_size=7)(coco)
drift = transforms.ColorJitter(brightness=.5, hue=.3)(coco)

# torch.save(drift, "cocodriftjitter.pt")


imshow(make_grid([coco[0], drift1[0], drift[0]]))
