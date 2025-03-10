import torchvision.datasets as d
import torchvision
import torchdrift
import torchvision.transforms as transforms
import torch 
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt
from functools import partial
# import torch.utils.data.DataLoader as Loader

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = np.squeeze(img.numpy())
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
     torch.flatten,
     torchdrift.reducers.PCAReducer() # partial(torchdrift.reducers.PCAReducer, n_components=5)])
    ])

loader = torch.utils.data.DataLoader(d.CIFAR10(root="./cifar", download=False, transform=transform))

collected = 0
l = []
for sample in loader:
    print(type(sample[0]))
    if sample[1] != 1:
        continue 
    if collected >= 100:
        break 
    collected += 1
    l.append(sample[0])

sample_tensor = torch.cat(l, axis=0)
print(sample_tensor.shape)
'''
b = False

print(b*1)
# imshow(torchvision.utils.make_grid(sample_tensor[:5, ...]))
# torch.save(sample_tensor[:100, ...], "training_sample.pt")
# torch.save(sample_tensor[100:, ...], "test_sample.pt")

# torch.save(sample_tensor, "drift_sample.pt")