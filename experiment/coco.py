import torchvision.datasets as d
import torchvision
import torchdrift
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch 
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt
from functools import partial

torch.manual_seed(0)

tf = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=(224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
coco = torchvision.datasets.CocoDetection(root="./val2017/val2017", annFile="./annotations_trainval2017/annotations/instances_val2017.json", transform=tf)

l = DataLoader(coco, batch_size=5000, shuffle=True, collate_fn=lambda x: x )

data = next(iter(l))
data = torch.cat([torch.unsqueeze(x[0], 0) for x in data], axis=0)

torch.save(data, "coco.pt")
print(data.shape)