import torch 


coco = torch.load("coco.pt")

cocotrain = coco[:1000, ...]

cocotest = coco[1000:, ...]

torch.save(cocotrain, "cocotrain.pt")
torch.save(cocotest, "cocotest.pt")