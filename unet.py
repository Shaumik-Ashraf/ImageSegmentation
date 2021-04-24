# -*- coding: utf-8 -*-
# unet.py

#https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

import torch;
import torchvision;
import numpy as np;
import segmentation_models_pytorch as smp;
from kidney_dataset import KidneyDataset;
from matplotlib import pyplot as plt;

print("======================== start unet.py ================================\n");

print("loading model... ", end="");
model = smp.Unet("resnet18", in_channels=3, classes=1, activation='sigmoid');
print("done.");

# using trainset but not actually training anything
print("loading data... ", end="");
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]);
trainset = KidneyDataset("data/train", train=True, transform=transform);
print("done.");

print("loading image and mask...", end="");
image, meta, mask = trainset.__getitem__(0);
print("done.");

print("plotting input image... ", end="");
#plt.figure()
#plt.subplot(1, 3, 1)
plt.xticks([])
plt.yticks([])
plt.title("image")
plt.imshow(image.permute(1,2,0));
plt.show();
print("done.");

print("running model... ", end="");
image = image.unsqueeze(0);

pred = model(image);

image = image.squeeze(0);
pred = pred.squeeze(0);
print("done.");

print("plotting prediction...", end="");
#plt.subplot(1, 3, 2)
plt.xticks([]);
plt.yticks([]);
plt.title("pred");
plt.imshow(pred.permute(1,2,0));
plt.show();
print("done.");

print("plotting ground truth...", end="");
#plt.subplot(1, 3, 3)
plt.xticks([])
plt.yticks([])
plt.title("true")
plt.imshow(mask.permute(1,2,0));
plt.show();
print("done.");

dice_loss = smp.utils.losses.DiceLoss(mode='binary');
pred = pred.squeeze(0);
mask = mask.squeeze(0);
print("loss: ", dice_loss(pred, mask));
#pred = pred.unsqueeze(0);
#mask = mask.unsqueeze(0);

print("\n============================= end ====================================");

#print(meta);
