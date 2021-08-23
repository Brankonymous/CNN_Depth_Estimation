#from Dataloader import NYU_Depth_V2
from matplotlib import pyplot as plt
import numpy as np
import torch
from time import time
from loss.loss_functions import l1_loss, smooth_loss
from preprocessing.data_transformations import get_split, denormalize

'''
gas = NYU_Depth_V2()

print(gas.images.shape)
print(gas.images[0].shape)
ide = np.swapaxes(gas.images[0], 0, 2).astype('uint8')
ide2 = np.swapaxes(ide, 0, 1).astype('uint8')
plt.imshow(ide2,cmap="gray")
plt.show()
'''

train_data, val_data, test_data = get_split(train=True)

train_data.initBatch()
imgs, dpts = train_data.getBatch()

img = denormalize(imgs)
imgs = np.swapaxes(np.swapaxes(imgs.numpy(),1,2),2,3)
dpts = np.swapaxes(np.swapaxes(dpts.numpy(),1,2),2,3)

print('Train')

for i in range(5):

    im = imgs[i,:,:,:].astype('uint8')

    plt.figure()
    plt.imshow(im)
    plt.show()

    plt.figure()
    plt.imshow(dpts[i,:,:,0])
    plt.show()

print('Test data')
test_data.initBatch()
imgs, dpts = test_data.getBatch()

imgs = denormalize(imgs)
imgs = np.swapaxes(np.swapaxes(imgs.numpy(),1,2),2,3)
dpts = np.swapaxes(np.swapaxes(dpts.numpy(),1,2),2,3)


for i in range(5):

    im = imgs[i,:,:,:].astype('uint8')
  
    plt.figure()
    plt.imshow(im)
    plt.show()

    plt.figure()
    plt.imshow(dpts[i,:,:,0])
    plt.show()