from hyperparameters import *
import numpy as np
from random import randint, random, choice, choices, uniform
from scipy import ndimage
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt

NYUD_MEAN = [0.481215, 0.41197756, 0.39314577]
NYUD_STD = [0.28848645, 0.29521945, 0.3089535]


def get_split(path='./dataset/', split_ratio=SPLIT_RATIO, train=False):
    # Input data
    images = np.load(path + "images.npy")
    # Target data
    depths = np.load(path + "depths.npy")

    # Split data on train and test
    X_train, X_test, y_train, y_test = train_test_split(images, depths, test_size=split_ratio, random_state=42)

    # Split train data on train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_ratio, random_state=42)

    return DataSet(X_train, y_train, train=train), DataSet(X_val, y_val), DataSet(X_test, y_test)

# 96x128, 192x256, 288x384
class DataSet():

    def __init__(self, images=None, depths=None, train=False):

        self.images = images/255 # 0-255 => 0-1
        self.depths = depths
        self.N, self.C, self.H, self.W = images.shape
        self.itr = 0
        self.batch_size = 1
        self.N_itr = 1
        self.train = train

        # Rescale images
        self.rescale()

        # Normalization
        for i in range(3):
            self.images[:,i,:,:] = (self.images[:,i,:,:] - NYUD_MEAN[i]) / NYUD_STD[i]


    def size(self):
        return self.N


    def randomCrop(self, imgs, dpts):

        images = np.zeros((BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH))
        depths = np.zeros((BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH))

        for i in range(BATCH_SIZE):
            # Top left corner of a crop
            x = randint(0, self.W - IMG_WIDTH)
            y = randint(0, self.H - IMG_HEIGHT)
            # Cropping original images
            images[i,:,:,:] = imgs[i,:,y:y+IMG_HEIGHT,x:x+IMG_WIDTH]
            depths[i,:,:,:] = dpts[i,:,y:y+IMG_HEIGHT,x:x+IMG_WIDTH]

        return images, depths


    def centerCrop(self, imgs, dpts):

        y1 = int( (IMG_HEIGHT_RESCALE - IMG_HEIGHT) // 2 )
        y2 = int( IMG_HEIGHT_RESCALE - (IMG_HEIGHT_RESCALE - IMG_HEIGHT) // 2 )
        x1 = int( (IMG_WIDTH_RESCALE - IMG_WIDTH) // 2 )
        x2 = int( IMG_WIDTH_RESCALE - (IMG_WIDTH_RESCALE - IMG_WIDTH) // 2 )

        return imgs[:,:,y1:y2,x1:x2], dpts[:,:,y1:y2,x1:x2]


    def rescale(self, output_shape=(IMG_HEIGHT_RESCALE,IMG_WIDTH_RESCALE)):

        # Rescaled images
        images_rescaled = np.zeros((self.N, 3, output_shape[0], output_shape[1]))
        depths_rescaled = np.zeros((self.N, 1, output_shape[0], output_shape[1]))

        # Border size
        border = BORDER_SIZE

        # Iterate through dataset
        for i in range(self.N):
            # Rescale images
            images_rescaled[i,:,:,:] = np.swapaxes(np.swapaxes(resize(
                np.swapaxes(np.swapaxes(self.images[i,:,:,:],0,1),1,2),
                output_shape = (output_shape[0]+2*border, output_shape[1]+2*border, 3),
                clip=True, 
                anti_aliasing=True,
                #multichannel=True,
                preserve_range=True
                ),2,1),0,1)[:,border:-border,border:-border]

            # Rescale depth
            depths_rescaled[i,0,:,:] = resize(
                self.depths[i,0,:,:],
                output_shape=(output_shape[0]+2*border, output_shape[1]+2*border),
                clip=True, 
                anti_aliasing=True,
                #multichannel=False,
                preserve_range=True
                )[border:-border,border:-border]

        # Save images
        self.images = images_rescaled
        self.depths = depths_rescaled 

        self.H = output_shape[0]
        self.W = output_shape[1]


    def initBatch(self, batch_size=BATCH_SIZE):
        # Shuffle indices
        shuffle = np.random.permutation(self.N)

        # Shuffle data
        self.images = self.images[shuffle]
        self.depths = self.depths[shuffle]

        # Reset iterator
        self.itr = 0

        # Batch size
        self.batch_size = batch_size

        # Number of possible iterations
        self.N_itr = int(self.N // self.batch_size)

        return self.N_itr


    def getBatch(self):

        # Extract batch
        imgs = self.images[(self.itr * self.batch_size) : ((self.itr + 1) * self.batch_size)]
        dpts = self.depths[(self.itr * self.batch_size) : ((self.itr + 1) * self.batch_size)]

        # If data is used for training then random crop is used insted of rescale method
        if self.train:
            imgs, dpts = self.randomCrop(imgs, dpts)
            imgs, dpts = RandomHorizontalFlip(imgs, dpts)
            imgs, dpts = RandomVerticalFlip(imgs, dpts)
            if ROTATE:
                imgs, dpts = RandomRotate(imgs, dpts)
            if MIXUP:
                imgs, dpts = MixUp(imgs, dpts, size=MIXUP_SIZE)
            if BLEND:
                imgs, dpts = Blend(imgs, dpts, size=BLEND_SIZE)
        else:
            imgs, dpts = self.centerCrop(imgs, dpts)

        # Increment iterator
        self.itr += 1

        '''
        !MOJ PREDLOG!
        plt.figure()
        plt.imshow(np.swapaxes(np.swapaxes(imgs[0],0,1),1,2))
        plt.show()
        plt.figure()
        plt.imshow(dpts[0])
        plt.show()
        '''

        # Tensor conversion
        tensor_images = torch.from_numpy(imgs).float()
        tensor_depths = torch.from_numpy(dpts).float()

        return tensor_images, tensor_depths


    def getSample(self, num=None):

        if num == None:
            num = randint(0, self.N-1)

        image, depth = self.images[num].reshape((1, self.C, self.H, self.W)), self.depths[num].reshape((1, 1, self.H, self.W))

        image, depth = self.centerCrop(image, depth)

        # Get sample
        sample_img = torch.from_numpy(image).float().view(1, self.C, IMG_HEIGHT, IMG_WIDTH)
        sample_dpt = torch.from_numpy(depth).float().view(1, 1, IMG_HEIGHT, IMG_WIDTH)

        return sample_img, sample_dpt


def RandomHorizontalFlip(images, depth, p=P_FLIP):

    # Iterate through batch
    for i in range(images.shape[0]):
        if random() < p:
            images[i] = np.flip(images[i], 1)
            depth[i] = np.flip(depth[i], 1)

    return images, depth


def RandomVerticalFlip(images, depth, p=P_FLIP):

    # Iterate through batch
    for i in range(images.shape[0]):
        if random() < p:
            images[i] = np.flip(images[i], 2)
            depth[i] = np.flip(depth[i], 2)

    return images, depth


def denormalize(image):

    # Denormalization
    for i in range(3):
        image[:,i,:,:] = (image[:,i,:,:]*NYUD_STD[i] + NYUD_MEAN[i])*255

    # Hard limit
    image[image<0] = 0
    image[image>255] = 255

    return image


def MixUp(images, depths, param=0.2, size=MIXUP_SIZE):
    # Number of mixed images
    ssize = int(BATCH_SIZE*size)

    # Index range
    ind_range = np.arange(start=0, stop=BATCH_SIZE, dtype='int')

    for i in range(ssize):
        # Chose two indexes
        inds = choices(ind_range, k=2)

        # Generate sample from beta distribution
        lmbd = np.random.beta(a=param, b=param, size=1)

        # MixUp augmentation
        img_tmp = images[inds[0]]*lmbd + images[inds[1]]*(1-lmbd)
        dpt_tmp = depths[inds[0]]*lmbd + depths[inds[1]]*(1-lmbd)

        # Randomly choose where to save image
        ind = choice(inds)

        # Save augmented image
        images[ind] = img_tmp
        depths[ind] = dpt_tmp

    return images, depths


def Blend(images, depths, size=BLEND_SIZE):
    # Number of mixed images
    ssize = int(BATCH_SIZE*size)

    # Index range
    ind_range = np.arange(start=0, stop=BATCH_SIZE, dtype='int')

    for i in range(ssize):
        # Chose two indexes
        inds = choices(ind_range, k=2)

        # Blend augmentation
        crop_width = np.random.randint(IMG_WIDTH/5, 4*IMG_WIDTH/5)
        img_tmp1 = np.concatenate((images[inds[0], :, :, :crop_width], images[inds[1], :, :, crop_width:]), axis=2)
        dpt_tmp1 = np.concatenate((depths[inds[0], :, :, :crop_width], depths[inds[1], :, :, crop_width:]), axis=2)
        img_tmp2 = np.concatenate((images[inds[0], :, :, crop_width:], images[inds[1], :, :, :crop_width]), axis=2)
        dpt_tmp2 = np.concatenate((depths[inds[0], :, :, crop_width:], depths[inds[1], :, :, :crop_width]), axis=2)

        # Save augmented image
        images[inds[0]] = img_tmp1
        depths[inds[0]] = dpt_tmp1
        images[inds[1]] = img_tmp2
        depths[inds[1]] = dpt_tmp2


    return images, depths


def RandomRotate(images, depths, angle=ROTATION_ANGLE, size=ROTATION_SIZE):

    # Index range
    ind_range = np.arange(start=0, stop=BATCH_SIZE, dtype='int')

    # Select images to be rotated
    inds = choices(ind_range, k=int(size*BATCH_SIZE))

    for ind in inds:
        images[ind] = ndimage.rotate(images[ind], angle=uniform(-angle, angle), reshape=False, axes=(2,1), mode='mirror')
        depths[ind] = ndimage.rotate(depths[ind], angle=uniform(-angle, angle), reshape=False, axes=(2,1), mode='mirror')

    return images, depths