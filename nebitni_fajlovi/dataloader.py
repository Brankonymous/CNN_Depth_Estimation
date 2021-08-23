import numpy as np
import torch
from skimage.transform import warp, AffineTransform
import torch.utils.data as data
from torchvision.transforms import Lambda, Normalize, ToTensor
from .image_process import (EnhancedCompose, Merge, RandomCropNumpy, Split, to_tensor,
                          BilinearResize, CenterCropNumpy, RandomRotate, AddGaussianNoise,
                          RandomFlipHorizontal, RandomColor)

from sklearn.model_selection import train_test_split


def get_DataSets(path='./dataset/', split_ratio=0.1):

    # Input data
    images = np.load(path + "images.npy")
    # Target data
    depths = np.load(path + "depths.npy")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(images, depths, test_size=split_ratio, random_state=42)

    # Get train data loader
    train_loader = NYU_Depth_V2(dataset=(X_train, y_train), transform=NYU_Depth_V2.get_transform())
    val_loader = NYU_Depth_V2(dataset=(X_val, y_val), transform=NYU_Depth_V2.get_transform())

    return train_loader, val_loader


#imagenet pretrained normalization
NYUD_MEAN = [0.485, 0.456, 0.406]
NYUD_STD = [0.229, 0.224, 0.225]

class NYU_Depth_V2(data.Dataset):
    def __init__(self, dataset, transform=None):

        # Tranformations
        self.transform = transform

        # Storing data
        self.images = dataset[0]
        self.depths = dataset[1]


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image = self.images[index]
        depth = self.depths[index]

        if self.transform is not None:
            image, depth = transform_chw(self.transform, [image, depth])

        return image, depth


    def compute_image_mean(self):
        return np.mean(self.images / 255, axis=(0, 2, 3))


    def compute_image_std(self):
        return np.std(self.images / 255, axis=(0, 2, 3))
        
    @staticmethod
    def get_transform(size=(256, 352), normalize=True):

        transforms = [
            Merge(),
            RandomFlipHorizontal(),
            RandomRotate(angle_range=(-5, 5), mode='constant'),
            #crop size (257, 353) is same as the DORN one and this is not suitable for dispnet since it is not the int times of 32
            RandomCropNumpy(size=size),
            RandomAffineZoom(scale_range=(1.0, 1.5)),
            Split([0, 3], [3, 5]), #split merged data into rgb and depth
            # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
            [RandomColor(multiplier_range=(0.8, 1.2)), None],
        ]


        transforms.extend([
            # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not

            [ToTensor(), Lambda(to_tensor)], # this ToTensor did not maps from [0, 255] to [0, 1]
                Double_Float(),
            [Normalize(mean=NYUD_MEAN, std=NYUD_STD), None] if normalize else None
        ])

        return EnhancedCompose(transforms)

class RandomAffineZoom():
    def __init__(self, scale_range=(1.0, 1.5), random_state=np.random):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range
        self.random_state = random_state

    def __call__(self, image):
        scale = self.random_state.uniform(self.scale_range[0],
                                          self.scale_range[1])
        if isinstance(image, np.ndarray):
            af = AffineTransform(scale=(scale, scale))
            image = warp(image, af.inverse)
            rgb = image[:, :, 0:3]
            depth = image[:, :, 3:4] / scale
            mask = image[:, :, 4:5]
            return np.concatenate([rgb, depth, mask], axis=2)
        else:
            raise Exception('unsupported type')

class Double_Float():
    def __init__(self):
        pass
    def __call__(self, image):
        return [image[0].float(), image[1].float()]

def transform_chw(transform, lst):
    """Convert each array in lst from CHW to HWC"""
    return transform([x.transpose((1, 2, 0)) for x in lst])