from skimage.io import imread
import numpy as np
from preprocessing.data_transformations import NYUD_MEAN, NYUD_STD
image_path = 'custom_images/ok.png'


if __name__ == '__main__':

    # Read image
    image_org = imread(image_path)
    # Image reshape
    image = np.swapaxes(np.swapaxes(image_org,2,1),0,1).reshape((1,3,image_org[1],image_org[2]))
    # Image normalization
    for i in range(3):
        image[:,0,:,:] = (image - NYUD_MEAN) / NYUD_STD
    # Image rescale


    