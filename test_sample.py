from models.vgg16bn_disp import DepthNet
from PIL import Image
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from models.vgg16bn_disp import DepthNet
import torch
from hyperparameters import *

NYUD_MEAN = [0.481215, 0.41197756, 0.39314577]
NYUD_STD = [0.28848645, 0.29521945, 0.3089535]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'models/' + MODEL_NAME
sample_img_name = "sample_image_2.jpg"


def rescale_img(img, output_shape=(IMG_HEIGHT_RESCALE, IMG_WIDTH_RESCALE)):
    # Rescaled images
    images_rescaled = np.zeros((1, 3, output_shape[0], output_shape[1]))

    # Border size
    border = BORDER_SIZE

        # Rescale images
    images_rescaled[0,:,:,:] = np.swapaxes(np.swapaxes(resize(
        np.swapaxes(np.swapaxes(img[0,:,:,:],0,1),1,2),
        output_shape = (output_shape[0]+2*border, output_shape[1]+2*border, 3),
        clip=True, 
        anti_aliasing=True,
        #multichannel=True,
        preserve_range=True
        ),2,1),0,1)[:,border:-border,border:-border]

    # Save images
    return images_rescaled


def centerCrop(imgs):

    y1 = int( (IMG_HEIGHT_RESCALE - IMG_HEIGHT) // 2 )
    y2 = int( IMG_HEIGHT_RESCALE - (IMG_HEIGHT_RESCALE - IMG_HEIGHT) // 2 )
    x1 = int( (IMG_WIDTH_RESCALE - IMG_WIDTH) // 2 )
    x2 = int( IMG_WIDTH_RESCALE - (IMG_WIDTH_RESCALE - IMG_WIDTH) // 2 )

    return imgs[:,:,y1:y2,x1:x2]


def denormalize(img):
    for i in range(3):
        img[:,i,:,:] = (img[:,i,:,:]*NYUD_STD[i] + NYUD_MEAN[i])*255

    # Hard limit
    img[img<0] = 0
    img[img>255] = 255

    return img


def visualize_sample(model, img, title, nsamples=1):

    fig, axes = plt.subplots(nrows=nsamples, ncols=2, dpi=120)
    ax = axes.ravel()

    img = torch.from_numpy(img).float()
    for r in range(nsamples):
        # To Cuda
        img = img.to(device).float()

        with torch.no_grad():
            # Prediction
            disp = model(img)
            depth = 1 / disp

            # Limit values
            depth = torch.clamp(depth, min=0.1, max=10)

        # Conversion to numpy
        image_numpy = torch.squeeze(denormalize(img)).swapaxes(0,1).swapaxes(1,2).to(torch.uint8).cpu().numpy()
        depth_numpy = torch.squeeze(depth).detach().cpu().numpy()

        # Visualize
        ax[r*2].imshow(depth_numpy)
        ax[r*2].set_axis_off()
        ax[r*2].set_title('Depth prediction')
        ax[r*2+1].imshow(image_numpy)
        ax[r*2+1].set_axis_off()
        ax[r*2+1].set_title('Original image')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def image_init(img_name="sample_image.jpg"):
    # Import image
    img = np.zeros((1, 3, IMG_HEIGHT_RESCALE, IMG_WIDTH_RESCALE))
    img = Image.open("nebitni_fajlovi/" + img_name)

    # Initialize transformations
    img = np.array(img)
    img = img[np.newaxis, ...]
    img = np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    # Rescale
    img = rescale_img(img=img)

    # Crop
    img = centerCrop(img)

    # Normalization
    img /= 255
    for i in range(3):
        img[:,i,:,:] = (img[:,i,:,:] - NYUD_MEAN[i]) / NYUD_STD[i]

    return img


if __name__ == '__main__':
    # Init image
    img = image_init(img_name=sample_img_name)

    # Load pretrained network
    print('Loading model...')
    model = DepthNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device).eval()
    print('Model loaded!')

    # Visualize Sample
    visualize_sample(model, img=img, title='Sample image')
