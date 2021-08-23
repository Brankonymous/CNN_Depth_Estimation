from models.vgg16bn_disp import DepthNet
import matplotlib.pyplot as plt
from numpy import float32
from models.vgg16bn_disp import DepthNet
from time import time
import torch
from loss.loss_functions import *
import pathlib
from preprocessing.data_transformations import denormalize, get_split
from hyperparameters import *

# Device recognition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is', device)

# Paths
from hyperparameters import *
model_path = 'models/' + MODEL_NAME
images_dir = 'images/' + model_path.split('/')[-1]
pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True) 

# Loss weights
w1, w2 = W1, W2


def visualize_sample(model, dataset, title, nsamples=3):

    fig, axes = plt.subplots(nrows=nsamples, ncols=3, dpi=120)
    ax = axes.ravel()

    for r in range(nsamples):
        img, gt_depth = dataset.getSample()

        # To Cuda
        img = img.to(device).float()
        gt_depth = gt_depth.to(device).float()

        with torch.no_grad():
            # Prediction
            disp = model(img)
            depth = 1 / disp

            # Limit values
            depth = torch.clamp(depth, min=0, max=10)

        # Conversion to numpy
        image_numpy = torch.squeeze(denormalize(img)).swapaxes(0,1).swapaxes(1,2).to(torch.uint8).cpu().numpy()
        gt_depth_numpy = torch.squeeze(gt_depth).detach().cpu().numpy()
        depth_numpy = torch.squeeze(depth).detach().cpu().numpy()

        # Visualize
        vmin = 0.1
        vmax = 7
        ax[r*3].imshow(gt_depth_numpy, vmin=vmin, vmax=vmax, cmap='gray')
        ax[r*3].set_axis_off()
        ax[r*3].set_title('Ground truth depth')
        ax[r*3+1].imshow(depth_numpy, vmin=vmin, vmax=vmax, cmap='gray')
        ax[r*3+1].set_axis_off()
        ax[r*3+1].set_title('Prediction depth')
        ax[r*3+2].imshow(image_numpy)
        ax[r*3+2].set_axis_off()
        ax[r*3+2].set_title('Original image')

    fig.suptitle(title)
    fig.savefig(images_dir + '/' + title + ' results.png', dpi=fig.dpi)
    plt.tight_layout()
    #plt.show()


def test(model, test_set):
    global w1, w2

    # Loss function dictionary
    loss_func = {'l1' : l1_loss, 'l2' : l2_loss, 'behru' : behru_loss}

    # Initialize running loss
    running_loss_photo = 0
    running_loss_smooth = 0
    running_loss = 0

    # Calculate forward mean time
    mean_time = 0

    # Evaluation on test dataset
    N_test = test_set.initBatch(batch_size=1)

    # Iterate through test dataset
    for itr in range(N_test):
        # Verbose
        print('Iteration %d/%d' %(itr+1, N_test))

        # Get images and depths
        tgt_img, gt_depth = test_set.getBatch()

        # Move tensors to device
        tgt_img = tgt_img.to(device).float()
        gt_depth = gt_depth.to(device).float()

        with torch.no_grad():
            # Prediction
            start = time()
            disparities = model(tgt_img)
            end = time()
            mean_time += (end-start) / BATCH_SIZE
            depth = 1 / disparities
            
            # Calculate loss
            loss_1 = loss_func[LOSS](gt_depth, depth)
            loss_3 = smooth_loss(depth)
            loss = weighted_loss(loss_1, loss_3, w1, w2)
            
            # Update running loss
            running_loss_photo += loss_1.item() / N_test
            running_loss_smooth += loss_3.item() / N_test
            running_loss += loss.item() / N_test

            torch.cuda.empty_cache()

    mean_time /= N_test

    # Print results on training dataset
    print('------------------------------------------------')
    print('################ Test results ##################')
    print('Photometric loss {:.4f}, Smooth loss {:.4f}, Overall loss {:.4f}'.format(running_loss_photo, running_loss_smooth, running_loss))
    print('Average inference time: {:.4f}'.format(mean_time))
    print('------------------------------------------------')

    
if __name__ == '__main__':

    # Load pretrained network
    print('Loading model...')
    model = DepthNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device).eval()
    print('Model loaded!')

    # Load dataset
    print('Loading data...')
    train_set, val_set, test_set = get_split(train=False)
    print('Data loaded!')

    # Test model
    print('Testing a model...')
    test(model=model, test_set=test_set)
    print('Testing finished!')

    # Visualization of results
    visualize_sample(model, train_set, 'Training dataset')
    visualize_sample(model, val_set, 'Validation dataset')
    visualize_sample(model, test_set, 'Test dataset')
