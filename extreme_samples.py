from models.vgg16bn_disp import DepthNet
import matplotlib.pyplot as plt
import numpy as np
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
model_path = 'models/' + MODEL_NAME
images_dir = 'images/' + model_path.split('/')[-1]
pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True)

# Loss weights
w1, w2 = W1, W2


def visualize_sample(img, gt_depth, depth, loss, title):

    # Initialize figure
    fig, axes = plt.subplots(figsize=(16,8), nrows=2, ncols=3, dpi=120)
    ax = axes.ravel()

    # Limit values
    depth = torch.clamp(depth, min=1e-3, max=10)

    # Conversion to numpy
    image_numpy = torch.squeeze(denormalize(img)).swapaxes(0,1).swapaxes(1,2).to(torch.uint8).cpu().numpy()
    gt_depth_numpy = torch.squeeze(gt_depth).detach().cpu().numpy()
    depth_numpy = torch.squeeze(depth).detach().cpu().numpy()

    # Visualize
    # Real range
    vmin = 0.1
    vmax = 7
    ax[0].imshow(gt_depth_numpy, vmin=vmin, vmax=vmax, cmap='gray')
    ax[0].set_axis_off()
    ax[0].set_title('Ground truth depth')
    ax[0].set_ylabel('Black = 0.1m / White = 10m')
    ax[1].imshow(depth_numpy, vmin=vmin, vmax=vmax, cmap='gray')
    ax[1].set_axis_off()
    ax[1].set_title('Prediction depth')
    ax[2].imshow(image_numpy)
    ax[2].set_axis_off()
    ax[2].set_title('Original image')
    # Normalized image
    ax[3].imshow(gt_depth_numpy)
    ax[3].set_axis_off()
    ax[3].set_title('Ground truth depth')
    ax[3].set_ylabel('Normalized range')
    ax[4].imshow(depth_numpy)
    ax[4].set_axis_off()
    ax[4].set_title('Prediction depth')
    ax[5].imshow(image_numpy)
    ax[5].set_axis_off()
    ax[5].set_title('Original image')

    fig.suptitle(title + ' (loss = {:.4f}'.format(loss) + ')')
    fig.savefig(images_dir + '/' + title + ' results.png', dpi=fig.dpi)
    plt.tight_layout()
    #plt.show()


def test(model, test_set, title):
    global w1, w2

    # Loss function dictionary
    loss_func = {'l1' : l1_loss, 'l2' : l2_loss, 'behru' : behru_loss}

    # Initialize running loss
    running_loss_photo = 0
    running_loss_smooth = 0
    running_loss = 0

    # Evaluation on test dataset
    N_test = test_set.initBatch(batch_size=1)

    # Iterate through test dataset
    best_img, best_dpt, best_loss = None, None, None
    worst_img, worst_dpt, worst_loss = None, None, None

    # Statistics of error
    l1_error = []
    rmse_error = []

    for itr in range(N_test):
        # Verbose
        # print('Iteration %d/%d' %(itr+1, N_test))

        # Get images and depths
        tgt_img, gt_depth = test_set.getSample(num=itr)

        # Move tensors to device
        tgt_img = tgt_img.to(device).float()
        gt_depth = gt_depth.to(device).float()

        with torch.no_grad():
            # Prediction
            disparities = model(tgt_img)
            depth = 1 / disparities
            
            # Calculate loss
            loss_1 = loss_func[LOSS](gt_depth, depth)
            loss_3 = smooth_loss(depth)
            loss = weighted_loss(loss_1, loss_3, w1, w2)
            
            # Update running loss
            running_loss_photo += loss_1.item() / N_test
            running_loss_smooth += loss_3.item() / N_test
            running_loss += loss.item() / N_test

            # L1 error & RMSE error
            l1_error.append(loss_1.item())
            rmse_error.append(RMSE(gt_depth, depth))

            if best_loss == None or best_loss > loss.item():
                best_img, best_gt_dpt, best_dpt, best_loss = tgt_img, gt_depth, depth, loss.item()
            if worst_loss == None or worst_loss < loss.item():
                worst_img, worst_gt_dpt, worst_dpt, worst_loss = tgt_img, gt_depth, depth, loss.item()

            torch.cuda.empty_cache()

    # Evaluate l1 error statistics
    error_l1 = np.array(l1_error)
    error_rmse = np.array(rmse_error)
    mean_l1_error = np.mean(error_l1)
    std_l1_error = np.std(error_l1)
    mean_rmse_error = np.mean(error_rmse)
    std_rmse_error = np.std(error_rmse)

    # Print results on training dataset
    print('------------------------------------------------')
    print('################ ' + title + ' results ##################')
    print('Photometric loss {:.4f}, Smooth loss {:.4f}, Overall loss {:.4f}'.format(running_loss_photo, running_loss_smooth, running_loss))
    print('Mean absolute error is {:.4f} with standard deviation of {:.4f}'.format(mean_l1_error, std_l1_error))
    print('Mean RMSE error is {:.4f} with standard deviation of {:.4f}'.format(mean_rmse_error, std_rmse_error))
    print('------------------------------------------------')
	
	# Visualize sample
    visualize_sample(best_img, best_gt_dpt, best_dpt, best_loss, 'Best sample in ' + title.lower() + ' dataset')
    visualize_sample(worst_img, worst_gt_dpt, worst_dpt, worst_loss, 'Worst sample in ' + title.lower() + ' dataset')


if __name__ == '__main__':

    # Load pretrained network
    print('Loading model...')
    model = DepthNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    print('Model loaded!')

    # Load dataset
    print('Loading data...')
    train_set, val_set, test_set = get_split(train=False)
    print('Data loaded!')
    
    # Choosing best sample
    print('Choosing best sample in train dataset...')
    test(model=model, test_set=train_set, title="Training")

    print('Choosing best sample in validation dataset...')
    test(model=model, test_set=val_set, title="Validation")

    print('Choosing best sample in test dataset...')
    test(model=model, test_set=test_set, title="Test")