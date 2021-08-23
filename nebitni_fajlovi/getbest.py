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
model_path = 'models/' + MODEL_NAME
images_dir = 'images/' + model_path.split('/')[-1]
pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True)

# Loss weights
w1, w2 = W1, W2


def visualize_sample(model, img, gt_depth, loss, title, nsamples=3):

    fig, axes = plt.subplots(figsize=(16,8), nrows=nsamples, ncols=3, dpi=120)
    ax = axes.ravel()

    for r in range(nsamples):

        # To Cuda
        img = img.to(device).float()
        gt_depth = gt_depth.to(device).float()

        with torch.no_grad():
            # Prediction
            disp = model(img)
            depth = 1 / disp

        # Conversion to numpy
        image_numpy = torch.squeeze(denormalize(img)).swapaxes(0,1).swapaxes(1,2).to(torch.uint8).cpu().numpy()
        gt_depth_numpy = torch.squeeze(gt_depth).detach().cpu().numpy()
        depth_numpy = torch.squeeze(depth).detach().cpu().numpy()

        # Visualize
        ax[r*3].imshow(gt_depth_numpy)
        ax[r*3].set_axis_off()
        ax[r*3].set_title('Ground truth depth')
        ax[r*3+1].imshow(depth_numpy)
        ax[r*3+1].set_axis_off()
        ax[r*3+1].set_title('Prediction depth')
        ax[r*3+2].imshow(image_numpy)
        ax[r*3+2].set_axis_off()
        ax[r*3+2].set_title('Original image')

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

            if best_loss == None or best_loss > loss.item():
                best_img, best_dpt, best_loss = tgt_img, gt_depth, loss.item()
            if worst_loss == None or worst_loss < loss.item():
                worst_img, worst_dpt, worst_loss = tgt_img, gt_depth, loss.item()

            torch.cuda.empty_cache()


    # Print results on training dataset
    print('------------------------------------------------')
    print('################ ' + title + ' results ##################')
    print('Photometric loss {:.4f}, Smooth loss {:.4f}, Overall loss {:.4f}'.format(running_loss_photo, running_loss_smooth, running_loss))
    print('------------------------------------------------')

    return worst_img, worst_dpt, worst_loss, best_img, best_dpt, best_loss

    
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
    worst_img, worst_dpt, worst_loss, best_img, best_dpt, best_loss = test(model=model, test_set=train_set, title="Training")
    visualize_sample(model, best_img, best_dpt, best_loss, 'Best training dataset', nsamples=1)
    visualize_sample(model, worst_img, worst_dpt, worst_loss, 'Worst training dataset', nsamples=1)

    print('Choosing best sample in validation dataset...')
    worst_img, worst_dpt, worst_loss, best_img, best_dpt, best_loss = test(model=model, test_set=val_set, title="Validation")
    visualize_sample(model, best_img, best_dpt, best_loss, 'Best validation dataset', nsamples=1)
    visualize_sample(model, worst_img, worst_dpt, worst_loss, 'Worst validation dataset', nsamples=1)

    print('Choosing best sample in test dataset...')
    worst_img, worst_dpt, worst_loss, best_img, best_dpt, best_loss = test(model=model, test_set=test_set, title="Test")
    visualize_sample(model, best_img, best_dpt, best_loss, 'Best test dataset', nsamples=1)
    visualize_sample(model, worst_img, worst_dpt, worst_loss, 'Worst test dataset', nsamples=1)
