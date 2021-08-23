import matplotlib.pyplot as plt
from models.vgg16bn_disp import DepthNet
from numpy import float32
from loss.loss_functions import *
import time
import torch
import pathlib
from preprocessing.data_transformations import get_split

# Device setup/recognition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is', device)

# Loading hyperparameters
from hyperparameters import *
w1, w2 = W1, W2 # Loss weights
lr = LR
batch_size = BATCH_SIZE
gamma, step = GAMMA, STEP
use_scheduler = USE_SCHEDULER

# Initialize loss list
training_loss = []
validation_loss = []

# Paths
model_path = 'models/' + MODEL_NAME
images_dir = 'images/' + model_path.split('/')[-1]
pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True) 


# Summary writer function
def summary_writter():
    summary_file = open(images_dir + '/summary.txt', "w")
    summary = "Model name : " + str(MODEL_NAME) + \
            "\nNumber of epochs : " + str(EPOCHS) + \
            "\nlearning rate : " + str(lr) + \
            "\nbatch size : " + str(batch_size) + \
            "\nWeighted loss : " + str(w1) + ' ' + str(w2) + \
            "\nLRScheduler : " + str(USE_SCHEDULER) + \
            "\nLoss : " + LOSS + \
            "\nLRScheduler (gamma, step) : " + str(gamma) + ' ' + str(step) + \
            "\nL1 smooth : " + str(SMOOTH_L1) + \
            "\nRescaled image : " + str(IMG_HEIGHT_RESCALE) + ', ' + str(IMG_WIDTH_RESCALE) + \
            "\nCropped image : " + str(IMG_HEIGHT) + ', ' + str(IMG_WIDTH) + \
            "\nBorder size : " + str(BORDER_SIZE) + \
            "\nMixUp : " + str(MIXUP) + \
            "\nMixUp ratio : " + str(MIXUP_SIZE) + \
            "\nBlend : " + str(BLEND) + \
            "\nBlend ratio : " + str(BLEND_SIZE) + \
            "\nRotation : " + str(ROTATE) + \
            "\nRotation angle : " + str(ROTATION_ANGLE) + \
            "\nRotation ratio : " + str(ROTATION_SIZE) + \
            "\n"
    n = summary_file.write(summary)
    summary_file.close()


# Train function
def train(batch_size, epochs):
    global training_loss, validation_loss, model_path, lr, w1, w2, gamma, step

    # Loss function dictionary
    loss_func = {'l1' : l1_loss, 'l2' : l2_loss, 'behru' : behru_loss}

    print('Loading data...')

    # Loading dataset
    train_set, val_set, test_set = get_split(train=True)

    print('Model setup...')

    # Creating a model
    model = DepthNet().to(device)

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma, verbose = True)

    print('Training...')

    best_loss = None

    training_loss = []
    validation_loss = []

    N_train = int(train_set.size() / batch_size)
    N_val = int(val_set.size() / batch_size)

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        print('Epoch %d/%d' %(epoch+1,epochs))
        
        # Prepare model for training
        model.train()

        # Initialize running loss
        running_loss_photo = 0
        running_loss_smooth = 0
        running_loss = 0

        # Number of training iterations
        N_train = train_set.initBatch(batch_size=batch_size)

        for itr in range(N_train):
            
            # Get images and depths
            tgt_img, gt_depth = train_set.getBatch()

            # Move tensors to device
            tgt_img = tgt_img.to(device).float()
            gt_depth = gt_depth.to(device).float()

            # Clear gradients
            optimizer.zero_grad()

            # Prediction
            disparities = model(tgt_img)
            depth = 1 / disparities

            # Calculate loss
            loss_1 = loss_func[LOSS](gt_depth, depth)
            loss_3 = smooth_loss(depth)
            loss = weighted_loss(loss_1, loss_3, w1, w2)

            # Calculate gradients
            loss.backward()

            # Update weights
            optimizer.step()

            print('Iteration {}/{}, loss = {:.4f}'.format(itr+1, N_train, loss.item()))

            # Update running loss
            running_loss_photo += loss_1.item() / N_train
            running_loss_smooth += loss_3.item() / N_train
            running_loss += loss.item() / N_train

            torch.cuda.empty_cache()

        # Print results on training dataset
        print('------------------------------------------------')
        print('########### Training results {}/{} #############'.format(epoch+1, epochs))
        print('Photometric loss {:.4f}, Smooth loss {:.4f}, Overall loss {:.4f}'.format(running_loss_photo, running_loss_smooth, running_loss))
        print('------------------------------------------------')

        # Save training loss for current epoch
        training_loss.append(running_loss)

        # Initialize running loss
        running_loss_photo = 0
        running_loss_smooth = 0
        running_loss = 0

        # Prepare model for validation
        model.eval()

        # Number of validation iterations
        N_val = val_set.initBatch(batch_size=batch_size)

        for itr in range(N_val):

            # Get images and depths
            tgt_img, gt_depth = val_set.getBatch()

            # Move tensors to device
            tgt_img = tgt_img.to(device).float()
            gt_depth = gt_depth.to(device).float()
            #gt_depth = torch.squeeze(gt_depth[:, 0, :, :])

            with torch.no_grad():
                # Prediction
                disparities = model(tgt_img)
                depth = 1 / disparities
                
                # Calculate loss
                loss_1 = loss_func[LOSS](gt_depth, depth)
                loss_3 = smooth_loss(depth)
                loss = weighted_loss(loss_1, loss_3, w1, w2)

                print('Iteration {}/{}, loss = {:.4f}'.format(itr+1, N_val, loss.item()))
                
                # Update running loss
                running_loss_photo += loss_1.item() / N_val
                running_loss_smooth += loss_3.item() / N_val
                running_loss += loss.item() / N_val

                torch.cuda.empty_cache()

        # Print results on validation dataset
        print('------------------------------------------------')
        print('########### Validation results {}/{} ###########'.format(epoch+1, epochs))
        print('Photometric loss {:.4f}, Smooth loss {:.4f}, Overall loss {:.4f}'.format(running_loss_photo, running_loss_smooth, running_loss))
        print('------------------------------------------------')

        # Save validation loss for current epoch
        validation_loss.append(running_loss)

        # Saving the best model
        if (best_loss == None) or (validation_loss[-1] < best_loss):
            # Get best loss
            best_loss = validation_loss[-1]
            # Write in log file
            log_file = open(images_dir + "/best_validation loss.txt","w")
            log_file.write("loss : " + str(round(best_loss,4)) + "\nepoch " + str(epoch+1))
            log_file.close()
            # Save best model
            torch.save(model.state_dict(), model_path)

        # Save loss plot
        if epoch%5 == 0:
            fig = plt.figure(figsize=(16,10), dpi=120)
            plt.plot(training_loss)
            plt.plot(validation_loss) 
            plt.legend(['Training loss','Validation loss'])
            plt.title('Loss function')
            plt.grid(b=True, which='minor')
            fig.savefig(images_dir + '/learning_curve.png', dpi=fig.dpi)

        
        scheduler.step()

    return (training_loss, validation_loss)


# Main
if __name__ == '__main__':

    summary_writter()

    start = time.time()

    try:

        loss = train(batch_size=batch_size, epochs=EPOCHS)

        fig = plt.figure(figsize=(16,10), dpi=120)
        plt.plot(loss[0])
        plt.plot(loss[1]) 
        plt.legend(['Training loss','Validation loss'])
        plt.title('Loss function')
        plt.grid(b=True, which='minor')
        fig.savefig(images_dir + '/learning_curve.png', dpi=fig.dpi)
        # plt.show()
        
    
    except KeyboardInterrupt:

        fig = plt.figure(figsize=(16,10), dpi=120)
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.legend(['Training loss','Validation loss'])
        plt.title('Loss function')
        plt.grid(b=True, which='minor')
        fig.savefig(images_dir + '/learning_curve.png', dpi=fig.dpi)
        # plt.show()

    end = time.time()
    print('Training time {}s'.format(end-start))