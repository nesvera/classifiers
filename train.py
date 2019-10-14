import torch
import torch.nn as nn
from torchsummary import summary
from models import MobileNet, Darknet
import torchvision
import torchvision.transforms as T
from utils.utils import Average, SGDR

import argparse
import os
import numpy as np
import yaml
import cv2
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        dest='config_path',
                        required=True,
                        help='Configuration file with train hyperparameters')

    args = parser.parse_args()
    
    # ------------------------
    #    Load configuration 
    # ------------------------
    config_path = args.config_path
    if os.path.exists(config_path) == False:
        print('Error: config file does not exist')
        exit(1)

    # Load hyperparameters from the configuration file
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    config_model_name =     config['MODEL']['NAME']
    config_input_size =     config['MODEL']['IMAGE_SIZE']
    config_num_classes =    config['MODEL']['NUM_CLASSES']
    config_alpha =          config['MODEL']['ALPHA']
    config_optimizer =      config['TRAIN']['OPTIMIZER']['OPTIMIZER']
    config_lr =             config['TRAIN']['OPTIMIZER']['LEARNING_RATE']
    config_momentum =       config['TRAIN']['OPTIMIZER']['MOMENTUM']
    config_weight_decay =   config['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    config_lr_scheduler =   config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER']
    config_sgdr_min_lr =    config['TRAIN']['LR_SCHEDULER']['MIN_LR']
    config_sgdr_max_lr =    config['TRAIN']['LR_SCHEDULER']['MAX_LR']
    config_sgdr_lr_decay =  config['TRAIN']['LR_SCHEDULER']['LR_DECAY']
    config_sgdr_cycle =     config['TRAIN']['LR_SCHEDULER']['CYCLE']
    config_sgdr_cycle_mult= config['TRAIN']['LR_SCHEDULER']['CYCLE_MULT']
    config_workers =        config['TRAIN']['WORKERS']
    config_max_epochs =     config['TRAIN']['MAX_EPOCHS']
    config_train_batch =    config['TRAIN']['BATCH_SIZE']
    config_val_batch =      config['TEST']['BATCH_SIZE']
    config_print_freq =     config['TRAIN']['PRINT_FREQ']
    config_train_dataset =  config['DATASET']['TRAIN']
    config_val_dataset =    config['DATASET']['VALID']
    config_experiment_path =config['EXP_DIR']
    config_checkpoint =     config['RESUME_CHECKPOINT']

    checkpoint_path = config_experiment_path + "/" + config_model_name
    checkpoint_path += "/" + config_checkpoint

    if os.path.exists(checkpoint_path) == False:
        config_checkpoint = ''
        print("Warning: Checkpoint was not found!")
    else:
        print("Warning: Loading checkpoint!")

    # Set the seed for reproducibility
    if config['TRAIN']['REPRODUCIBILITY']['REPRODUCIBILITY'] == True:
        config_seed =       config['TRAIN']['REPRODUCIBILITY']['SEED']

        torch.manual_seed(config_seed)
        np.random.seed(config_seed)

        # when running on the CuDNN backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------
    #    Build/Load model
    # ------------------------
    # Initialize a new model
    if config_checkpoint == '':
    
        # Keep track of losses
        train_loss_log = {}
        val_loss_log = {}
        top_5_log = {}
        top_1_log = {}
        lr_log = {}

        # MobileNet V1 (224x224x3) with standard convolutional layers
        if config_model_name == 'mobilenet_standard_conv_224':
            model = MobileNet.MobileNetV1Conv224(alpha=config_alpha, 
                                                 num_classes=config_num_classes)

        # MobileNet V1 (224x224x3) with depthwise separable convolutions
        # Deptwise + Pointwise layers
        elif config_model_name == 'mobilenet_dw_224':
            model = MobileNet.MobileNetV1Dw224(alpha=config_alpha, 
                                               num_classes=config_num_classes)

        # Darknet 19 (yolo9000)
        elif config_model_name == 'darknet-19':
            model = Darknet.Darknet19(num_classes=config_num_classes)

        else:
            model = MobileNet.MobileNetV1Conv224()

            # Initialize the optimizer with different learning rates 
            # for weights and bias
            biases = list()
            not_biases = list()
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)

            optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                        lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Initialize optimizers
        if config_optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=config_lr,
                                        momentum=config_momentum)

        elif config_optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config_lr,
                                         weight_decay=config_weight_decay)            

        start_epoch = 1

    # or load checkpoint
    else:
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch =   checkpoint['epoch'] + 1
        model =         checkpoint['model']
        optimizer =     checkpoint['optimizer']

        # Keep track of losses
        train_loss_log =checkpoint['train_loss_log']
        val_loss_log =  checkpoint['val_loss_log']
        top_5_log =     checkpoint['top_5_log']
        top_1_log =     checkpoint['top_1_log']
        lr_log =        checkpoint['lr_log']

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # summarize the model
    print()
    print("----------------------------------------------------------------")
    print("--------------------- Model summary ----------------------------")
    print("----------------------------------------------------------------")
    summary(model, (config_input_size[2], config_input_size[0], config_input_size[1]))


    # TODO: converter para grayscale as images do dataset da pista, pra ver se 
    # aquele efeito louco desaparece

    transform = T.Compose([
        T.Resize(config_input_size[0]),          # resize the smaller edge of the image, keeping ratio
        T.CenterCrop(config_input_size[0]),      # crop the image to the correct size        
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataloaders
    # "list" of train data
    train_dataset = torchvision.datasets.ImageFolder(root=config_train_dataset,
                                                     transform=transform)

    test_dataset = torchvision.datasets.ImageFolder(root=config_val_dataset,
                                                    transform=transform)
    
    # "list" of batches
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config_train_batch,
                                               shuffle=True,
                                               num_workers=config_workers,
                                               pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=config_val_batch,
                                                    shuffle=True,
                                                    num_workers=config_workers,
                                                    pin_memory=True)

    # Keep track for improvement
    best_loss = 9000.
    epochs_since_last_improvement = 0

    # Keep track of learning rate
    lr_schedule = SGDR(min_lr=config_sgdr_min_lr,
                       max_lr=config_sgdr_max_lr,
                       lr_decay=config_sgdr_lr_decay,
                       epochs_per_cycle=config_sgdr_cycle,
                       mult_factor=config_sgdr_cycle_mult)

    #lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)

    # 1 epoch:
    #   - train over all images of the dataset
    #   - validate over all images of the validation set
    for epoch in range(start_epoch, config_max_epochs):

        # ------------------------
        #          Train
        # ------------------------        
        train_loss = train(model=model,
                           loader=train_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           print_freq=config_print_freq)

        train_loss_log[epoch] = train_loss
        
        # ------------------------
        #        Validation 
        # ------------------------
        val_loss = validation(model=model,
                              loader=validation_loader,
                              criterion=criterion,
                              optimizer=optimizer,
                              epoch=epoch,
                              print_freq=config_print_freq)

        val_loss_log[epoch] = val_loss

        # Check if the model improved
        is_best = 0
        epochs_since_last_improvement += 1

        if val_loss < best_loss:
            is_best = 1
            best_loss = val_loss
            epochs_since_last_improvement = 0
            print("Melhorouuuuuuuuuuuuuuu\n")

        print("Val loss: {0:.3f} - Best loss: {1:.3f} \n Epochs since last improvement: {2}\n"
              .format(val_loss, best_loss, epochs_since_last_improvement))

        # ------------------------
        #      Test accuracy 
        # ------------------------
        top_1_acc, top_5_acc = calc_accuracy(model=model,
                                             loader=validation_loader)

        top_5_log[epoch] = top_5_acc
        top_1_log[epoch] = top_1_acc
        lr_log[epoch] = config_lr

        print("Accuracy - Top-5: {0:.2f} - Top-1: {1:.2f}\n"
              .format(top_5_acc*100, top_1_acc*100))

        # ------------------------
        #        Save model 
        # ------------------------
        exp_folder = config_experiment_path + "/" + config_model_name

        # Create a folder for a new topology/experiment
        if os.path.isdir(exp_folder) == False:
            os.mkdir(exp_folder)

        state = {'model_name': config_model_name,
                 'epoch': epoch,
                 'loss': val_loss,
                 'train_loss_log': train_loss_log,
                 'val_loss_log': val_loss_log,
                 'top_5_log': top_5_log,
                 'top_1_log': top_1_log,
                 'lr_log': lr_log,
                 'model': model,
                 'optimizer': optimizer}

        exp_filename = exp_folder + '/' + config_model_name + '.pth.tar'
        torch.save(state, exp_filename)

        if is_best:
            exp_filename = exp_folder + '/BEST_' + config_model_name + '.pth.tar'
            torch.save(state, exp_filename)

        if epoch == (config_max_epochs-1):
            exp_filename = exp_folder + '/LAST_' + config_model_name + '.pth.tar'
            torch.save(state, exp_filename)

        # ------------------------
        #  Learning rate schedule 
        # ------------------------
        if config_lr_scheduler == True:
            config_lr = lr_schedule.update()

            for opt in optimizer.param_groups:
                opt['lr'] = config_lr

            print('LR Scheduler - Cycle: [{0}/{1}]'
                  .format(lr_schedule.epoch_since_restart, lr_schedule.epochs_per_cycle))
            print('LR: {0:.5f}\n'.format(config_lr))

        else:
            print("Sem lr scheduler\n")

def train(model, loader, criterion, optimizer, epoch, print_freq):
    
    model.train() # training mode, enables dropout

    epoch_fetch_time = Average()      # data loading
    partial_fetch_time = Average()    

    epoch_train_time = Average()      # forward prop. + backprop.
    partial_train_time = Average()      # forward prop. + backprop., reset for each batch

    epoch_loss = Average()        # loss average

    batch_start = time.time()

    # loop through dataset, batch by batch
    for i, (images, labels) in enumerate(loader):
        
        epoch_fetch_time.add_value(time.time()-batch_start)
        partial_fetch_time.add_value(time.time()-batch_start)

        images = images.to(device)          # (batch_size, 3, width, height)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward prop.
        prediction_prob = model(images)     # (batch_size, n_classes)

        # Calculate loss
        loss = criterion(prediction_prob, labels)

        # Backprop
        loss.backward()

        # Clip gradient? estudar

        # Update model
        optimizer.step()

        epoch_loss.add_value(loss.item())   

        epoch_train_time.add_value(time.time()-batch_start)       # measure train time
        partial_train_time.add_value(time.time()-batch_start)

        batch_start = time.time()

        # print statistics
        if i % print_freq == 0:
            print('Epoch: [{0}] - Batch: [{1}/{2}]'.format(epoch, i, len(loader)))
            print('Partial fetch time: {0:.4f} - Epoch fetch time: {1:.4f} (seconds/batch)'
                 .format(partial_fetch_time.get_average(), epoch_fetch_time.get_average()))
            print('Partial train time: {0:.4f} - Epoch train time: {1:.4f} (seconds/batch)'
                 .format(partial_train_time.get_average(), epoch_train_time.get_average()))  
            print('Loss: {0:.5f} - Current loss: {1:.5f}'.format(epoch_loss.get_average(), loss.item()))
            print()

            # Reset measurment for each
            partial_fetch_time = Average()
            partial_train_time = Average()


    # time measurments

    return epoch_loss.get_average()

def validation(model, loader, criterion, optimizer, epoch, print_freq):

    model.eval()        # evaluation mode, disables dropout

    partial_eval_time = Average()
    epoch_eval_time = Average()

    epoch_loss = Average()        # loss average

    batch_start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):    
            
            images = images.to(device)                      # (batch_size, 3, width, height)
            labels = labels.to(device)

            prediction_prob = model(images)                 # (batch_size, n_classes)

            eval_loss = criterion(prediction_prob, labels) 
            
            epoch_loss.add_value(eval_loss.item())

            partial_eval_time.add_value(time.time()-batch_start)
            epoch_eval_time.add_value(time.time()-batch_start)
            batch_start = time.time()

            # print statistics
            if i % print_freq == 0:
                print('Validation - Epoch: [{0}] - Batch: [{1}/{2}]'.format(epoch, i, len(loader)))
                print('Partial eval time: {0:.4f} - Epoch eval time: {1:.4f} (seconds/batch)'
                     .format(partial_eval_time.get_average(), epoch_eval_time.get_average()))  
                print('Loss: {0:.5f} - Current loss: {1:.5f}'.format(epoch_loss.get_average(), eval_loss.item()))
                print()

                partial_eval_time = Average()

    return epoch_loss.get_average()

def calc_accuracy(model, loader):
    model.eval()

    top_1_acc = 0
    top_5_acc = 0 
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
       
            images = images.to(device)              # (batch_size, 3, width, height)
            labels = labels.to(device)              # (batch_size, 1)

            prediction_prob = model(images)         # (batch_size, n_classes)

            # top-5 
            '''            
            - Create a new tensor, copy the label tensor through axis 0
            - Compare the new tensor, with the top-5 will result in a tensor with true/false
            - Sum this tensor to know the number of predictions with the true label in the best 5
            '''
            top_5_label = labels.repeat(5,1).permute(1,0)           # (batch_size, 5)
            top_5_value, top_5_ind = torch.topk(prediction_prob, 5) # (batch_size, 5)
            
            top_5_match = torch.eq(top_5_ind, top_5_label)          # (batch_size, 5) (True/false)
            top_5_sum = top_5_match.sum()                           # (1)
            top_5_acc += top_5_sum.item()

            # top-1
            top_1_value = top_5_value[:,0]                          # (batch_size)
            top_1_ind = top_5_ind[:,0]                              # (batch_size)

            top_1_sum = torch.eq(top_1_ind, labels).sum()           # (1)
            top_1_acc += top_1_sum.item()

    top_5_acc = float(top_5_acc)/len(loader.dataset)
    top_1_acc = float(top_1_acc)/len(loader.dataset)

    return top_1_acc, top_5_acc

if __name__ == '__main__':
    main()
