import torch
import torch.nn as nn
from torchsummary import summary
from models import MobileNet, Darknet
import torchvision
import torchvision.transforms as T
from utils.utils import Average

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
                        help='Configuration file with train hyperparamenters')

    args = parser.parse_args()

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

    config_workers =        config['TRAIN']['WORKERS']
    config_max_epochs =     config['TRAIN']['MAX_EPOCHS']
    config_batch_size =     config['TRAIN']['BATCH_SIZE']
    config_print_freq =     config['TRAIN']['PRINT_FREQ']

    config_train_dataset =  config['DATASET']['TRAIN']
    config_val_dataset =    config['DATASET']['VALID']

    config_experiment_path =config['EXP_DIR']

    # Set the seed for reproducibility
    if config['TRAIN']['REPRODUCIBILITY']['REPRODUCIBILITY'] == True:
        config_seed =       config['TRAIN']['REPRODUCIBILITY']['SEED']

        torch.manual_seed(config_seed)
        np.random.seed(config_seed)

        # when running on the CuDNN backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize a new model or load checkpoint
    if config['RESUME_CHECKPOINT'] == '':

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

        criterion = nn.CrossEntropyLoss()
    else:

        start_epoch = 1 + 1
        pass

    model = model.to(device)

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
                                               batch_size=config_batch_size,
                                               shuffle=True,
                                               num_workers=config_workers,
                                               pin_memory=False)

    validation_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=config_batch_size,
                                                    shuffle=False,
                                                    num_workers=config_workers,
                                                    pin_memory=True)

    # 1 epoch:
    #   - train over all images of the dataset
    #   - validate over all images of the validation set
    for epoch in range(start_epoch, config_max_epochs):

        train(model=model,
              loader=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              print_freq=config_print_freq)

        val_loss = validation(model=model,
                              loader=validation_loader,
                              criterion=criterion,
                              optimizer=optimizer,
                              epoch=epoch,
                              print_freq=config_print_freq)

        save(config_model_name, config_experiment_path, epoch, model, optimizer, val_loss, 0)

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

    for i, (images, labels) in enumerate(loader):    
        
        images = images.to(device)          # (batch_size, 3, width, height)
        labels = labels.to(device)

        prediction_prob = model(images)     # (batch_size, n_classes)

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

            partial_eval_time = Average()

    return epoch_loss.get_average()

def save(model_name, path, epoch, model, optimizer, loss, is_best):

    filename = path + "/" + model_name

    if os.path.isdir(filename) == False:
        os.mkdir(filename)
        print("caiu")

    state = {'epoch': epoch,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}

    filename += '/' + model_name + '_' + str(epoch) + '.pth.tar'
    torch.save(state, filename)

    if is_best:
        pass

if __name__ == '__main__':
    main()
