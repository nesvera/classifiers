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
    config_split =          config['DATASET']['SPLIT']

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
                                        momentum=config_momentum,
                                        weight_decay=config_weight_decay)

        elif config_optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config_lr,
                                         weight_decay=config_weight_decay)            


        start_epoch = 1

        criterion = nn.CrossEntropyLoss()
    else:

        start_epoch = 1 + 1
        pass

    # summarize the model
    print()
    print("----------------------------------------------------------------")
    print("--------------------- Model summary ----------------------------")
    print("----------------------------------------------------------------")
    summary(model, (config_input_size[2], config_input_size[0], config_input_size[1]))


    # TODO: converter para grayscale as images do dataset da pista, pra ver se 
    # aquele efeito louco desaparece

    val_transform = T.Compose([
        T.Resize(224),          # resize the smaller edge of the image, keeping ratio
        T.CenterCrop(224),      # crop the image to the correct size
        T.ToTensor()
    ])

    # Dataloaders
    # "list" of train data
    train_dataset = torchvision.datasets.ImageFolder(root=config_train_dataset,
                                                     transform=val_transform)

    test_dataset = None
    
    # "list" of batches
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               shuffle=True,
                                               num_workers=config_workers,
                                               pin_memory=True)

    test_loader = None

    for epoch in range(start_epoch, config_max_epochs):

        train(model=model,
              loader=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              print_freq=config_print_freq)

        val_loss = validation(model=model,
                              loader=None,
                              criterion=criterion,
                              optimizer=optimizer,
                              epoch=epoch,
                              print_freq=config_print_freq)


def train(model, loader, criterion, optimizer, epoch, print_freq):
    
    model.train() # training mode, enables dropout

    fetch_time = Average()      # data loading
    train_time = Average()      # forward prop. + backprop.

    loss_avg = Average()        # loss average

    batch_start = time.time()

    # loop through dataset, batch by batch
    for i, (images, labels) in enumerate(loader):
        
        fetch_time.add_value(time.time()-batch_start)

        images = images.to(device)          # (batch_size, 3, width, height)
        labels = labels.to(device)

        # Forward prop.
        prediction_prob = model(images)     # (batch_size, n_classes)

        # Calculate loss
        loss = criterion(prediction_prob, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip gradient? estudar
        print(loss.item())

        # Update model
        optimizer.step()

        train_time.add_value(time.time()-batch_start)       # measure train time
        batch_start = time.time()

        # print statistics
        if i % print_freq == 0:
            pass

        input()

    # time measurments

def validation(model, loader, criterion, optimizer, epoch, print_freq):

    model.eval()        # evaluation mode, disables dropout

    val_time = Average()

    batch_start = time.time()

    for i, (images, labels) in enumerate(loader):
        
        images = images.to(device)          # (batch_size, 3, width, height)
        labels = labels.to(device)

        prediction_prob = model(images)     # (batch_size, n_classes)

        eval_loss = criterion(prediction_prob, labels)

        val_time.add_value(time.time()-batch_start)
        batch_start = time.time()

        if i % print_freq == 0:
            pass

if __name__ == '__main__':
    main()