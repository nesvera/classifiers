import torch
from torchsummary import summary
from models import MobileNet, Darknet

import argparse
import os
import numpy as np
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        dest='config_path',
                        required=True,
                        help='configuration file with train hyperparamenters')

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

    config_max_epochs =     config['TRAIN']['MAX_EPOCHS']
    config_batch_size =     config['TRAIN']['BATCH_SIZE']

    config_dataset_path =   config['DATASET']['DATASET_DIR']
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

    else:
        pass

    # summarize the model
    print()
    print("----------------------------------------------------------------")
    print("--------------------- Model summary ----------------------------")
    print("----------------------------------------------------------------")
    summary(model, (config_input_size[2], config_input_size[0], config_input_size[1]))


    # TODO: converter para grayscale as images do dataset da pista, pra ver se 
    # aquele efeito louco desaparece

    # TODO: load dataset
    #train_dataset = 
    #test_dataset = 
    #train_loader = 
    #test_loader = 

    # TODO: dataloaders
    '''
    

    model = model.MobilenetV1Conv224(alpha=1.0)
    model = model.to(device)

    # Repeat N epochs
    for epoch in range(epoch, config_max_epochs):
    '''

    # TODO: train

    # TODO: test

    # TODO: track of improvement

    # TODO: save_checkpoint
