import argparse
import os
import random
from distutils.dir_util import copy_tree

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        dest='imagenet_path',
                        help='imagenet dataset path',
                        required=True)
    parser.add_argument('-o',
                        dest='output_path',
                        help='output folder',
                        required=True)
    parser.add_argument('-c',
                        dest='num_classes',
                        help='number of classes',
                        required=True)

    args = parser.parse_args()

    imagenet_path = args.imagenet_path
    imagenet_train_path = imagenet_path + "/train"
    imagenet_val_path = imagenet_path + "/val"

    # check if the imagenet folder exist
    # imagenet
    # | train
    # | val
    if(os.path.isdir(imagenet_path) and
       os.path.isdir(imagenet_train_path) and
       os.path.isdir(imagenet_val_path)) == False:
        print("Error: imagenet folder is not correct")
        exit(1)
    
    output_path = args.output_path
    output_train_path = output_path + "/train"
    output_val_path = output_path + "/val"
    
    # check if output folder exist
    if os.path.isdir(output_path) == True:
        print("Error: Output folder already exist. ")
        print("       Use other name for the dataset to not overwrite")
        exit(1)

    num_classes = int(args.num_classes)
    if num_classes <= 0:
        print("Error: Number of classes must be positive")
        exit(1)
   
    os.mkdir(output_path)
    os.mkdir(output_train_path)
    os.mkdir(output_val_path)

    # get imagenet classes
    imagenet_classes = os.listdir(imagenet_train_path)

    random.shuffle(imagenet_classes)

    output_classes = imagenet_classes[:num_classes]

    # copy classes to the new dataset
    for i, c in enumerate(output_classes):
        print("Copying folder: [{0}/{1}]".format(i+1, len(output_classes)))        
        imagenet_train_class_path = imagenet_train_path + "/" + c
        imagenet_val_class_path = imagenet_val_path + "/" + c

        output_train_class_path = output_train_path + "/" + c
        output_val_class_path = output_val_path + "/" + c

        copy_tree(imagenet_train_class_path, output_train_class_path)  
        copy_tree(imagenet_val_class_path, output_val_class_path)     
    
