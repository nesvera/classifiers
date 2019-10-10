#!/bin/bash
#
# script to extract ImageNet dataset
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar

imagenet_path=$1
output_path=$2
n_classes=$3

current_path="$(pwd)"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]
then
    echo "Error: Script requires 3 argument"
    echo "Usage: sh extract_ILSVRC2012.sh /path/to/imagenet /path/to/output num_of_classes"
    exit 1
fi

# check if directory exist
if [ ! -d "$imagenet_path" ]
then
    echo "Dataset directory does not exist"
    exit 1
fi

if [ ! -d "$output_path" ]
then
    echo "Output directory does not exist"
    exit 1
fi

# Create train and val folders in the output directory
cd $output_path
if [ -d "train" ] || [ -d "val" ]
then
    echo "Output directory already has train and valid folders"
    echo "Choose other directory to not overwrite the classes"
    #exit 1
else
    # Check if output directory already has train and val folder
    mkdir train
    mkdir val
fi
cd ..

imagenet_train="${imagenet_path}/train"
imagenet_val="${imagenet_path}/val"
output_train="${output_path}/train"
output_val="${output_path}/val"

# go to the train directory inside imagenet and get all classes
cd $imagenet_train



exit 1

# check if train dataset exist
if [ ! -f "ILSVRC2012_img_train.tar" ]
then
    echo "Training dataset: tar file was not found"
    echo "Skipping extraction of the train set"
    echo " "
else
    
    # check if the train was already extracted
    if [ -d "train" ]
    then
        echo "Train folder already exist"
        echo "Skipping extraction of the train set"
        echo " "
    else
        echo "Extracting train dataset ..."
        mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
        tar -xvf ILSVRC2012_img_train.tar
        mv ILSVRC2012_img_train.tar ../
        find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        echo " "
    fi
fi

# check if validation dataset exist
if [ ! -f "ILSVRC2012_img_val.tar" ]
then
    echo "Validation dataset: tar file was not found"
    echo "Skipping extraction of the train set"
    echo " "
else
    # check if the validation was already extracted
    if [ -d "val" ]
    then
        echo "Validation folder already exist"
        echo "Skipping extraction of the validation set"
        echo " "
    else
        echo "Extracting validation dataset ..."          
        mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
        wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
        mv ILSVRC2012_img_val.tar ../
        cd ..
        echo " "
    fi
fi

cd val      
mv ILSVRC2012_img_val.tar ../
cd ..

# Check total files after extract
echo "Training folder: checking if train folder has 1281167 files"
find train/ -name "*.JPEG" | wc -l

echo "Validation folder: checking if validation folder has 50000 files"
find val/ -name "*.JPEG" | wc -l