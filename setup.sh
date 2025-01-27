#!/usr/bin/env -S bash -i

# Create mamba environment
mamba create --name 04_image_translation python=3.10

# Install ipykernel in the environment.
mamba install -y ipykernel nbformat nbconvert black jupytext --name 04_image_translation
# Specifying the environment explicitly.
# mamba activate sometimes doesn't work from within shell scripts.

# install viscy and its dependenciex`s in the environment using pip.
mkdir -p ~/code/
cd ~/code/
git clone https://github.com/mehta-lab/viscy.git
cd viscy
git checkout dlmbl2023
# Find path to the environment - mamba activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep 04_image_translation | awk '{print $NF}')
$ENV_PATH/bin/pip install ."[metrics]"
# Store the code directory path.
CODE_DIR=$(pwd)


# Create data directory
mkdir -p ~/data/04_image_translation
cd ~/data/04_image_translation
wget https://dl-at-mbl-2023-data.s3.us-east-2.amazonaws.com/DLMBL2023_image_translation_data_pyramid.tar.gz
wget https://dl-at-mbl-2023-data.s3.us-east-2.amazonaws.com/DLMBL2023_image_translation_test.tar.gz 
tar -xzf DLMBL2023_image_translation_data_pyramid.tar.gz 
tar -xzf DLMBL2023_image_translation_test.tar.gz

# Go back to the code directory
cd $CODE_DIR


# this didn't not work from within shell scripts on TA1 node even after mamba init.
# mamba activate 04_image_translation 