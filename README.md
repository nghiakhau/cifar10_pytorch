### CIFAR-10 - Object Recognition in Images

CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. [It](http://www.cs.toronto.edu/~kriz/cifar.html) was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

This repo contains the code (PyTorch) for training a simple CNNs model and has following directory structure:
```
├───data
│   └───cifar-10   # Download cifar 10 dataset then put here 
│
├───data_loader
│   └───data_loader.py
│
│
├───model
│   ├───batch.py
│   └───model.py
│
│
├───saved
│   ├───1st_train      
│       ├───config
│       |───log
│       └───model
│   └───2nd_train
│       ├───config
│       |───log
│       └───model
│
│
├───utils
│   └───utils.py
│
├───config.json   # training configurations
├───train.py
├───test.py
│
│
├───requirements.txt
├───setup.py
└───settings.py
```

## Setup
* Create virtual enviroment
* pip install -r requirements.txt

## Train
```
    python train.py --config config.json
```

## Test
```
    python test.py --save_path your_saved_dir_in_config.json
```

## TensorBoard
```
    tensorboard --logdir your_saved_dir_in_config.json
```
