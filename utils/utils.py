import pickle
import numpy as np
import os
import matplotlib.pyplot as plt


def load_cfar10_classes(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path + '/batches.meta', mode='rb') as file:
        classes = pickle.load(file, encoding='latin1')
    return classes


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.close()


def plot_results(save_dir, name, train_results, val_results):
    plt.style.use('ggplot')
    plt.plot(train_results, label='training ' + name)
    plt.plot(val_results, label='validation ' + name)
    plt.legend()
    plt.savefig(os.path.join(save_dir, name + '.jpg'))
    plt.close()


def normalize_img(images, mean, std):

    means = np.vstack([np.expand_dims((np.ones((32, 32)) * mean[0]), 0),
                       np.expand_dims((np.ones((32, 32)) * mean[1]), 0),
                       np.expand_dims((np.ones((32, 32)) * mean[2]), 0)])

    stds = np.vstack([np.expand_dims((np.ones((32, 32)) * std[0]), 0),
                      np.expand_dims((np.ones((32, 32)) * std[1]), 0),
                      np.expand_dims((np.ones((32, 32)) * std[2]), 0)])

    images = (images - means) / stds

    return images.tolist()


