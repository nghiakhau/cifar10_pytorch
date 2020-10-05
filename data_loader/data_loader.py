import pickle
import numpy as np
from torch.utils.data import Dataset
from utils.utils import normalize_img


def load_cfar10_with_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32))
    labels = batch['labels']

    return features, labels


def load_cfar10_train(cifar10_dataset_folder_path, batches):
    tr_x = None
    tr_y = []

    for i in batches:
        features, labels = load_cfar10_with_batch(cifar10_dataset_folder_path, i)
        if tr_x is None:
            tr_x = features
        else:
            tr_x = np.concatenate([tr_x, features])
        tr_y = tr_y + labels

    return tr_x, tr_y


def load_cfar10_test(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        test = pickle.load(file, encoding='latin1')
        te_x = test['data'].reshape((len(test['data']), 3, 32, 32))
        te_y = test['labels']

    return te_x, te_y


class Cifar10DataSet(Dataset):
    def __init__(self, cifar10_dataset_folder_path, test, batches, train_mean=None, train_std=None):
        self.samples = []
        self.mean = None
        self.std = None

        if test:
            xs, ys = load_cfar10_test(cifar10_dataset_folder_path)
            xs_np = np.array(xs) / 255
            xs = normalize_img(xs_np, train_mean, train_std)
        else:
            xs, ys = load_cfar10_train(cifar10_dataset_folder_path, batches)

            # normalize training data & save mean, std for testing data
            xs_np = np.array(xs) / 255
            self.mean = xs_np.mean(axis=(0, 2, 3)).tolist()
            self.std = xs_np.std(axis=(0, 2, 3)).tolist()
            xs = normalize_img(xs_np, self.mean, self.std)

        for data in zip(xs, ys):
            self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
