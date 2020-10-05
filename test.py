from model.batch import generate_batch
from data_loader.data_loader import Cifar10DataSet
from model.model import Cifar10

import numpy as np
import random
import argparse
import json
import os
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


parser = argparse.ArgumentParser(description="Test CIFAR 10 model")
parser.add_argument('--save_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(data_loader, model, criterion, device):
    model.eval()
    losses = []
    prediction_correct = 0
    data_size = 0

    for i, batch in enumerate(data_loader):
        data_size += batch.size
        batch.to_device(device)
        with torch.no_grad():
            out = model(batch.xs)
        loss = criterion(out, batch.ys)
        losses.append(loss.item())

        _, predicted = torch.max(out.data, 1)
        prediction_correct += (predicted == batch.ys).sum().item()

    return np.sum(losses)/data_size, prediction_correct/data_size


if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.save_path

    config = json.load(open(os.path.join(save_path, "config/config.json")))
    logging.basicConfig(filename=os.path.join(save_path, 'log/test.log'), level=logging.INFO)
    data_dir = config['data_dir']
    device = config['device']

    model = Cifar10()
    model.load_state_dict(torch.load(os.path.join(save_path, "model/model.pt")))
    model.to(device)

    set_seed(config['seed'])

    te_data = Cifar10DataSet(data_dir, test=True, batches=[], train_mean=config['mean'], train_std=config['std'])

    num_test = len(te_data)

    te_data_loader = DataLoader(te_data, batch_size=config['te_batch_size'], shuffle=False, collate_fn=generate_batch)

    criterion = nn.CrossEntropyLoss()
    epochs = config['epochs']

    print('Begin testing')
    logging.info('Begin testing')
    print('Testing size: {}'.format(num_test))
    logging.info('Testing size: {}'.format(num_test))
    try:
        te_loss, te_acc = test(te_data_loader, model, criterion, device)
        print('Acc: {:1.3f} | Loss: {:1.3f}'.format(te_acc, te_loss))
        logging.info('Acc: {:1.3f} | Loss: {:1.3f}'.format(te_acc, te_loss))
    except KeyboardInterrupt:
        print('Exiting from training early')
