from model.batch import generate_batch
from model.model import Cifar10
from data_loader.data_loader import Cifar10DataSet
from utils.utils import plot_results

import numpy as np
import random
import argparse
import json
import os
import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


parser = argparse.ArgumentParser(description="Train CIFAR 10 model")
parser.add_argument('--config_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    losses = []
    prediction_correct = 0
    data_size = 0

    for i, batch in enumerate(data_loader):
        data_size += batch.size
        batch.to_device(device)
        out = model(batch.xs)
        loss = criterion(out, batch.ys)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, predicted = torch.max(out.data, 1)
        prediction_correct += (predicted == batch.ys).sum().item()

    return np.sum(losses)/data_size, prediction_correct/data_size


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
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    save_dir = config['save_dir']
    device = config['device']

    set_seed(config['seed'])

    data = Cifar10DataSet(data_dir, test=False, batches=config['train_batch'])
    config['mean'] = data.mean
    config['std'] = data.std

    num_data = len(data)
    num_val = round(num_data * config['split_ratio'])
    num_train = num_data - num_val

    train_data, val_data = random_split(data, [num_train, num_val])

    tr_data_loader = DataLoader(train_data, batch_size=config['tr_batch_size'], shuffle=True, collate_fn=generate_batch)
    val_data_loader = DataLoader(val_data, batch_size=config['te_batch_size'], shuffle=False, collate_fn=generate_batch)

    save_model_dir = os.path.join(save_dir, 'model')
    log_dir = os.path.join(save_dir, 'log')
    config_dir = os.path.join(save_dir, 'config')

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    writer = SummaryWriter(log_dir)
    writer.add_text('training_config', json.dumps(config))
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    model = Cifar10(dropout=config['dropout'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_reg'])
    criterion = nn.CrossEntropyLoss()
    epochs = config['epochs']

    best_acc = -np.inf

    logging.info('Begin training')
    logging.info(config['msg'])
    print('Begin training')

    logging.info('Training size: {} | Validation size: {}'.format(num_train, num_val))
    print('Training size: {} | Validation size: {}'.format(num_train, num_val))
    try:
        tr_losses = []
        val_losses = []
        tr_accuracy = []
        val_accuracy = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            tr_loss, tr_acc = train(tr_data_loader, model, criterion, optimizer, device)
            val_loss, val_acc = test(val_data_loader, model, criterion, device)

            writer.add_scalars('losses', {'training': tr_loss, 'validation': val_loss}, epoch)
            writer.add_scalars('acc', {'training': tr_acc, 'validation': val_acc}, epoch)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            tr_accuracy.append(tr_acc)
            val_accuracy.append(val_acc)

            if val_acc > best_acc:
                with open(os.path.join(save_model_dir, "model.pt"), 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_acc = val_acc

            logging.info('| epoch {:3d} | time {:3.2f} mins | tr_loss {:1.3f} | val_loss {:1.3f} | tr_acc {:1.3f} | val'
                         '_acc {:1.3f} | best_val_acc {:1.3f} |'.format(epoch, (time.time() - start) / 60, tr_loss,
                                                                        val_loss, tr_acc, val_acc, best_acc))
            print('| epoch {:3d} | time {:3.2f} mins | tr_loss {:1.3f} | val_loss {:1.3f} | tr_acc {:1.3f} | '
                  'val_acc {:1.3f} | best_val_acc {:1.3f} |'.format(epoch, (time.time() - start) / 60, tr_loss, val_loss
                                                                    , tr_acc, val_acc, best_acc))

        plot_results(log_dir, 'loss', tr_losses, val_losses)
        plot_results(log_dir, 'acc', tr_accuracy, val_accuracy)
        writer.close()
    except KeyboardInterrupt:
        writer.close()
        print('Exiting from training early')
