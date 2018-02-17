import os
import argparse
import json
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from models import SimpleConvNet
from metrics import SegmentationMetrics, flatten_metrics
from evaluate import evaluate

import datasets
from utils import prompt_delete_dir
from logger import StepLogger

parser = argparse.ArgumentParser(description='PyTorch Segmentation Framework')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to the dataset root dir.')
parser.add_argument('--model-dir', type=str, required=True,
                    help='Path to store output data.')
parser.add_argument('--dataset', type=str, required=True,
                    help=('A dataset handler. Required to be defined inside'
                          'datasets.handler module'))
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of CPU data workers')
parser.add_argument('--evaluate-frequency', type=int, default=1000,
                    required=False, help='how often to evaluate the model')
parser.add_argument('--checkpoint', type=str,
                    required=False,
                    help='Load model on checkpoint.')


def save(model, optimizer, model_dir, epoch, args):
    save_dict = {
        'state': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'args': args.__dict__
    }
    torch.save(
        save_dict,
        os.path.join(model_dir, 'checkpoint-{}'.format(epoch)))


def restore(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def train_epoch(
        model, loader, criterion, optimizer, epoch, logger,
        summary_writer):
    model.train()

    metrics = SegmentationMetrics(256)

    for step, (data, target) in enumerate(loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        output = output.data.numpy()
        predictions = np.argmax(output, axis=2)

        metrics_dict = metrics.add(
            predictions,
            target.data.numpy(),
            float(loss.data.numpy()[0]))

        if step % args.log_interval == 0:
            for k, v in flatten_metrics(metrics_dict).items():
                summary_writer.add_scalar(
                    "train/{}".format(k), v, step)
            metrics_dict.pop('class')
            logger.log(step, epoch, loader, data, metrics_dict)


def train(args):
    prompt_delete_dir(args.model_dir)
    os.makedirs(args.model_dir)

    # store args
    with open(os.path.join(args.model_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # seed torch and cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # initialize dataset
    assert args.dataset in datasets.handlers.__dict__, (
        "Handler for dataset {} not available.".format(args.dataset))

    # train dataset loading
    train_dataset = datasets.handlers.__dict__[args.dataset](
        args.data_dir, mode='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    # validation dataset loading
    validate_dataset = datasets.handlers.__dict__[args.dataset](
        args.data_dir, mode='val')

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    summary_writer = SummaryWriter(log_dir=args.model_dir)

    # initialize model
    model = SimpleConvNet(n_classes=256)

    # transfer to cuda
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters())

    start_epoch = 0
    if args.checkpoint:
        start_epoch = restore(args.checkpoint, model, optimizer)

    criterion = torch.nn.NLLLoss()
    if args.cuda:
        criterion = criterion.cuda()

    log_filepath = os.path.join(args.model_dir, 'train.log')

    with StepLogger(filename=log_filepath) as logger:
        for epoch in range(start_epoch, args.epochs):
            train_epoch(
                model, train_loader, criterion, optimizer, epoch, logger,
                summary_writer)
            evaluate(
                model, validate_loader, criterion, logger, epoch,
                summary_writer)
            save(model, optimizer, args.model_dir, epoch)


if '__main__' == __name__:
    args = parser.parse_args()
    train(args)
