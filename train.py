import os
import sys
import argparse
import shutil

import torch
import torch.optim as optim
from torch.autograd import Variable

from models import SimpleConvNet
from metrics import SegmentationMetrics
import datasets
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


def train(model, loader, criterion, epoch, logger):
    model.train()

    metrics = SegmentationMetrics(2)

    for step, (data, target) in enumerate(loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        metrics_dict = metrics.add(
            output.data.numpy(),
            target.data.numpy(),
            float(loss.data.numpy()[0]))

        if step % args.log_interval == 0:
            logger.log(step, epoch, loader, data, metrics_dict)

if '__main__' == __name__:
    args = parser.parse_args()

    if os.path.exists(args.model_dir):
        answer = input("Model dir exists. Do you want to delete it?[y/n]")
        if answer == 'y':
            shutil.rmtree(args.model_dir)
        elif answer != 'n':
            sys.exit(1)

    os.makedirs(args.model_dir)

    # seed torch and cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # initialize model
    model = SimpleConvNet()

    # initialize dataset
    assert args.dataset in datasets.handlers.__dict__, (
        "Handler for dataset {} not available.".format(args.dataset))

    train_dataset = datasets.handlers.__dict__[args.dataset](
        args.data_dir, mode='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    # transfer to cuda
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters())

    criterion = torch.nn.BCELoss()
    if args.cuda:
        criterion = criterion.cuda()

    log_filepath = os.path.join(args.model_dir, 'out.log')

    with StepLogger(filename=log_filepath) as logger:
        for epoch in range(args.epochs):
            train(model, train_loader, criterion, epoch, logger)
