import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .models import SimpleConvNet
import datasets


parser = argparse.ArgumentParser(description='PyTorch Segmentation Framework')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to the dataset root dir.')
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


if '__main__' == __name__:
    args = parser.parse_args()

    # seed torch and cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # initialize model
    model = SimpleConvNet()

    # initialize dataset
    assert args.dataset in datasets.func.__globals__, (
        "Handler for dataset {} not available.".format(args.dataset))

    train_dataset = datasets.func.__globals__[args.dataset](
        args.data_dir, model='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    # transfer to cuda
    if args.cuda:
        model.cuda()

    optimizer = optim.adam(model.parameters())

    criterion = torch.nn.BCELoss()
    if args.cuda:
        criterion = criterion.cuda()

    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
