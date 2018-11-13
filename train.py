import os
import argparse
import json
import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from metrics import SegmentationMetrics
from loggers import TensorboardLogger, VisdomLogger
from evaluate import evaluate

import datasets
from utils import (
    prompt_delete_dir, restore, tensor_to_numpy, import_class_module,
    flatten_dict, get_latest_checkpoint)
from logger import StepLogger


def define_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Framework')
    parser.add_argument('--model', type=str, required=True,
                        help=('A path to the model including the module. '
                              'Should be resolvable'))
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
    parser.add_argument('--optimizer', type=str, default='RMSprop',
                        required=False,
                        help='Optimizer type.')
    parser.add_argument('--optimizer_args', type=json.loads, default={},
                        required=False,
                        help='Optimizer args.')
    parser.add_argument('--lr_scheduler', type=str, required=False,
                        help='Learning rate scheduler type.')
    parser.add_argument('--lr_scheduler_args', type=str, required=False,
                        help='Learning rate scheduler args.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='logging training status frequency')
    parser.add_argument('--log-images-interval', type=int, default=200, metavar='N',
                        help='Frequency of logging images and larger plots')
    parser.add_argument('--loss_reduction', type=str, default='elementwise_mean',
                        choices=['elementwise_mean', 'sum'],
                        help='Sum or average individual pixel losses.')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of CPU data workers')
    parser.add_argument('--checkpoint', type=str,
                        required=False,
                        help='Load model on checkpoint.')
    parser.add_argument('--save-model-frequency', type=int,
                        required=False, default=5,
                        help='Save model checkpoint every nth epoch.')
    parser.add_argument('--weights', type=str, required=False,
                        help=('Class weights passed as a JSON object, e.g:'
                              '{"0": 5.0, "2": 3.0}, missing classes get'
                              'weight one one'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--allow-missing-keys', action='store_true', default=False,
                        help='Whether to allow module keys to differ from checkpoint keys'
                             ' when loading a checkpoint')
    group.add_argument('--continue_training', action='store_true', default=False,
                       help='Continue experiment from the last checkpoint in the model dir')
    return parser


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


def train_epoch(
        model, loader, criterion, optimizer, lr_scheduler,
        epoch, console_logger, visual_logger):
    model.train()

    metrics = SegmentationMetrics(
        loader.dataset.number_of_classes,
        loader.dataset.labels,
        ignore_index=loader.dataset.ignore_index
    )
    epoch_metrics = SegmentationMetrics(
        loader.dataset.number_of_classes,
        loader.dataset.labels,
        ignore_index=loader.dataset.ignore_index)

    for step, (ids, data, target) in enumerate(loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        start_time = time.time()

        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        output = F.softmax(output, dim=1)
        output, target, loss = [
            tensor_to_numpy(t.data) for t in [output, target, loss]
        ]
        predictions = np.argmax(output, axis=1)

        if criterion.reduction == 'sum':
            if loader.dataset.ignore_index:
                loss = loss / np.sum(target != loader.dataset.ignore_index)
            else:
                loss = loss / np.prod(target.shape)

        metrics.add(predictions, target, float(loss))

        epoch_metrics.add(predictions, target, float(loss))

        if step % args.log_interval == 0:

            metrics_dict = metrics.metrics()
            metrics_dict['time'] = time.time() - start_time

            metrics_dict.pop('class')
            console_logger.log(step, epoch, loader, data, metrics_dict)

            metrics = SegmentationMetrics(loader.dataset.number_of_classes)

            visual_logger.log_prediction_images(
                step,
                tensor_to_numpy(data.data),
                target,
                predictions,
                name='images',
                prefix='Train'
            )

    visual_logger.log_metrics(epoch, epoch_metrics.metrics(), 'Train')


def _get_class_weights(weights_json_str, n_classes, cuda=False):
    weights_obj = json.loads(weights_json_str)
    weights = np.ones(n_classes, dtype=np.float32)
    for k, v in weights_obj.items():
        weights[int(k)] = v
    weights = torch.FloatTensor(weights)
    if cuda:
        weights = weights.cuda()
    return weights


def train(args):
    if not args.continue_training:
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
        validate_dataset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.num_workers)

    visual_logger = VisdomLogger(
        log_directory=args.model_dir, color_palette=train_dataset.color_palette,
        continue_logging=args.continue_training)

    visual_logger.log_args(args.__dict__)

    # initialize model
    model_class = import_class_module(args.model)
    model = model_class(
        in_channels=3, n_classes=train_dataset.number_of_classes)

    # transfer to cuda
    if args.cuda:
        model = model.cuda()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(
    #    model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    optimizer_class = import_class_module('torch.optim.' + args.optimizer)
    print(args.optimizer_args)
    optimizer = optimizer_class(
        model.parameters(), lr=args.lr, **args.optimizer_args)
    start_epoch = 0

    if args.continue_training:
        args.checkpoint = get_latest_checkpoint(args.model_dir)
        assert args.checkpoint is not None

    if args.checkpoint:
        start_epoch = restore(
            args.checkpoint, model, optimizer, strict=not args.allow_missing_keys) + 1

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    weights=None
    if args.weights:
        weights = _get_class_weights(
            args.weights, train_dataset.number_of_classes, args.cuda)

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, reduction=args.loss_reduction,
        ignore_index=train_dataset.ignore_index)

    if args.cuda:
        criterion = criterion.cuda()

    log_filepath = os.path.join(args.model_dir, 'train.log')

    with StepLogger(filename=log_filepath) as logger:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_epoch(
                model, train_loader, criterion, optimizer, lr_scheduler,
                epoch, logger, visual_logger)
            evaluate(
                model, validate_loader, criterion, logger, epoch,
                visual_logger, args.cuda)
            if epoch % args.save_model_frequency == 0:
                save(model, optimizer, args.model_dir, epoch, args)
            lr_scheduler.step()


if '__main__' == __name__:
    parser = define_args()
    args = parser.parse_args()
    train(args)
