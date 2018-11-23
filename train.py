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
    flatten_dict, get_latest_checkpoint, save)
from logger import ConsoleLogger


def define_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Framework')
    parser.add_argument('--model', type=str, required=True,
                        help=('A path to the model including the module. '
                              'Should be resolvable'))
    parser.add_argument('--model_args', type=json.loads, required=False, default={},
                        help=('Args passed to the model constructor'))
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
                        default='lr_schedulers.ConstantLR',
                        help='Learning rate scheduler type.')
    parser.add_argument('--lr_scheduler_args', type=json.loads, default={},
                        required=False,
                        help='Learning rate scheduler args.')
    parser.add_argument('--transformer', type=str, required=False,
                        help='Transformer type')
    parser.add_argument('--transformer_args', type=json.loads, default={},
                        required=False,
                        help='Transformer args.')
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
                              'weight one'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--allow-missing-keys', action='store_true', default=False,
                        help='Whether to allow module keys to differ from checkpoint keys'
                             ' when loading a checkpoint')
    group.add_argument('--continue_training', action='store_true', default=False,
                       help='Continue experiment from the last checkpoint in the model dir')
    return parser


def train_epoch(
        model, loader, criterion, optimizer, lr_scheduler,
        epoch, console_logger, visual_logger, cuda, log_interval):
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
        if cuda:
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

        if step % log_interval == 0:

            metrics_dict = metrics.metrics()
            metrics_dict['time'] = time.time() - start_time

            metrics_dict.pop('class')
            console_logger.log(step, epoch, loader, data, metrics_dict)

            metrics = SegmentationMetrics(
                loader.dataset.number_of_classes,
                loader.dataset.labels,
                ignore_index=loader.dataset.ignore_index
            )

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


def _create_data_loaders(
        data_dir, dataset_cls, transformer_cls, transformer_args,
        train_batch_size, val_batch_size, num_workers):
    train_dataset = datasets.create_dataset(
        data_dir, dataset_cls, transformer_cls, transformer_args, mode='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        shuffle=True, num_workers=num_workers)

    validate_dataset = datasets.create_dataset(
        data_dir, dataset_cls, transformer_cls, transformer_args, mode='val')

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers)

    return train_loader, validate_loader


def _store_args(args, model_dir):
    with open(os.path.join(model_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)


def _set_seed(seed, cuda):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def train(args):
    if not args.continue_training:
        prompt_delete_dir(args.model_dir)
        os.makedirs(args.model_dir)

    _store_args(args, args.model_dir)

    # seed torch and cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    _set_seed(args.seed, args.cuda)

    dataset_cls = import_class_module(args.dataset)
    transformer_cls = import_class_module(args.transformer)

    train_loader, validate_loader = _create_data_loaders(
        args.data_dir, dataset_cls, transformer_cls,  args.transformer_args,
        args.batch_size,  args.test_batch_size, args.num_workers
    )

    visual_logger = VisdomLogger(
        log_directory=args.model_dir,
        color_palette=train_loader.dataset.color_palette,
        continue_logging=args.continue_training
    )

    visual_logger.log_args(args.__dict__)

    model_class = import_class_module(args.model)
    model = model_class(
        in_channels=train_loader.dataset.in_channels,
        n_classes=train_loader.dataset.number_of_classes
    )

    weights = (
        _get_class_weights(args.weights, train_dataset.number_of_classes,
                           args.cuda)
        if args.weights else None
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, reduction=args.loss_reduction,
        ignore_index=train_loader.dataset.ignore_index
    )

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer_class = import_class_module('torch.optim.' + args.optimizer)
    optimizer = optimizer_class(
        model.parameters(), lr=args.lr, **args.optimizer_args
    )

    start_epoch = 0

    if args.continue_training:
        args.checkpoint = get_latest_checkpoint(args.model_dir)
        assert args.checkpoint is not None

    lr_scheduler_cls = import_class_module(args.lr_scheduler)
    lr_scheduler = lr_scheduler_cls(optimizer, **args.lr_scheduler_args)

    if args.checkpoint:
        start_epoch = restore(
            args.checkpoint, model, optimizer, lr_scheduler,
            strict=not args.allow_missing_keys) + 1

    log_filepath = os.path.join(args.model_dir, 'train.log')

    with ConsoleLogger(filename=log_filepath) as logger:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_epoch(
                model, train_loader, criterion, optimizer, lr_scheduler,
                epoch, logger, visual_logger, args.cuda, args.log_interval)
            evaluate(
                model, validate_loader, criterion, logger, epoch,
                visual_logger, args.cuda)
            if epoch % args.save_model_frequency == 0:
                save(model, optimizer, lr_scheduler, args.model_dir,
                     train_loader.dataset.in_channels,
                     train_loader.dataset.number_of_classes, epoch, args)
            lr_scheduler.step()


def main():
    parser = define_args()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
