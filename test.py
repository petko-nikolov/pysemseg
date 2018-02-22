import os
import argparse
import json
import tqdm

import torch
from torch.autograd import Variable
import numpy as np

import datasets
from utils import prompt_delete_dir, restore
from models import SimpleConvNet


parser = argparse.ArgumentParser(
    description='PyTorch Segmentation Framework - Test')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to the dataset root dir.')
parser.add_argument('--dataset', type=str, required=True,
                    help=('A dataset handler. Required to be defined inside'
                          'datasets.handler module'))
parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to the output directory.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of CPU data workers')
parser.add_argument('--checkpoint', type=str,
                    required=True,
                    help='Load model on checkpoint.')


def test(args):
    prompt_delete_dir(args.output_dir)
    os.makedirs(args.output_dir)

    assert os.path.exists(args.checkpoint), (
        "Chechkpoint does not exists")

    # store args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # should we use cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # initialize dataset
    assert args.dataset in datasets.handlers.__dict__, (
        "Handler for dataset {} not available.".format(args.dataset))

    # test dataset loading
    dataset = datasets.handlers.__dict__[args.dataset](
        args.data_dir, mode='test')

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    # initialize model
    model = SimpleConvNet(n_classes=dataset.number_of_classes)

    # transfer to cuda
    if args.cuda:
        model.cuda()

    criterion = torch.nn.NLLLoss()
    if args.cuda:
        criterion = criterion.cuda()

    restore(args.checkpoint, model)

    model.eval()

    with tqdm.tqdm(total=len(dataset)) as pbar:
        for _, (ids, data) in tqdm.tqdm(enumerate(loader)):
            if args.cuda:
                data = data.cuda()

            data = Variable(data)
            output = model(data)

            output = output.data.numpy()
            predictions = np.argmax(output, axis=2)

            for i in range(len(ids)):
                np.savez(
                    os.path.join(args.output_dir, ids[i]), predictions[i])

            pbar.update(output.shape[0])

    return predictions

if '__main__' == __name__:
    args = parser.parse_args()
    test(args)
