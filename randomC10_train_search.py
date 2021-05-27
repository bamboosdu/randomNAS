import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import randomCIFAR
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--warm_up_epochs', type=int, default=5, help='num of warm up epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--label_corrupt_prob', type=float, default=0, help='label corrupt prob')
parser.add_argument('--resume', type=bool, default=False, help='resume from previous training checkpoint')
parser.add_argument('--num_class', type=int, default=10, help='num of random label classes')
parser.add_argument('--data_aug', type=int, default=0, help='0: Horizontal Flip, 1: Random Crop, 2: Random Rotation, 3: Add Gaussian Noise, 4: Contrast, 5: Gaussian Blur, 6: Random Noise, 7: Only Normalize, 8: weak noise')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = args.num_class


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.data_aug == 0:
        train_transform, valid_transform = utils._flip_data_transforms_cifar10(args)
        print("Data aug is random flip")
    elif args.data_aug == 1:
        train_transform, valid_transform = utils._crop_data_transforms_cifar10(args)
        print("Data aug is random crop")
    elif args.data_aug == 2:
        train_transform, valid_transform = utils._rotation_data_transforms_cifar10(args)
        print("Data aug is random rotation")
    elif args.data_aug == 3:
        train_transform, valid_transform = utils._noise_data_transforms_cifar10(args)
        print("Data aug is low level noise injection")
    elif args.data_aug == 4:
        train_transform, valid_transform = utils._contrast_data_transforms_cifar10(args)
        print("Data aug is contrast")
    elif args.data_aug == 5:
        train_transform, valid_transform = utils._blur_data_transforms_cifar10(args)
        print("Data aug is gaussian blur")
    elif args.data_aug == 6:
        train_transform, valid_transform = utils._random_noise_transforms_cifar10(args)
        print("Data aug is only random noise")
    elif args.data_aug == 7:
        train_transform, valid_transform = utils._plain_data_transforms_cifar10(args)
        print("Data aug is no aug")
    elif args.data_aug == 8:
        train_transform, valid_transform = utils._weak_noise_data_transforms_cifar10(args)
        print("Data aug is weak gaussian noise")

    train_data = randomCIFAR.CIFAR10RandomLabels(root=args.data, train=True, download=True, transform=train_transform,
                                                 corrupt_prob=args.label_corrupt_prob, num_classes=args.num_class)

    val_data = randomCIFAR.CIFAR10RandomLabels(root=args.data, train=True, download=True, transform=valid_transform,
                                                 corrupt_prob=args.label_corrupt_prob, num_classes=args.num_class)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs + args.warm_up_epochs):

        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        logging.info('normal_alphas = %s', F.softmax(model.alphas_normal, dim=-1))
        logging.info('reduce_alphas = %s', F.softmax(model.alphas_reduce, dim=-1))
        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, args.warm_up_epochs)
        scheduler.step()
        logging.info('train_acc %f', train_acc)

        # validation
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, warm_up_epochs):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        if epoch >= warm_up_epochs:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, min(5,args.num_class)))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, min(5,args.num_class)))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

