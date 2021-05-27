import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed

from torch.autograd import Variable
from model import NetworkImageNet as Network


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,help='url used to set up distributed training')
parser.add_argument('--world-size', default=1, type=int,help='number of nodes for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss
 
def main():
  # if args.seed is not None:
  #       random.seed(args.seed)
  #       torch.manual_seed(args.seed)
  #       cudnn.deterministic = True
  #       warnings.warn('You have chosen to seed training. '
  #                     'This will turn on the CUDNN deterministic setting, '
  #                     'which can slow down your training considerably! '
  #                     'You may see unexpected behavior when restarting '
  #                     'from checkpoints.')

  
  args.distributed = args.world_size > 1 or args.multiprocessing_distributed
  
  ngpus_per_node = torch.cuda.device_count()# the num of gpus

  if args.multiprocessing_distributed:
    args.world_size=ngpus_per_node*args.world_size
    
    mp.spawn(main_worker,nprocs=ngpus_per_node, args=(ngpus_per_node,args))
    #spawn to launch distributed processes
  else:
    main_worker(args.gpu, ngpus_per_node,args)


def main_worker(gpu,ngpus_per_node, args):
  args.gpu=gpu

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  """
  'You have chosen to seed training. '
  'This will turn on the CUDNN deterministic setting, '
  'which can slow down your training considerably! '
  'You may see unexpected behavior when restarting '
  'from checkpoints.'
  """
  if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed(args.seed)

  torch.cuda.set_device(args.gpu)# use the specific gpu
  cudnn.benchmark = True
  cudnn.enabled=True
  """
  """
  if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("back_end:",args.dist_backend)
        print("init_method:",args.dist_url)
        print("world_size:",args.world_size)
        print("rank:",args.rank)

  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  
  if args.parallel:#we do not need divide and allocate batch_size to all available GPUs as it is not the fastest way
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda(args.gpu)
  
  if args.gpu is not None:
    args.batch_size=int(args.batch_size/ngpus_per_node)
    args.workers=int((args.workers+ngpus_per_node-1)/ngpus_per_node)#for what?
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])


  

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))
  
  if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
  else:
        train_sampler = None


  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, 
    num_workers=args.workers, sampler=train_sampler)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  best_acc_top1 = 0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
    logging.info('train_acc %f', train_acc)

    with torch.no_grad:
      valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc_top1 %f', valid_acc_top1)
      logging.info('valid_acc_top5 %f', valid_acc_top5)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    if not args.multiprocessing_distributed or(args.multiprocessing_distributed and args.rand %ngpus_per_node==0):
      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc_top1': best_acc_top1,
        'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

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

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
