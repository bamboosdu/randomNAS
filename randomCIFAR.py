"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    #labels = np.array(self.train_labels if self.train else self.test_labels)
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    #if self.train:
      #self.train_labels = labels
    #else:
      #self.test_labels = labels
      #self.targets = labels

    self.targets = labels


class CIFAR100RandomLabels(datasets.CIFAR100):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=100, **kwargs):
    super(CIFAR100RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    #labels = np.array(self.train_labels if self.train else self.test_labels)
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    #if self.train:
      #self.train_labels = labels
    #else:
      #self.test_labels = labels
      #self.targets = labels

    self.targets = labels
#
# if __name__ == '__main__':
#     import torchvision.transforms as transforms
#     args.data = "../data"
#     args.label_corrupt_prob = 1.0
#     train_transform = transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#             ])
#     train_data = CIFAR10RandomLabels(root=args.data,
#                                train=True,
#                                download=True,
#                                transform=train_transform,
#                                corrupt_prob=args.label_corrupt_prob)