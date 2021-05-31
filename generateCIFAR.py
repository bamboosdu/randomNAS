"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch

import torchvision.datasets as datasets

from torchvision.datasets.vision import VisionDataset

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import random


class GenCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
 
    base_folder='../data/FracDB-100-shuffle'
    dir_number =['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010',
                      '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021',
                      '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032',
                      '00033', '00034', '00035', '00036', '00037', '00038', '00039', '00040', '00041', '00042', '00043',
                      '00044', '00045', '00046', '00047', '00048', '00049', '00050', '00051', '00052', '00053', '00054',
                      '00055', '00056', '00057', '00058', '00059', '00060', '00061', '00062', '00063', '00064', '00065',
                      '00066', '00067', '00068', '00069', '00070', '00071', '00072', '00073', '00074', '00075', '00076',
                      '00077', '00078', '00079', '00080', '00081', '00082', '00083', '00084', '00085', '00086', '00087',
                      '00088', '00089', '00090', '00091', '00092', '00093', '00094', '00095', '00096', '00097', '00098',
                      '00099']

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(GenCIFAR10, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        self.train = train  # training set or test set


        self.data: Any = []
        self.targets = []


        for file_name in self.dir_number:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            pic_paths = os.listdir(file_path)
            for pic_path in pic_paths:
                pic_path_full = os.path.join(file_path, pic_path)
                im = Image.open(pic_path_full)
                im_array = np.array(im)
                self.data.append(im_array)
                self.targets.append(int(file_name))

        shuffle_data = list(zip(self.data, self.targets))
        random.shuffle(shuffle_data)
        self.data[:], self.targets[:] = zip(*shuffle_data)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class GenCIFAR100(GenCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = '../data/FracDB-100-shuffle'
    dir_number = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010',
                  '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021',
                  '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032',
                  '00033', '00034', '00035', '00036', '00037', '00038', '00039', '00040', '00041', '00042', '00043',
                  '00044', '00045', '00046', '00047', '00048', '00049', '00050', '00051', '00052', '00053', '00054',
                  '00055', '00056', '00057', '00058', '00059', '00060', '00061', '00062', '00063', '00064', '00065',
                  '00066', '00067', '00068', '00069', '00070', '00071', '00072', '00073', '00074', '00075', '00076',
                  '00077', '00078', '00079', '00080', '00081', '00082', '00083', '00084', '00085', '00086', '00087',
                  '00088', '00089', '00090', '00091', '00092', '00093', '00094', '00095', '00096', '00097', '00098',
                  '00099']


class CIFAR10GenLabels(GenCIFAR10):
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
        super(CIFAR10GenLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]


        self.targets = labels


class CIFAR100GenLabels(GenCIFAR100):
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
        super(CIFAR100GenLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels

        labels = [int(x) for x in labels]

        self.targets = labels



class GenCIFAR1060000(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder='FracDB-100-60000-shuffle'
    dir_number =['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010',
                      '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021',
                      '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032',
                      '00033', '00034', '00035', '00036', '00037', '00038', '00039', '00040', '00041', '00042', '00043',
                      '00044', '00045', '00046', '00047', '00048', '00049', '00050', '00051', '00052', '00053', '00054',
                      '00055', '00056', '00057', '00058', '00059', '00060', '00061', '00062', '00063', '00064', '00065',
                      '00066', '00067', '00068', '00069', '00070', '00071', '00072', '00073', '00074', '00075', '00076',
                      '00077', '00078', '00079', '00080', '00081', '00082', '00083', '00084', '00085', '00086', '00087',
                      '00088', '00089', '00090', '00091', '00092', '00093', '00094', '00095', '00096', '00097', '00098',
                      '00099']

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(GenCIFAR1060000, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []


        self.split = 5
        for file_name in self.dir_number:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            pic_paths = os.listdir(file_path)
            pic_number = len(pic_paths)
            pic_number = pic_number / (self.split + 1)
            if self.train:
                for pic_path in pic_paths[:int(pic_number * self.split)]:
                    pic_path_full = os.path.join(file_path, pic_path)
                    im = Image.open(pic_path_full)
                    im_array = np.array(im)
                    self.data.append(im_array)
                    self.targets.append(int(file_name))
            else:
                for pic_path in pic_paths[int(pic_number * self.split):]:
                    pic_path_full = os.path.join(file_path, pic_path)
                    im = Image.open(pic_path_full)
                    im_array = np.array(im)
                    self.data.append(im_array)
                    self.targets.append(int(file_name))



        shuffle_data = list(zip(self.data, self.targets))
        random.shuffle(shuffle_data)
        self.data[:], self.targets[:] = zip(*shuffle_data)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class GenCIFAR10060000(GenCIFAR1060000):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = 'FracDB-100-60000-shuffle'
    dir_number = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010',
                  '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021',
                  '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032',
                  '00033', '00034', '00035', '00036', '00037', '00038', '00039', '00040', '00041', '00042', '00043',
                  '00044', '00045', '00046', '00047', '00048', '00049', '00050', '00051', '00052', '00053', '00054',
                  '00055', '00056', '00057', '00058', '00059', '00060', '00061', '00062', '00063', '00064', '00065',
                  '00066', '00067', '00068', '00069', '00070', '00071', '00072', '00073', '00074', '00075', '00076',
                  '00077', '00078', '00079', '00080', '00081', '00082', '00083', '00084', '00085', '00086', '00087',
                  '00088', '00089', '00090', '00091', '00092', '00093', '00094', '00095', '00096', '00097', '00098',
                  '00099']


class CIFAR10GenLabels60000(GenCIFAR1060000):
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
        super(CIFAR10GenLabels60000, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]


        self.targets = labels


class CIFAR100GenLabels60000(GenCIFAR10060000):
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
        super(CIFAR100GenLabels60000, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]


        self.targets = labels


class GenCIFAR10Ori56(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """


    base_folder = 'FractalDB-50-shuffle'
    dir_number = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010',
                  '00011',
                  '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021', '00022',
                  '00023',
                  '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032', '00033', '00034',
                  '00035',
                  '00036', '00037', '00038', '00039', '00040', '00041', '00042', '00043', '00044', '00045', '00046',
                  '00047',
                  '00048', '00049']
 

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(GenCIFAR10Ori56, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        self.train = train  # training set or test set

      

        self.data: Any = []
        self.targets = []


        self.split = 5
        for file_name in self.dir_number:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            pic_paths = os.listdir(file_path)
            pic_number = len(pic_paths)
            pic_number = pic_number / (self.split + 1)
            if self.train:
                for pic_path in pic_paths[:int(pic_number * self.split)]:
                    pic_path_full = os.path.join(file_path, pic_path)
                    im = Image.open(pic_path_full)
                    im_array = np.array(im)
                    self.data.append(im_array)
                    self.targets.append(int(file_name))
            else:
                for pic_path in pic_paths[int(pic_number * self.split):]:
                    pic_path_full = os.path.join(file_path, pic_path)
                    im = Image.open(pic_path_full)
                    im_array = np.array(im)
                    self.data.append(im_array)
                    self.targets.append(int(file_name))


        shuffle_data = list(zip(self.data, self.targets))
        random.shuffle(shuffle_data)
        self.data[:], self.targets[:] = zip(*shuffle_data)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # self._load_meta()

  

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



class GenCIFAR100Ori56(GenCIFAR10Ori56):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    

    base_folder = 'FractalDB-50-shuffle'
    dir_number = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010',
                  '00011',
                  '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021', '00022',
                  '00023',
                  '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032', '00033', '00034',
                  '00035',
                  '00036', '00037', '00038', '00039', '00040', '00041', '00042', '00043', '00044', '00045', '00046',
                  '00047',
                  '00048', '00049']



class CIFAR10GenLabelsOri56(GenCIFAR10Ori56):
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
        super(CIFAR10GenLabelsOri56, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        # if self.train:
        # self.train_labels = labels
        # else:
        # self.test_labels = labels
        # self.targets = labels

        self.targets = labels


class CIFAR100GenLabelsOri56(GenCIFAR100Ori56):
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
        super(CIFAR100GenLabelsOri56, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # labels = np.array(self.train_labels if self.train else self.test_labels)
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        # if self.train:
        # self.train_labels = labels
        # else:
        # self.test_labels = labels
        # self.targets = labels

        self.targets = labels



