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
    # base_folder = 'cifar-10-batches-py'
    # url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # filename = "cifar-10-python.tar.gz"
    # tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    # train_list = [
    #     ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    #     ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    #     ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    #     ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    #     ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    # ]
    #
    # test_list = [
    #     ['test_batch', '40351d587109b95175f43aff81a1287e'],
    # ]
    # meta = {
    #     'filename': 'batches.meta',
    #     'key': 'label_names',
    #     'md5': '5ff9c542aee3614f3951f8cda6e48888',
    # }
    # 5:1
    # 预先把数据分好
    #     from PIL import Image

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

    # write_folder = "FractalDB-50-shuffle"
    # idx = 0
    # for file_name in dir_number:
    #     file_path = os.path.join("/rscratch/zhendong/randomNAS", base_folder, file_name)
    #     pic_paths = os.listdir(file_path)
    #     im_list = []
    #     for pic_path in pic_paths:
    #         pic_path_full = os.path.join(file_path,pic_path)
    #         im = Image.open(pic_path_full)
    #         im_list.append(im)
    #     random.shuffle(im_list)
    #     for image in im_list:
    #         image.save("/rscratch/zhendong/randomNAS/FractalDB-50-shuffle/"+file_name+"/"+("%06d"%(idx))+".png")
    #         idx += 1
    #
    # data = np.zeros((32, 32, 3))
    # for file_name in dir_number:
    #     file_path = os.path.join(file_name)
    #     pic_paths = os.listdir(file_path)
    #     for pic_path in pic_paths:
    #         pic_path_full = os.path.join(file_path, pic_path)
    #         im = Image.open(pic_path_full)
    #         im_array = np.array(im)
    #         data += im_array
    #     data /= 1000
    #     data /= 50
    # err=0
    # from sklearn.metrics import mean_squared_error, r2_score
    # import numpy as np
    #
    # rec_lis = []
    # for file_name in dir_number:
    #     file_path = os.path.join(file_name)
    #     pic_paths = os.listdir(file_path)
    #     for pic_path in pic_paths:
    #         pic_path_full = os.path.join(file_path, pic_path)
    #         im = Image.open(pic_path_full)
    #         im_array = np.array(im)
    #         err += mean_squared_error(data.reshape(32,32*3), im_array.reshape(32,32*3))
    # print(f"均方误差(MSE)：{mean_squared_error(result, test_y)}")

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

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')
        # 167*5
        # 165
        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # # now load the picked numpy arrays
        # for file_name in self.dir_number:
        #     file_path = os.path.join(self.root, self.base_folder, file_name)
        #     with open(file_path, 'rb') as f:
        #         entry = pickle.load(f, encoding='latin1')
        #         self.data.append(entry['data'])
        #         if 'labels' in entry:
        #             self.targets.extend(entry['labels'])
        #         else:
        #             self.targets.extend(entry['fine_labels'])
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

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     if not check_integrity(path, self.meta['md5']):
    #         raise RuntimeError('Dataset metadata file not found or corrupted.' +
    #                            ' You can use download=True to download it')
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

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

    # def _check_integrity(self) -> bool:
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True

    # def download(self) -> None:
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class GenCIFAR100(GenCIFAR10):
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
    # base_folder = 'cifar-100-python'
    # url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    # filename = "cifar-100-python.tar.gz"
    # tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    # train_list = [
    #     ['train', '16019d7e3df5f24257cddd939b257f8d'],
    # ]
    #
    # test_list = [
    #     ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    # ]
    # meta = {
    #     'filename': 'meta',
    #     'key': 'fine_label_names',
    #     'md5': '7973b15100ade9c7d40fb424638fde48',
    # }


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

        # if self.train:
        # self.train_labels = labels
        # else:
        # self.test_labels = labels
        # self.targets = labels

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
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        # if self.train:
        # self.train_labels = labels
        # else:
        # self.test_labels = labels
        # self.targets = labels

        self.targets = labels


class RandomNoiseCIFAR10(VisionDataset):
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

    # write_folder = "FractalDB-50-shuffle"
    # idx = 0
    # for file_name in dir_number:
    #     file_path = os.path.join("/rscratch/zhendong/randomNAS", base_folder, file_name)
    #     pic_paths = os.listdir(file_path)
    #     im_list = []
    #     for pic_path in pic_paths:
    #         pic_path_full = os.path.join(file_path,pic_path)
    #         im = Image.open(pic_path_full)
    #         im_list.append(im)
    #     random.shuffle(im_list)
    #     for image in im_list:
    #         image.save("/rscratch/zhendong/randomNAS/FractalDB-50-shuffle/"+file_name+"/"+("%06d"%(idx))+".png")
    #         idx += 1
    #
    # data = np.zeros((32, 32, 3))
    # for file_name in dir_number:
    #     file_path = os.path.join(file_name)
    #     pic_paths = os.listdir(file_path)
    #     for pic_path in pic_paths:
    #         pic_path_full = os.path.join(file_path, pic_path)
    #         im = Image.open(pic_path_full)
    #         im_array = np.array(im)
    #         data += im_array
    #     data /= 1000
    #     data /= 50
    # err=0
    # from sklearn.metrics import mean_squared_error, r2_score
    # import numpy as np
    #
    # rec_lis = []
    # for file_name in dir_number:
    #     file_path = os.path.join(file_name)
    #     pic_paths = os.listdir(file_path)
    #     for pic_path in pic_paths:
    #         pic_path_full = os.path.join(file_path, pic_path)
    #         im = Image.open(pic_path_full)
    #         im_array = np.array(im)
    #         err += mean_squared_error(data.reshape(32,32*3), im_array.reshape(32,32*3))
    # print(f"均方误差(MSE)：{mean_squared_error(result, test_y)}")

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(RandomNoiseCIFAR10, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)

        self.train = train  # training set or test set

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')
        # 167*5
        # 165
        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # # now load the picked numpy arrays
        # for file_name in self.dir_number:
        #     file_path = os.path.join(self.root, self.base_folder, file_name)
        #     with open(file_path, 'rb') as f:
        #         entry = pickle.load(f, encoding='latin1')
        #         self.data.append(entry['data'])
        #         if 'labels' in entry:
        #             self.targets.extend(entry['labels'])
        #         else:
        #             self.targets.extend(entry['fine_labels'])
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
        for i, img in enumerate(self.data):
            torch.manual_seed(12345)
            np.random.seed(12345)
            torch.cuda.manual_seed(12345)
            # print(self.data[i])
            # print(self.targets[i])
            # print(img.size())
            self.data[i] = np.array(
                img) + np.array(torch.randn(img.shape) * self.targets[i] * 0.1)
        # print(self.data[0])
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i,
                             _class in enumerate(self.classes)}

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

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class RandomNoiseCIFAR100(RandomNoiseCIFAR10):
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
    # base_folder = 'cifar-100-python'
    # url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    # filename = "cifar-100-python.tar.gz"
    # tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    # train_list = [
    #     ['train', '16019d7e3df5f24257cddd939b257f8d'],
    # ]
    #
    # test_list = [
    #     ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    # ]
    # meta = {
    #     'filename': 'meta',
    #     'key': 'fine_label_names',
    #     'md5': '7973b15100ade9c7d40fb424638fde48',
    # }


class RandomNoiseCIFAR10GenLabels(RandomNoiseCIFAR10):
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
        super(RandomNoiseCIFAR10GenLabels, self).__init__(**kwargs)
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


class RandomNoiseCIFAR100GenLabels(RandomNoiseCIFAR100):
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
        super(RandomNoiseCIFAR100GenLabels, self).__init__(**kwargs)
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
