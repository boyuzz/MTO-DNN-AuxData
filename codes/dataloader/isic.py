from __future__ import print_function
from PIL import Image
import numpy as np
import os
import csv

from dataloader import util
import torch.utils.data as data


class ISIC(data.Dataset):
	"""`OxfordFlower <http://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

	Args:
		root (string): Root directory of dataset where directory
			``cifar-10-batches-py`` exists or will be saved to if download is set to True.
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.

	"""
	# data_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
	# label_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
	# datasplit_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

	# data_filename = "102flowers.tgz"
	train_label = "train_fold0.csv"
	# val_label = "train_fold0.csv"
	test_label = "val_fold0.csv"
	data_folder = "ISIC2018_Task3_Training_Input"

	# data_md5 = '52808999861908f626f3c1f4e79d11fa'
	# label_md5 = 'e0620be6f572b9609742df49c70aed4d'
	# datasplit_md5 = 'a5357ecc9cb78c4bef273ce3793fc85c'

	def __init__(self, opt, train='train',
				 transform=None, target_transform=None):
		self.root = os.path.expanduser(opt['dataroot'])
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # training set or test set
		self.create_lmdb = opt['lmdb']
		lmdb_ext = '_{}.lmdb'.format(train)

		# if download:
		# 	self.download()

		# if not self._check_integrity():
		# 	raise RuntimeError('Dataset not found or corrupted.' +
		# 					   ' You can use download=True to download it')

		# im_list = glob.glob(os.path.join(self.root, 'jpg')+'/*.jpg')
		self.data_path = os.path.join(self.root, self.data_folder)

		if self.train == 'train':
			label_path = os.path.join(self.root, self.train_label)
		# elif self.train == 'val':
		# 	label_path = os.path.join(self.root, self.val_label)
		else:
			label_path = os.path.join(self.root, self.test_label)

		data_label_dict = self.parse_csv(label_path)
		self.data_label_dict = list(
			map(lambda x: (os.path.join(self.data_path, x[0]+'.jpg'),x[1]), data_label_dict))

		if self.create_lmdb:
			basename = os.path.basename(self.root) + lmdb_ext
			lmdb_save_path = os.path.join(self.root, basename)
			if not os.path.exists(lmdb_save_path):
				util.create_lmdb(self.data_label_dict, lmdb_save_path)

			self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path)

		# if self.train == 'train' and opt['resample']:
		# 	re_index = np.random.randint(0, len(self.data_label_dict), len(self.data_label_dict))
		# 	self.data_label_dict = list(map(lambda x: self.data_label_dict[x], re_index))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""

		img_path, target = self.data_label_dict[index]
		if not self.create_lmdb:
			img = Image.open(img_path)
			# img = img_path
		else:
			img = util._read_lmdb_img(self.env, img_path)

		target = int(target)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data_label_dict)

	def parse_csv(self, path):
		label_dict = []
		with open(path, 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			for row in spamreader:
				label_dict.append(row)
		return label_dict

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		tmp = 'train' if self.train is True else 'test'
		fmt_str += '    Split: {}\n'.format(tmp)
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str


# class OxfordFlower_test(OxfordFlower):
# 	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#
# 	This is a subclass of the `OxfordFlower` Dataset.
# 	"""
# 	base_folder = 'cifar-100-python'
# 	url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# 	filename = "cifar-100-python.tar.gz"
# 	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
# 	train_list = [
# 		['train', '16019d7e3df5f24257cddd939b257f8d'],
# 	]
#
# 	test_list = [
# 		['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
# 	]
