from __future__ import print_function
from PIL import Image
import csv
import os
import numpy as np

import torch.utils.data as data
from dataloader import util
from torchvision.datasets.utils import download_url, check_integrity


class RSSCN7(data.Dataset):
	"""`RSSCN7 <https://www.dropbox.com/s/j80iv1a0mvhonsa/RSSCN7.zip?dl=0>`_ Dataset.

	Args:
		root (string): Root directory of dataset where directory
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): Currently not available. [If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.]

	"""

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

		# csv_path = os.path.join(self.root, '{}.csv'.format(train))
		#
		# data_label_dict = self.parse_csv(csv_path)
		# self.data_label_dict = list(
		# 	map(lambda x: (os.path.join(self.root, x[0]), x[1]), data_label_dict))

		if self.create_lmdb:
			basename = os.path.basename(self.root) + '_train.lmdb'
			lmdb_save_path = os.path.join(self.root, basename)
			if not os.path.exists(lmdb_save_path):
				util.create_lmdb(self.data_label_dict, lmdb_save_path)

			self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path)

		# if opt['resample']:
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
		else:
			img = util._read_lmdb_img(self.env, img_path)
			# img, target = self.test_data[index], self.test_labels[index]

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
