from __future__ import print_function
from PIL import Image
import os
# import numpy as np

import torch.utils.data as data
from dataloader import util
from utils.util import list_all_files
# from torchvision.datasets.utils import download_url, check_integrity


class ImgReader(data.Dataset):
	def __init__(self, opt, transform=None):
		self.root = os.path.expanduser(opt['dataroot'])
		self.transform = transform
		self.create_lmdb = False #opt['lmdb']

		self.img_list = list_all_files(self.root)

		if self.create_lmdb:
			basename = os.path.basename(self.root) + '_train.lmdb'
			lmdb_save_path = os.path.join(self.root, basename)
			if not os.path.exists(lmdb_save_path):
				util.create_lmdb(self.img_list, lmdb_save_path)

			self.env, self.img_list = util._get_paths_from_lmdb(lmdb_save_path)

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img_path = self.img_list[index]

		if not self.create_lmdb:
			img = Image.open(img_path)
		else:
			img = util._read_lmdb_img(self.env, img_path)
			# img, target = self.test_data[index], self.test_labels[index]

		if self.transform is not None:
			img = self.transform(img)

		return img

	def __len__(self):
		return len(self.img_list)
